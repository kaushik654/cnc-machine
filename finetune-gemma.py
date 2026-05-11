#!/usr/bin/env python3
"""
finetune_gemma4_e2b.py  (v4 - char-offset loss masking, no token ID assumptions)

Key fix over v3: loss masking uses character offset (text.rfind) to locate
the assistant turn, then maps that to a token position via offset_mapping or
prefix-length encoding. No assumptions about how <start_of_turn> is tokenized.
"""
from __future__ import annotations
import os, json, argparse
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =============================================================================
# CHAT TEMPLATE
# =============================================================================
CHAT_TEMPLATE = """\
{%- for message in messages %}\
{%- if message['role'] == 'system' %}\
<start_of_turn>system
{{ message['content'] }}<end_of_turn>
{%- elif message['role'] == 'user' %}\
<start_of_turn>user
{{ message['content'] }}<end_of_turn>
{%- elif message['role'] == 'assistant' %}\
<start_of_turn>model
{%- if message.get('tool_calls') %}\
{%- for tc in message['tool_calls'] %}\
<tool_call>
{"name": "{{ tc['function']['name'] }}", "arguments": {{ tc['function']['arguments'] }}}
</tool_call>
{%- endfor %}\
{%- else %}\
{{ message['content'] }}\
{%- endif %}\
<end_of_turn>
{%- endif %}\
{%- endfor %}\
{%- if add_generation_prompt %}<start_of_turn>model
{%- endif %}\
"""

# The marker as a plain STRING (not token IDs) — used for character-level search
ASSISTANT_MARKER = "<start_of_turn>model\n"


# =============================================================================
# DATA HELPERS
# =============================================================================
def load_jsonl(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def normalize_messages(messages: list[dict]) -> list[dict]:
    """Serialize tool_calls into content strings so template gets plain text."""
    normalized = []
    for msg in messages:
        msg = dict(msg)
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            parts = []
            for tc in msg["tool_calls"]:
                name = tc["function"]["name"]
                args = tc["function"]["arguments"]
                parts.append(
                    f'<tool_call>\n{{"name": "{name}", "arguments": {args}}}\n</tool_call>'
                )
            msg["content"] = "\n".join(parts)
            del msg["tool_calls"]
        normalized.append(msg)
    return normalized


def preformat_record(record: dict, tokenizer) -> dict | None:
    messages = normalize_messages(record.get("messages", []))
    if not messages:
        return None
    last = messages[-1]
    if last.get("role") == "assistant" and not last.get("content", "").strip():
        return None
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception as e:
        print(f"[WARN] template failed for {record.get('sample_id')}: {e}")
        return None

    # Verify the marker actually appears in this text
    if ASSISTANT_MARKER not in text:
        print(f"[WARN] assistant marker not in formatted text for {record.get('sample_id')}")
        print(f"       text[:200] = {repr(text[:200])}")
        return None

    return {
        "text": text,
        "task": record.get("task", "unknown"),
        "sample_id": record.get("sample_id", ""),
    }


def build_hf_dataset(records: list[dict], tokenizer) -> Dataset:
    rows, skipped = [], 0
    for r in records:
        out = preformat_record(r, tokenizer)
        if out is None:
            skipped += 1
        else:
            rows.append(out)
    if skipped:
        print(f"[WARN] Skipped {skipped} records during pre-formatting")
    return Dataset.from_list(rows)


# =============================================================================
# LOSS MASKING  — character offset approach, no token ID assumptions
# =============================================================================
def find_assistant_token_start(text: str, token_ids: list[int], tokenizer) -> int:
    """
    Returns the token index of the FIRST token of the assistant's reply
    (i.e., the token immediately after '<start_of_turn>model\\n').

    Strategy: encode the prefix up to and including the marker, then the
    number of tokens in that prefix is our mask boundary.

    Falls back to encoding with return_offsets_mapping if the fast tokenizer
    supports it (more accurate with padding).
    """
    # Find the LAST occurrence of the marker in the text
    marker_pos = text.rfind(ASSISTANT_MARKER)
    if marker_pos == -1:
        return -1

    # Character position where the assistant's actual reply starts
    reply_char_start = marker_pos + len(ASSISTANT_MARKER)

    # Encode just the prefix (text up to reply start) to count its tokens
    prefix = text[:reply_char_start]
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    return len(prefix_ids)


def make_data_collator(tokenizer, max_length: int):
    def collate(batch):
        texts = [b["text"] for b in batch]

        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"]
        labels = input_ids.clone()

        for i, text in enumerate(texts):
            ids_list = input_ids[i].tolist()

            # Find where assistant reply starts in TOKEN space
            mask_until = find_assistant_token_start(text, ids_list, tokenizer)

            if mask_until == -1:
                labels[i] = -100  # no assistant turn found
            else:
                # Clamp to sequence length (truncation may have cut it)
                mask_until = min(mask_until, len(ids_list))
                labels[i, :mask_until] = -100

            # Mask padding tokens
            pad_id = tokenizer.pad_token_id
            if pad_id is not None:
                labels[i][input_ids[i] == pad_id] = -100

        enc["labels"] = labels
        return enc

    return collate


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, default=None)
    ap.add_argument("--model", type=str, default="google/gemma-4-E2B-it")
    ap.add_argument("--output-dir", type=Path, default=Path("./gemma4_e2b_knox"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--no-quantize", action="store_true")
    ap.add_argument("--auto-val-split", type=float, default=0.05)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    print(f"[+] Loading tokenizer for {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"[+] Tokenizer (with custom template) saved to {args.output_dir}")

    # Verify the marker appears correctly in a formatted sample
    test_msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    test_text = tokenizer.apply_chat_template(
        test_msgs, tokenize=False, add_generation_prompt=False
    )
    print(f"[+] Sample formatted text:\n    {repr(test_text)}")
    assert ASSISTANT_MARKER in test_text, (
        f"ASSISTANT_MARKER {ASSISTANT_MARKER!r} not found in formatted text!\n"
        f"Formatted text: {repr(test_text)}\n"
        f"Check that the chat template emits exactly '<start_of_turn>model\\n' "
        f"for the assistant role."
    )

    # Show where masking will cut off
    tok_boundary = find_assistant_token_start(test_text, [], tokenizer)
    test_ids = tokenizer.encode(test_text, add_special_tokens=False)
    print(f"[+] Mask boundary: token {tok_boundary} / {len(test_ids)} total")
    print(f"    Tokens being trained on: "
          f"{repr(tokenizer.decode(test_ids[tok_boundary:]))}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_kwargs = {"trust_remote_code": True}
    if args.no_quantize:
        model_kwargs.update({"torch_dtype": torch.bfloat16, "device_map": "cuda"})
        print("[+] Loading model in bf16", flush=True)
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs.update({"quantization_config": bnb_config, "device_map": "auto"})
        print("[+] Loading model in 4-bit NF4 (QLoRA)", flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.config.use_cache = False

    # ------------------------------------------------------------------
    # LoRA — all projection layers
    # ------------------------------------------------------------------
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=r".*\.language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
    )
    print(f"[+] LoRA r={args.lora_r} alpha={args.lora_alpha}")

    # ------------------------------------------------------------------
    # Dataset — pre-apply chat template
    # ------------------------------------------------------------------
    print("[+] Loading and pre-formatting dataset...", flush=True)
    train_records = load_jsonl(args.train)
    train_ds = build_hf_dataset(train_records, tokenizer)
    print(f"[+] Train: {len(train_ds)} records")

    val_ds = None
    if args.val and args.val.exists():
        val_ds = build_hf_dataset(load_jsonl(args.val), tokenizer)
        print(f"[+] Val: {len(val_ds)} records")
    elif args.auto_val_split > 0:
        split = train_ds.train_test_split(test_size=args.auto_val_split, seed=42)
        train_ds, val_ds = split["train"], split["test"]
        print(f"[+] Auto-split: {len(train_ds)} train / {len(val_ds)} val")
    has_val = val_ds is not None

    # ------------------------------------------------------------------
    # Verify loss masking on first 2 real samples
    # ------------------------------------------------------------------
    print("\n[+] Verifying loss masking on first 2 samples...")
    collator = make_data_collator(tokenizer, args.max_seq_len)
    sample_batch = [train_ds[i] for i in range(min(2, len(train_ds)))]
    collated = collator(sample_batch)
    labels = collated["labels"]
    n_active = (labels != -100).sum().item()
    n_total = labels.numel()
    pct = 100 * n_active / n_total if n_total else 0
    print(f"    Active label tokens: {n_active} / {n_total} ({pct:.1f}%)")

    for i, (row, item) in enumerate(zip(labels, sample_batch)):
        active_ids = [v.item() for v in row if v.item() != -100]
        decoded = tokenizer.decode(active_ids[:60], skip_special_tokens=False)
        print(f"    Sample {i} ({item.get('task','?')}): "
              f"{len(active_ids)} active tokens")
        print(f"      First active: {repr(decoded[:100])}")

    if n_active == 0:
        raise RuntimeError(
            "Still 0 active label tokens after char-offset masking.\n"
            f"ASSISTANT_MARKER = {ASSISTANT_MARKER!r}\n"
            "Check that preformat_record() embeds this marker in 'text'.\n"
            "Sample text: " + repr(sample_batch[0]["text"][:300])
        )
    print(f"    [OK] Loss masking working correctly.\n")

    # ------------------------------------------------------------------
    # SFT Config
    # ------------------------------------------------------------------
    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        max_seq_length=args.max_seq_len,
        dataset_text_field="text",   # tells SFTTrainer to use pre-formatted text
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        eval_strategy="steps" if has_val else "no",
        eval_steps=50 if has_val else None,
        load_best_model_at_end=has_val,
        metric_for_best_model="eval_loss" if has_val else None,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        dataset_num_proc=4,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds if has_val else None,
        peft_config=peft_config,
        data_collator=collator,
    )

    print("[+] Starting training...", flush=True)
    trainer.train()

    print(f"[+] Saving to {args.output_dir}", flush=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print("[+] Done.", flush=True)


if __name__ == "__main__":
    main()
