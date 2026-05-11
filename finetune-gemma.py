#!/usr/bin/env python3
"""
finetune_gemma4_e2b.py  (v3 - fixed loss masking for Gemma special tokens)
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
    """Serialize tool_calls dicts into content strings before template apply."""
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
# LOSS MASKING  -- robust version using special token IDs directly
# =============================================================================
def get_assistant_marker_ids(tokenizer) -> list[int]:
    """
    Find the token ID sequence for '<start_of_turn>model\n'.
    Gemma tokenizes <start_of_turn> as a SINGLE special token, so we must
    use token IDs rather than encoding the string character by character.
    """
    # Try encoding the full marker string
    ids = tokenizer.encode("<start_of_turn>model\n", add_special_tokens=False)
    print(f"[+] Assistant marker '<start_of_turn>model\\n' -> token IDs: {ids}")

    # Sanity: also check each piece
    sot_ids = tokenizer.encode("<start_of_turn>", add_special_tokens=False)
    model_ids = tokenizer.encode("model", add_special_tokens=False)
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    print(f"    <start_of_turn> -> {sot_ids}")
    print(f"    model           -> {model_ids}")
    print(f"    newline         -> {newline_ids}")
    return ids


def find_last_subseq(seq: list[int], subseq: list[int]) -> int:
    """Return start index of LAST occurrence of subseq in seq, or -1."""
    n, m = len(seq), len(subseq)
    for i in range(n - m, -1, -1):
        if seq[i: i + m] == subseq:
            return i
    return -1


def make_data_collator(tokenizer, max_length: int, marker_ids: list[int]):
    """
    Tokenizes pre-formatted text and masks all tokens BEFORE (and including)
    the last '<start_of_turn>model\n' marker. Only the assistant's output
    contributes to the loss.
    """
    n_marker = len(marker_ids)

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

        for i, ids in enumerate(input_ids):
            ids_list = ids.tolist()
            last_start = find_last_subseq(ids_list, marker_ids)
            if last_start == -1:
                # Marker not found — mask entire sequence (skip this sample)
                labels[i] = -100
            else:
                # Mask everything up to and including the marker itself
                labels[i, : last_start + n_marker] = -100
            # Also mask padding tokens
            pad_id = tokenizer.pad_token_id
            if pad_id is not None:
                labels[i][ids == pad_id] = -100

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
    print(f"[+] Tokenizer saved to {args.output_dir}")

    # Get marker IDs AFTER setting the custom template
    marker_ids = get_assistant_marker_ids(tokenizer)

    # Sanity: verify marker is found in a sample formatted record
    test_msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    test_text = tokenizer.apply_chat_template(
        test_msgs, tokenize=False, add_generation_prompt=False
    )
    test_ids = tokenizer.encode(test_text, add_special_tokens=False)
    found_at = find_last_subseq(test_ids, marker_ids)
    print(f"[+] Marker found at position {found_at} in test sequence (length {len(test_ids)})")
    if found_at == -1:
        # Marker not found — diagnose
        print("[!] DIAGNOSTIC: first 30 token IDs of test sequence:", test_ids[:30])
        print("[!] Looking for marker IDs:", marker_ids)
        print("[!] Formatted text repr:", repr(test_text[:200]))
        raise RuntimeError(
            "Assistant turn marker not found in test sequence.\n"
            "This usually means <start_of_turn> is tokenized differently.\n"
            "Run python debug_tokens.py --model <your_model> to inspect, "
            "then update marker_ids manually if needed."
        )
    print(f"[+] Marker check OK. Text after marker: {repr(test_text[test_text.find('<start_of_turn>model'):test_text.find('<start_of_turn>model')+40])}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_kwargs = {"trust_remote_code": True}
    if args.no_quantize:
        model_kwargs.update({"torch_dtype": torch.bfloat16, "device_map": "cuda"})
        print("[+] Loading model in bf16 (no quantization)", flush=True)
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
    # LoRA — target ALL linear projections
    # ------------------------------------------------------------------
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=r".*\.language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
    )
    print(f"[+] LoRA r={args.lora_r} alpha={args.lora_alpha} targets=q/k/v/o/gate/up/down")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    print("[+] Loading and pre-formatting dataset...", flush=True)
    train_records = load_jsonl(args.train)
    train_ds = build_hf_dataset(train_records, tokenizer)
    print(f"[+] Train: {len(train_ds)} records")

    val_ds = None
    if args.val and args.val.exists():
        val_ds = build_hf_dataset(load_jsonl(args.val), tokenizer)
        print(f"[+] Val:   {len(val_ds)} records")
    elif args.auto_val_split > 0:
        split = train_ds.train_test_split(test_size=args.auto_val_split, seed=42)
        train_ds, val_ds = split["train"], split["test"]
        print(f"[+] Auto-split: {len(train_ds)} train / {len(val_ds)} val")
    has_val = val_ds is not None

    # ------------------------------------------------------------------
    # Verify loss masking on 2 real samples BEFORE training
    # ------------------------------------------------------------------
    print("\n[+] Verifying loss masking on first 2 samples...")
    collator = make_data_collator(tokenizer, args.max_seq_len, marker_ids)
    sample_batch = [train_ds[i] for i in range(min(2, len(train_ds)))]
    collated = collator(sample_batch)
    labels = collated["labels"]
    n_active = (labels != -100).sum().item()
    n_total = labels.numel()
    pct = 100 * n_active / n_total if n_total else 0
    print(f"    Active label tokens: {n_active} / {n_total} ({pct:.1f}%)")
    for i, row in enumerate(labels):
        active = (row != -100).sum().item()
        print(f"    Sample {i}: {active} active tokens")
        # Show what the active tokens decode to (first 80 chars)
        active_ids = [v.item() for v in row if v != -100]
        decoded = tokenizer.decode(active_ids[:40], skip_special_tokens=False)
        print(f"             first active tokens: {repr(decoded[:80])}")

    if n_active == 0:
        print("\n[!] STILL 0 active tokens. Running deep diagnostic...")
        sample_text = sample_batch[0]["text"]
        sample_ids = tokenizer.encode(sample_text, add_special_tokens=False)
        print(f"    Sample text (first 300 chars): {repr(sample_text[:300])}")
        print(f"    First 40 token IDs: {sample_ids[:40]}")
        print(f"    Looking for marker IDs: {marker_ids}")
        # Try to find partial matches
        for length in range(len(marker_ids), 0, -1):
            partial = marker_ids[:length]
            pos = find_last_subseq(sample_ids, partial)
            if pos != -1:
                print(f"    Partial match (len={length}): found at position {pos}")
                print(f"    Context around match: {sample_ids[max(0,pos-2):pos+length+2]}")
                break
        raise RuntimeError(
            "Loss masking verification failed — 0 active tokens.\n"
            "See diagnostic output above to identify the token ID mismatch."
        )

    if pct > 80:
        print("    [WARN] >80% tokens are active — system/user masking may be off.")
    print("    [OK] Loss masking looks correct.\n")

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
        dataset_text_field="text",
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

    print(f"[+] Saving adapter to {args.output_dir}", flush=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print("[+] Done.", flush=True)


if __name__ == "__main__":
    main()
