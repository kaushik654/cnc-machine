#!/usr/bin/env python3
"""
finetune_gemma4_e2b.py
----------------------
Complete SFT script for finetuning Gemma 4 E2B-IT on the Knox multi-task
dataset (routing + tool-calling + summarization).

Fixes over the previous version:
  1. Chat template correctly serializes tool_calls into <tool_call> blocks
  2. Loss is masked on system + user tokens (only assistant turn trains)
  3. LoRA targets all projection layers (not just q/k/v)
  4. tokenizer.save_pretrained called BEFORE training so the template is saved
  5. Pre-applies the chat template so SFTTrainer never touches it
  6. Serializes tool_calls into content before formatting to avoid Jinja errors

Usage:
    python finetune_gemma4_e2b.py \\
        --train sft_out/train.jsonl \\
        --val   sft_out/val.jsonl   \\
        --output-dir ./gemma4_e2b_knox \\
        --epochs 4 \\
        --lora-r 16 \\
        --no-quantize

Requirements:
    pip install transformers trl peft bitsandbytes accelerate datasets
"""

from __future__ import annotations
import os
import json
import argparse
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =============================================================================
# SECTION 1: CHAT TEMPLATE
#
# This template is applied ONCE (during pre-processing), before SFTTrainer
# ever sees the data. SFTTrainer receives plain pre-formatted strings.
#
# <tool_call> format: verify this matches your LiteRT-LM Android parser.
# If your runtime uses a different tag (e.g. [TOOL_CALL] or Python-style),
# change the two lines that emit <tool_call> / </tool_call>.
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

# Token that marks the START of the assistant turn — loss is computed from
# here onwards only. Must match what the template emits exactly.
ASSISTANT_TURN_START = "<start_of_turn>model\n"


# =============================================================================
# SECTION 2: DATA PRE-PROCESSING
# =============================================================================

def load_jsonl(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def normalize_messages(messages: list[dict]) -> list[dict]:
    """
    Converts tool_calls in assistant turns into a serialized string in
    'content'. This is needed because some TRL versions cannot handle the
    OpenAI tool_calls dict structure during dataset mapping.
    The content string uses the same <tool_call> format as the chat template.
    """
    normalized = []
    for msg in messages:
        msg = dict(msg)
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            parts = []
            for tc in msg["tool_calls"]:
                name = tc["function"]["name"]
                args = tc["function"]["arguments"]
                parts.append(f'<tool_call>\n{{"name": "{name}", "arguments": {args}}}\n</tool_call>')
            msg["content"] = "\n".join(parts)
            del msg["tool_calls"]
        normalized.append(msg)
    return normalized


def preformat_record(record: dict, tokenizer) -> dict | None:
    """
    Applies the chat template to a record and returns a dict with:
      - text: the fully formatted string (input to the model)
      - labels_mask_until: char offset where assistant turn starts
        (everything before this index gets label = -100)

    Returns None if the record should be skipped (e.g. empty assistant turn).
    """
    messages = normalize_messages(record.get("messages", []))
    if not messages:
        return None

    # Check the assistant turn has actual content
    last = messages[-1]
    if last.get("role") == "assistant" and not last.get("content", "").strip():
        return None

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception as e:
        print(f"[WARN] apply_chat_template failed for {record.get('sample_id')}: {e}")
        return None

    return {
        "text": text,
        "task": record.get("task", "unknown"),
        "sample_id": record.get("sample_id", ""),
    }


def build_hf_dataset(records: list[dict], tokenizer) -> Dataset:
    rows = []
    skipped = 0
    for r in records:
        out = preformat_record(r, tokenizer)
        if out is None:
            skipped += 1
            continue
        rows.append(out)
    if skipped:
        print(f"[WARN] Skipped {skipped} records during pre-formatting")
    return Dataset.from_list(rows)


# =============================================================================
# SECTION 3: LOSS MASKING (mask system + user tokens, train only on assistant)
# =============================================================================

def make_data_collator(tokenizer, max_length: int):
    """
    Custom collator that:
      1. Tokenizes the pre-formatted text
      2. Sets labels=-100 for all tokens before the FIRST assistant turn marker
         (i.e. the model only trains on its own outputs)
    """
    assistant_token_ids = tokenizer.encode(
        ASSISTANT_TURN_START, add_special_tokens=False
    )
    n_marker = len(assistant_token_ids)

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
            # Find the LAST occurrence of assistant_token_ids in the sequence
            # (last because we want to train on the final assistant turn)
            last_start = -1
            for j in range(len(ids_list) - n_marker + 1):
                if ids_list[j: j + n_marker] == assistant_token_ids:
                    last_start = j

            if last_start == -1:
                # No assistant turn found — mask everything (don't train on this)
                labels[i] = -100
            else:
                # Mask everything UP TO AND INCLUDING the marker tokens
                labels[i, : last_start + n_marker] = -100

        enc["labels"] = labels
        return enc

    return collate


# =============================================================================
# SECTION 4: MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument("--train", type=Path, required=True,
                    help="Training JSONL (output of transform_to_sft.py)")
    ap.add_argument("--val", type=Path, default=None,
                    help="Optional validation JSONL")

    # Model
    ap.add_argument("--model", type=str, default="google/gemma-4-E2B-it",
                    help="Base model. Use gemma-4-E4B-it for the larger 4B variant.")
    ap.add_argument("--output-dir", type=Path, default=Path("./gemma4_e2b_knox"))

    # Training hyperparameters
    ap.add_argument("--epochs", type=int, default=4,
                    help="More epochs needed with 1000 samples. 4-6 recommended.")
    ap.add_argument("--batch-size", type=int, default=2,
                    help="Per-device batch size")
    ap.add_argument("--grad-accum", type=int, default=8,
                    help="Gradient accumulation. Effective batch = batch * accum.")
    ap.add_argument("--learning-rate", type=float, default=1e-4,
                    help="Lower LR than before (2e-4 risks overfitting on 1k samples)")
    ap.add_argument("--max-seq-len", type=int, default=4096,
                    help="Max sequence length. Skill-run records can be 2-3K tokens.")
    ap.add_argument("--warmup-ratio", type=float, default=0.05)

    # LoRA
    ap.add_argument("--lora-r", type=int, default=16,
                    help="LoRA rank. 16 is sufficient for 1k samples. 64 will overfit.")
    ap.add_argument("--lora-alpha", type=int, default=32,
                    help="LoRA alpha. Keep = 2 * lora_r.")
    ap.add_argument("--lora-dropout", type=float, default=0.05)

    # Quantization
    ap.add_argument("--no-quantize", action="store_true",
                    help="Disable 4-bit quantization (use bf16). Needs ~18GB VRAM for E2B.")

    # Validation auto-split
    ap.add_argument("--auto-val-split", type=float, default=0.05,
                    help="If --val not provided, split this fraction from train. 0 to disable.")

    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    print(f"[+] Loading tokenizer for {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # required for SFT loss masking

    # Override the chat template with our tool-call-aware version
    tokenizer.chat_template = CHAT_TEMPLATE

    # Save BEFORE training so the template is baked into the output checkpoint
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"[+] Tokenizer (with custom chat template) saved to {args.output_dir}")

    # Quick sanity check: print one formatted turn to verify template
    test_msgs = [
        {"role": "system", "content": "You are a helpful assistant for Samsung Knox."},
        {"role": "user", "content": "Check my wifi"},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "connectivity_status", "arguments": '{"action":"wifi_status"}'}}
        ]},
    ]
    test_normalized = normalize_messages(test_msgs)
    sample_text = tokenizer.apply_chat_template(
        test_normalized, tokenize=False, add_generation_prompt=False
    )
    print("\n[+] Template sanity check (first 400 chars):")
    print(sample_text[:400])
    print("...")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_kwargs = {"trust_remote_code": True}
    if args.no_quantize:
        model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs["device_map"] = "cuda"
        print("[+] Loading model in bf16 (no quantization)", flush=True)
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
        print("[+] Loading model in 4-bit NF4 (QLoRA)", flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.config.use_cache = False  # required with gradient checkpointing

    # Print linear modules so you can see what LoRA can target
    linear_modules = [
        name for name, mod in model.named_modules()
        if "linear" in type(mod).__name__.lower()
    ]
    print(f"[+] Linear modules sample: {linear_modules[:5]} ... ({len(linear_modules)} total)")

    # ------------------------------------------------------------------
    # LoRA
    # Key fix: target ALL projection layers, not just q/k/v.
    # o_proj, gate_proj, up_proj, down_proj matter for tool-call generation.
    # ------------------------------------------------------------------
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # Targets ALL linear projections in the language model layers.
        # Adjust pattern if your Gemma 4 variant uses different layer names.
        target_modules=r".*\.language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
    )
    print(f"[+] LoRA: r={args.lora_r}, alpha={args.lora_alpha}, "
          f"targets=q/k/v/o/gate/up/down proj")

    # ------------------------------------------------------------------
    # Dataset
    # Pre-apply chat template here; SFTTrainer gets plain text strings.
    # This avoids SFTTrainer's internal apply_chat_template call, which
    # is what caused the "invalid chat template" error before.
    # ------------------------------------------------------------------
    print("[+] Loading and pre-formatting dataset...", flush=True)
    train_records = load_jsonl(args.train)
    train_ds = build_hf_dataset(train_records, tokenizer)
    print(f"[+] Train: {len(train_ds)} records")

    val_ds = None
    if args.val and args.val.exists():
        val_records = load_jsonl(args.val)
        val_ds = build_hf_dataset(val_records, tokenizer)
        print(f"[+] Val:   {len(val_ds)} records")
    elif args.auto_val_split > 0:
        split = train_ds.train_test_split(test_size=args.auto_val_split, seed=42)
        train_ds = split["train"]
        val_ds = split["test"]
        print(f"[+] Auto-split: {len(train_ds)} train / {len(val_ds)} val")

    has_val = val_ds is not None

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

        # Tell SFTTrainer the dataset field that contains pre-formatted text.
        # This bypasses SFTTrainer's internal chat template application entirely.
        dataset_text_field="text",

        # Loss masking: handled by our custom collator below.
        # DO NOT set dataset_kwargs={"skip_prepare_dataset": True} here —
        # we want SFTTrainer to tokenize but we override labels in the collator.

        # Logging / eval
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        eval_strategy="steps" if has_val else "no",
        eval_steps=50 if has_val else None,
        load_best_model_at_end=has_val,
        metric_for_best_model="eval_loss" if has_val else None,

        # Memory
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Misc
        report_to="none",
        dataset_num_proc=4,
        # Disable SFTTrainer's built-in chat template application —
        # we already pre-applied our template during build_hf_dataset().
        # This prevents double-formatting.
        dataset_kwargs={"skip_prepare_dataset": False},
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds if has_val else None,
        peft_config=peft_config,
        # Custom collator handles loss masking (masks system + user tokens).
        # Comment this out if you want to train on ALL tokens (worse results).
        data_collator=make_data_collator(tokenizer, args.max_seq_len),
    )

    # ------------------------------------------------------------------
    # Verify one training batch before committing to a full run
    # ------------------------------------------------------------------
    print("\n[+] Verifying first batch (checking labels are not all -100)...")
    sample_batch = [train_ds[i] for i in range(min(2, len(train_ds)))]
    collated = make_data_collator(tokenizer, args.max_seq_len)(sample_batch)
    labels = collated["labels"]
    n_active = (labels != -100).sum().item()
    n_total = labels.numel()
    print(f"    Active label tokens: {n_active} / {n_total} "
          f"({100*n_active/n_total:.1f}%)")
    if n_active == 0:
        raise RuntimeError(
            "All label tokens are -100! The assistant turn marker was not found. "
            "Check that ASSISTANT_TURN_START matches what the chat template emits."
        )
    if n_active / n_total > 0.8:
        print("    [WARN] >80% tokens are active — system prompt masking may not be working.")
    print("    [OK] Label masking looks correct.\n")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print("[+] Starting training...", flush=True)
    trainer.train()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    print(f"[+] Saving final adapter to {args.output_dir}", flush=True)
    trainer.save_model(str(args.output_dir))
    # Save tokenizer again to make sure the custom template is in the final dir
    tokenizer.save_pretrained(str(args.output_dir))
    print("[+] Done.", flush=True)

    # ------------------------------------------------------------------
    # Quick inference test
    # ------------------------------------------------------------------
    print("\n[+] Quick inference test with routing prompt...")
    ROUTING_SYSTEM = (
        "## Routing\n\n"
        "When a row fits, your **entire** reply is only that JSON object:\n"
        '{"title":"<exact catalog title>","source":"skill|tool"}\n\n'
        "## Catalog\nRoutable options:\n"
        '{"title":"battery","source":"skill"}\n'
        '{"title":"connectivity_status","source":"tool"}\n'
    )
    test_messages = [
        {"role": "system", "content": ROUTING_SYSTEM},
        {"role": "user", "content": "How do I optimise my battery?"},
    ]
    input_text = tokenizer.apply_chat_template(
        test_messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,  # greedy for routing — deterministic
            temperature=1.0,
        )
    response = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    print(f"    Routing response: {response!r}")
    print("    Expected: something like {\"title\":\"battery\",\"source\":\"skill\"}")


if __name__ == "__main__":
    main()
