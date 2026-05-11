#!/usr/bin/env python3

from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from trl import SFTConfig, SFTTrainer


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--train",
        type=Path,
        required=True,
        help="Path to training JSONL",
    )

    ap.add_argument(
        "--val",
        type=Path,
        default=None,
        help="Optional validation JSONL",
    )

    ap.add_argument(
        "--model",
        type=str,
        default="google/gemma-4-e2b-it",
    )

    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./gemma4_e2b_tools"),
    )

    # =========================
    # Training Hyperparameters
    # =========================

    ap.add_argument("--epochs", type=int, default=3)

    ap.add_argument("--batch-size", type=int, default=2)

    ap.add_argument("--grad-accum", type=int, default=4)

    ap.add_argument("--learning-rate", type=float, default=2e-4)

    ap.add_argument("--max-seq-len", type=int, default=4096)

    ap.add_argument("--warmup-ratio", type=float, default=0.05)

    # =========================
    # LoRA
    # =========================

    ap.add_argument("--lora-r", type=int, default=64)

    ap.add_argument("--lora-alpha", type=int, default=128)

    ap.add_argument("--lora-dropout", type=float, default=0.05)

    # =========================
    # Quantization
    # =========================

    ap.add_argument(
        "--no-quantize",
        action="store_true",
    )

    # =========================
    # Validation Split
    # =========================

    ap.add_argument(
        "--auto-val-split",
        type=float,
        default=0.05,
    )

    args = ap.parse_args()

    # ==========================================================
    # TOKENIZER
    # ==========================================================

    print(f"[+] Loading tokenizer for {args.model}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ==========================================================
    # FIX CHAT TEMPLATE
    # ==========================================================

    tokenizer.chat_template = """
{% for message in messages %}
{% if message['role'] == 'system' %}
<start_of_turn>system
{{ message['content'] }}<end_of_turn>
{% elif message['role'] == 'user' %}
<start_of_turn>user
{{ message['content'] }}<end_of_turn>
{% elif message['role'] == 'assistant' %}
<start_of_turn>model
{{ message['content'] }}<end_of_turn>
{% endif %}
{% endfor %}
"""

    # ==========================================================
    # MODEL LOADING
    # ==========================================================

    model_kwargs = {
        "trust_remote_code": True
    }

    if args.no_quantize:

        model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs["device_map"] = "cuda"

        print("[+] Loading model in bf16", flush=True)

    else:

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"

        print("[+] Loading model in 4bit NF4", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        **model_kwargs
    )

    model.config.use_cache = False

    # ==========================================================
    # LORA
    # ==========================================================

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )

    # ==========================================================
    # LOAD DATASET
    # ==========================================================

    data_files = {
        "train": str(args.train)
    }

    if args.val:
        data_files["validation"] = str(args.val)

    print("[+] Loading dataset...", flush=True)

    dataset = load_dataset(
        "json",
        data_files=data_files,
    )

    # ==========================================================
    # AUTO VALIDATION SPLIT
    # ==========================================================

    if not args.val and args.auto_val_split > 0:

        split = dataset["train"].train_test_split(
            test_size=args.auto_val_split,
            seed=42,
        )

        dataset = {
            "train": split["train"],
            "validation": split["test"],
        }

        print(
            f"[+] Train: {len(dataset['train'])} | "
            f"Val: {len(dataset['validation'])}",
            flush=True,
        )

    else:

        print(
            f"[+] Train size: {len(dataset['train'])}",
            flush=True,
        )

        if args.val:
            print(
                f"[+] Val size: {len(dataset['validation'])}",
                flush=True,
            )

    has_val = "validation" in dataset

    # ==========================================================
    # FORMAT DATASET
    # ==========================================================

    def formatting_func(example):

        messages = example["messages"]

        # remove tool_calls if present
        cleaned_messages = []

        for msg in messages:

            cleaned = {
                "role": msg["role"],
                "content": msg.get("content", "")
            }

            cleaned_messages.append(cleaned)

        text = tokenizer.apply_chat_template(
            cleaned_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        return {
            "text": text
        }

    print("[+] Formatting dataset...", flush=True)

    dataset["train"] = dataset["train"].map(
        formatting_func,
        num_proc=4,
    )

    if has_val:

        dataset["validation"] = dataset["validation"].map(
            formatting_func,
            num_proc=4,
        )

    # ==========================================================
    # SFT CONFIG
    # ==========================================================

    sft_config = SFTConfig(

        output_dir=str(args.output_dir),

        num_train_epochs=args.epochs,

        per_device_train_batch_size=args.batch_size,

        per_device_eval_batch_size=args.batch_size,

        gradient_accumulation_steps=args.grad_accum,

        learning_rate=args.learning_rate,

        warmup_ratio=args.warmup_ratio,

        lr_scheduler_type="cosine",

        max_length=args.max_seq_len,

        logging_steps=10,

        save_steps=100,

        save_total_limit=3,

        eval_strategy="steps" if has_val else "no",

        eval_steps=50 if has_val else None,

        bf16=True,

        gradient_checkpointing=True,

        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },

        report_to="none",

        dataset_num_proc=4,
    )

    # ==========================================================
    # TRAINER
    # ==========================================================

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation") if has_val else None,
        dataset_text_field="text",
        peft_config=peft_config,
    )

    # ==========================================================
    # TRAIN
    # ==========================================================

    print("[+] Starting training...", flush=True)

    trainer.train()

    # ==========================================================
    # SAVE
    # ==========================================================

    print(f"[+] Saving model to {args.output_dir}", flush=True)

    trainer.save_model(str(args.output_dir))

    tokenizer.save_pretrained(str(args.output_dir))

    print("[+] Done.", flush=True)


if __name__ == "__main__":
    main()
