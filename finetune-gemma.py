from __future__ import annotations
import os
# NOTE: removed `os.environ["CUDA_VISIBLE_DEVICES"]="0"` — torchrun/accelerate
# sets LOCAL_RANK per-process and exposes the right GPU to each rank.
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
from accelerate import PartialState


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True,
                    help="Path to training JSONL (output of convert_to_gemma_sft.py)")
    ap.add_argument("--val", type=Path, default=None,
                    help="Optional validation JSONL")
    ap.add_argument("--model", type=str, default="google/gemma-4-E2B-it",
                    help="Base model. Use gemma-4-E4B-it for the larger 4B variant.")
    ap.add_argument("--output-dir", type=Path, default=Path("./gemma4_e2b_tools"))

    # Training hyperparameters
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2,
                    help="Per-device train batch size")
    ap.add_argument("--grad-accum", type=int, default=4,
                    help="Gradient accumulation steps. "
                         "effective batch = batch * accum * num_gpus")
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--max-seq-len", type=int, default=4096,
                    help="Max sequence length. Most tool-calling samples fit in 2-3K tokens.")
    ap.add_argument("--warmup-ratio", type=float, default=0.05)

    # LoRA
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--lora-dropout", type=float, default=0.05)

    # Quantization
    ap.add_argument("--no-quantize", action="store_true",
                    help="Disable 4-bit quantization (use bf16 instead). "
                         "Needs ~24 GB VRAM per GPU for E2B.")
    ap.add_argument("--auto-val-split", type=float, default=0.05,
                    help="If --val not provided, auto-split this fraction of "
                         "train as validation. Set 0 to disable. Default 0.05.")

    args = ap.parse_args()

    # Each rank loads its own copy of the model on its own GPU
    local_rank = PartialState().process_index
    is_main = PartialState().is_main_process

    if is_main:
        print(f"[+] Loading tokenizer for {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------------------------------------------------------------
    # Model loading
    # ---------------------------------------------------------------------
    # CHANGED for DDP: device_map must pin the *whole* model to the current
    # rank's GPU. `"auto"` would do naive model parallelism (only one GPU
    # active at a time), and `"cuda"` would stack all ranks on GPU 0.
    # ---------------------------------------------------------------------
    model_kwargs = {"trust_remote_code": True}
    if args.no_quantize:
        model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs["device_map"] = {"": local_rank}
        if is_main:
            print("[+] Loading model in bf16 (no quantization)", flush=True)
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = {"": local_rank}
        if is_main:
            print("[+] Loading model in 4-bit NF4 (QLoRA mode)", flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.config.use_cache = False

    if is_main:
        print([name for name, module in model.named_modules()
               if "linear" in str(type(module)).lower()])

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=r".*\.language_model.*\.(q_proj|k_proj|v_proj)",
    )

    # ---------------------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------------------
    data_files = {"train": str(args.train)}
    if args.val:
        data_files["validation"] = str(args.val)
    dataset = load_dataset("json", data_files=data_files)

    if not args.val and args.auto_val_split > 0:
        split = dataset["train"].train_test_split(
            test_size=args.auto_val_split, seed=42
        )
        dataset = {"train": split["train"], "validation": split["test"]}
        if is_main:
            print(f"[+] Auto-split train into {len(dataset['train'])} train / "
                  f"{len(dataset['validation'])} val "
                  f"(test_size={args.auto_val_split})", flush=True)
    else:
        if is_main:
            print(f"[+] Train size: {len(dataset['train'])}", flush=True)
            if args.val:
                print(f"[+] Val size: {len(dataset['validation'])}", flush=True)

    has_val = "validation" in dataset

    # ---------------------------------------------------------------------
    # SFT config
    # ---------------------------------------------------------------------
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
        # Logging / eval
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        eval_strategy="steps" if has_val else "no",
        eval_steps=50 if has_val else None,
        # Memory
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # DDP — critical for LoRA. Without this DDP throws
        # "expected all parameters to receive gradient" errors because
        # the LoRA-frozen base weights don't get grads.
        ddp_find_unused_parameters=False,
        # Misc
        report_to="none",
        dataset_num_proc=4,
        # SFTTrainer will automatically apply Gemma's chat template to
        # the `messages` field in each sample.
    )

    trainer = SFTTrainer(
        model=model,
        # tokenizer=tokenizer,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation") if has_val else None,
        peft_config=peft_config,
    )

    if is_main:
        print("[+] Starting training...", flush=True)
    trainer.train()

    # ---------------------------------------------------------------------
    # Save only on rank 0 — otherwise all 4 ranks race to write to the same
    # directory, producing duplicate logs and occasional corrupt saves.
    # ---------------------------------------------------------------------
    if trainer.accelerator.is_main_process:
        print(f"[+] Saving final adapter to {args.output_dir}", flush=True)
        trainer.save_model(str(args.output_dir))
        tokenizer.save_pretrained(str(args.output_dir))
        print("[+] Done.", flush=True)


if __name__ == "__main__":
    main()
