#!/usr/bin/env python3
"""
convert_to_gguf_nobuild.py
--------------------------
GGUF conversion WITHOUT needing to build llama.cpp from source.

Uses only the Python conversion scripts from llama.cpp (no C++ compilation):
  1. convert_hf_to_gguf.py  -> base model GGUF (f16)
  2. convert_lora_to_gguf.py -> adapter GGUF (f16)
  3. llama-cpp-python's quantize() -> base q4_0 GGUF

You still need to clone llama.cpp for the Python scripts, but you do NOT
need cmake or a C++ compiler.

Setup:
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp && pip install -r requirements.txt
    pip install llama-cpp-python   # provides quantize() in Python

Usage:
    python convert_to_gguf_nobuild.py \\
        --adapter_path /home/sharmila_m/.../gemma4_e2b_D3 \\
        --base_model google/gemma-4-E2B-it \\
        --output_dir ./gguf_out \\
        --llama_cpp /path/to/llama.cpp \\
        --quant q4_0
"""
import argparse
import subprocess
from pathlib import Path


def find_python_scripts(llama_cpp: Path):
    hf_script = llama_cpp / "convert_hf_to_gguf.py"
    lora_script = llama_cpp / "convert_lora_to_gguf.py"
    for s in [hf_script, lora_script]:
        if not s.exists():
            raise FileNotFoundError(f"Not found: {s}")
    return hf_script, lora_script


def download_base_model(base_model: str, cache_dir: Path) -> Path:
    if Path(base_model).exists():
        return Path(base_model)
    from huggingface_hub import snapshot_download
    print(f"[+] Downloading {base_model} -> {cache_dir}")
    local = snapshot_download(
        repo_id=base_model,
        local_dir=str(cache_dir),
        local_dir_use_symlinks=False,
    )
    return Path(local)


def copy_tokenizer_from_adapter(adapter_path: Path, base_local: Path):
    """
    Critical: GGUF embeds the chat template from tokenizer_config.json.
    Your adapter dir has the custom template — base HF dir has the default.
    Copy adapter's tokenizer files into the base dir BEFORE conversion so
    the GGUF carries your tool-calling chat template.
    """
    import shutil
    files_to_copy = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",  # SentencePiece model if present
    ]
    print(f"[+] Copying tokenizer files from adapter -> base (for chat template)")
    for fname in files_to_copy:
        src = adapter_path / fname
        if src.exists():
            dst = base_local / fname
            shutil.copy2(src, dst)
            print(f"    {fname}")


def run_python_script(script: Path, args: list[str]):
    cmd = ["python", str(script)] + args
    print(f"    $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def convert_base_to_f16(base_local: Path, hf_script: Path, output: Path):
    print(f"[+] Converting BASE model -> GGUF f16")
    run_python_script(hf_script, [
        str(base_local),
        "--outfile", str(output),
        "--outtype", "f16",
    ])
    show_size(output)


def convert_adapter(adapter_path: Path, lora_script: Path,
                    base_local: Path, output: Path):
    print(f"[+] Converting LoRA adapter -> GGUF f16")

    # Sanity check
    cfg = adapter_path / "adapter_config.json"
    if not cfg.exists():
        raise FileNotFoundError(
            f"{cfg} not found. --adapter_path must be the LoRA dir."
        )

    run_python_script(lora_script, [
        str(adapter_path),
        "--base", str(base_local),
        "--outfile", str(output),
        "--outtype", "f16",
    ])
    show_size(output)


def quantize_with_llama_cpp_python(src: Path, dst: Path, quant: str = "q4_0"):
    """
    Use llama-cpp-python's bundled quantize function — no cmake needed.
    Available in llama_cpp.llama_quantize.
    """
    try:
        from llama_cpp import Llama
        from llama_cpp.llama_cpp import llama_model_quantize, llama_model_quantize_params
    except ImportError:
        print("[!] llama-cpp-python not installed.")
        print("    Install with: pip install llama-cpp-python")
        print("    Then re-run with --skip_quant=False")
        raise

    # Map quant string to llama.cpp file type enum
    quant_map = {
        "q4_0":   2,
        "q4_1":   3,
        "q5_0":   8,
        "q5_1":   9,
        "q8_0":   7,
        "q4_K_M": 15,
        "q4_K_S": 14,
        "q5_K_M": 17,
        "q5_K_S": 16,
        "q6_K":   18,
    }
    if quant.lower() not in quant_map:
        raise ValueError(f"Unsupported quant {quant}. Try one of: {list(quant_map)}")

    print(f"[+] Quantizing {src.name} -> {quant.upper()} via llama-cpp-python")
    params = llama_model_quantize_params()
    params.ftype = quant_map[quant.lower()]
    params.nthread = 8

    result = llama_model_quantize(
        str(src).encode("utf-8"),
        str(dst).encode("utf-8"),
        params,
    )
    if result != 0:
        raise RuntimeError(f"Quantization failed with code {result}")
    show_size(dst)


def show_size(p: Path):
    if p.exists():
        print(f"    Size: {p.stat().st_size / (1024*1024):.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_path", type=Path, required=True)
    ap.add_argument("--base_model", type=str, default="google/gemma-4-E2B-it")
    ap.add_argument("--output_dir", type=Path, default=Path("./gguf_out"))
    ap.add_argument("--llama_cpp", type=Path, required=True)
    ap.add_argument("--quant", type=str, default="q4_0")
    ap.add_argument("--skip_base", action="store_true")
    ap.add_argument("--skip_adapter", action="store_true")
    ap.add_argument("--skip_quant", action="store_true",
                    help="Skip quantization (keep base as f16)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hf_script, lora_script = find_python_scripts(args.llama_cpp)

    # Resolve base model to a local path
    base_local = download_base_model(args.base_model, args.output_dir / "_base_hf")

    # Make sure chat template flows into GGUF
    copy_tokenizer_from_adapter(args.adapter_path, base_local)

    base_f16 = args.output_dir / "gemma4_e2b_base_f16.gguf"
    base_quant = args.output_dir / f"gemma4_e2b_base_{args.quant}.gguf"
    adapter_gguf = args.output_dir / "gemma4_e2b_lora.gguf"

    # ── BASE ─────────────────────────────────────────────────────────────────
    if not args.skip_base:
        if not base_f16.exists():
            convert_base_to_f16(base_local, hf_script, base_f16)
        else:
            print(f"[~] {base_f16.name} already exists, skipping conversion")

        if not args.skip_quant:
            quantize_with_llama_cpp_python(base_f16, base_quant, args.quant)
    else:
        print("[~] Skipping base")

    # ── ADAPTER ──────────────────────────────────────────────────────────────
    if not args.skip_adapter:
        convert_adapter(args.adapter_path, lora_script, base_local, adapter_gguf)
    else:
        print("[~] Skipping adapter")

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    for f in [base_f16, base_quant, adapter_gguf]:
        if f.exists():
            print(f"  {f}  ({f.stat().st_size / (1024*1024):.1f} MB)")

    print("\nFor Android deployment, ship these two files:")
    if base_quant.exists():
        print(f"  Base:    {base_quant.name}")
    elif base_f16.exists():
        print(f"  Base:    {base_f16.name}  (consider quantizing for Android)")
    if adapter_gguf.exists():
        print(f"  Adapter: {adapter_gguf.name}")


if __name__ == "__main__":
    main()
