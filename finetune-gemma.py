# Run this on your server to find the exact token IDs
# python /tmp/debug_tokens.py --model google/gemma-4-E2B-it
import argparse
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="google/gemma-4-E2B-it")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

# Check how <start_of_turn> is tokenized
for text in ["<start_of_turn>", "<start_of_turn>model", "<end_of_turn>", "model"]:
    ids = tok.encode(text, add_special_tokens=False)
    print(f"  {text!r:30s} -> {ids}")

# Check the full assistant marker
marker = "<start_of_turn>model\n"
ids = tok.encode(marker, add_special_tokens=False)
print(f"\n  Full marker {marker!r} -> {ids}")

# Also show a sample formatted output
msgs = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there"},
]

# Apply default template
try:
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    print(f"\n  Sample formatted text:\n{repr(text)}")
    token_ids = tok.encode(text, add_special_tokens=False)
    print(f"\n  Token IDs: {token_ids[:30]}...")
except Exception as e:
    print(f"  Template error: {e}")
