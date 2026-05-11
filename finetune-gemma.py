import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH = "/home/sharmila_m/kaushik/LLM-EVAL/deepeval/gemma-finetune/gemma4_e2b_D3"
DEVICE     = "cuda"   # change to "cpu" if needed
DTYPE      = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("[+] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
has_custom = tokenizer.chat_template and "<tool_call>" in tokenizer.chat_template
print(f"    Chat template: {'✓ custom (tool_call aware)' if has_custom else '✗ default — may be wrong'}")

print("[+] Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    device_map=DEVICE,
    trust_remote_code=True,
)
model.eval()
print("    Done.\n")

# ── HELPER ────────────────────────────────────────────────────────────────────
def ask(messages, max_new_tokens=128, greedy=False):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=4096).to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=not greedy,
            temperature=0.7 if not greedy else 1.0,
            top_p=0.9   if not greedy else 1.0,
        )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

# ── SYSTEM PROMPTS (minimal versions for quick testing) ───────────────────────
ROUTING_SYS = """\
## Routing
When a catalog row fits the user message, reply with ONLY this JSON (two keys, nothing else):
{"title":"<exact catalog title>","source":"skill|tool"}

## Catalog
Routable options:
{"title":"battery","source":"skill"}
{"title":"connectivity","source":"skill"}
{"title":"connectivity_status","source":"tool"}
{"title":"display_control","source":"tool"}
{"title":"location_status","source":"tool"}
{"title":"memory","source":"skill"}
"""

TOOL_SYS = """\
You are a helpful assistant for Samsung Knox.

## Tool calling (this turn)
When the message is in scope: call only the provided tool; no other assistant text.

<tools>
[{"type":"function","function":{"name":"connectivity_status","description":"Network connectivity status","parameters":{"type":"object","properties":{"action":{"type":"string","enum":["status","vpn_status","wifi_status"]}},"required":["action"]}}}]
</tools>
"""

SUMM_SYS = """\
Sequential GGUF tool blocks:

<|tool_response|>response:TOOL{value:"TEXT"}<|tool_response|>

One natural sentence per block, same order.
Preserve the language and script used in each block's TEXT;
do not translate it or switch to another language.

Reply with only a JSON string array, no markdown or extra text."""

# ── TEST 1: ROUTING ───────────────────────────────────────────────────────────
print("=" * 55)
print("TASK 1 — ROUTING")
print("=" * 55)

routing_tests = [
    ("How do I optimise battery life?",       "battery / skill"),
    ("Check my wifi and VPN status",          "connectivity / skill"),
    ("What is the weather today?",            "NO ROUTE — prose reply"),
    ("Turn off my display",                   "display_control / tool"),
]

for query, expected in routing_tests:
    resp = ask([{"role":"system","content":ROUTING_SYS},
                {"role":"user",  "content":query}],
               max_new_tokens=32, greedy=True)
    try:
        parsed = json.loads(resp)
        result = f"title={parsed.get('title')!r}  source={parsed.get('source')!r}  ✓ valid JSON"
    except json.JSONDecodeError:
        result = f"NOT valid JSON  ✗   raw={resp!r}"
    print(f"  Q: {query}")
    print(f"  Expected: {expected}")
    print(f"  Got:      {result}\n")

# ── TEST 2: TOOL CALLING ──────────────────────────────────────────────────────
print("=" * 55)
print("TASK 2 — TOOL CALLING")
print("=" * 55)

tool_tests = [
    "Check my wifi status",
    "What is my VPN connection status?",
]

for query in tool_tests:
    resp = ask([{"role":"system","content":TOOL_SYS},
                {"role":"user",  "content":query}],
               max_new_tokens=128, greedy=True)
    ok = "<tool_call>" in resp
    print(f"  Q: {query}")
    print(f"  Has <tool_call>: {'✓' if ok else '✗'}")
    print(f"  Response: {resp}\n")

# ── TEST 3: SUMMARIZATION ─────────────────────────────────────────────────────
print("=" * 55)
print("TASK 3 — SUMMARIZATION")
print("=" * 55)

summ_user = (
    '<|tool_response|>response:TOOL{value:"{\\"status\\":\\"connected\\",'
    '\\"ssid\\":\\"HomeNetwork\\",\\"ip_address\\":\\"192.168.1.5\\"}"}'
    '<|tool_response|>'
    '<|tool_response|>response:TOOL{value:"{\\"status\\":\\"disconnected\\",'
    '\\"reason\\":\\"No active VPN\\"}"}<|tool_response|>'
)

resp = ask([{"role":"system","content":SUMM_SYS},
            {"role":"user",  "content":summ_user}],
           max_new_tokens=128, greedy=False)
try:
    parsed = json.loads(resp)
    print(f"  ✓ Valid JSON array with {len(parsed)} sentences (expected 2)")
    for i, s in enumerate(parsed):
        print(f"    [{i}] {s}")
except json.JSONDecodeError:
    print(f"  ✗ NOT valid JSON array")
    print(f"  Raw response: {resp}")

print("\n[done] — ✓ = working  ✗ = needs retraining")
