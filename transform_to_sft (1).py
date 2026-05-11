"""
transform_to_sft.py

Transforms the raw Knox multi-task dataset into a combined SFT dataset for
finetuning Gemma E2B-IT. Uses both tool_catalog.json AND skills_catalog.json
so multi-tool samples can route to skills (not just to the first tool).

USAGE:
    python transform_to_sft.py \
        --raw_dataset path/to/raw.jsonl \
        --tool_schemas path/to/tool_suwon_ocfk_v2.json \
        --tool_catalog path/to/tool_catalog.json \
        --skills_catalog path/to/skills_catalog.json \
        --output_dir ./sft_out
"""

import json
import argparse
import random
import re
from pathlib import Path
from collections import Counter, defaultdict


# =============================================================================
# SECTION 1: SYSTEM PROMPTS (transcribed from your screenshots)
# Diff these against your .txt files before training.
# Curly braces in prompt body are doubled because str.format() injects fields.
# =============================================================================

ROUTING_PROMPT = """## Routing

**Scope:** Routing JSON below applies to normal **text-only** user requests.
**Skip it entirely** when the user message includes `<__media__>` / vision input;
use prose only for those turns.

`<documents>` rows include `title`, `description`, and `source` **for your reference only**.
When a row fits the user message, you do **not** echo those field names back.
You output a **separate**, tiny JSON that contains **only** `title` and `source` — two keys total.

When a row fits, your **entire** reply for this turn is **only** that JSON object:
no `<tool_call>`, no markdown, no explanation, **no** chat-template markers.

**Copy-paste pattern (two keys only — do not add a third key):**
- Skill: {{"title":"<exact catalog title>","source":"skill"}}
- Tool: {{"title":"<exact catalog title>","source":"tool"}}

**Critical — if you violate this, the run fails:**
- The object must have **exactly two keys:** `title` and `source`.
  **Never** add `params`, `arguments`, `action`, `flashlight`, or any nested `{{...}}`.
  Never add commas and a third key. One comma between the two string fields only.

- **Stop generation immediately after the final `}}` character.**
  Do not output `<turn|>`, `<|turn|>`, `<|turn>model`, `<eos>`, or any text or symbol after `}}`.
  Those tokens break parsing.

- Keep routing JSON **short** so it is not cut off: long JSON (e.g. with `params`)
  often loses the closing `}}` and fails.

- `title` must match one catalog `title` exactly (e.g. `device_control`).
  Never put the user's free text as `title`.

- `source` must be exactly `skill` or `tool` (lowercase).

- Routing **only** picks catalog `title` + `source`;
  the next pipeline step reads the user message for details like on/off, brightness, or locations.

- **Forbidden:** `params` · `arguments` · nested objects · plain `device_control flashlight` ·
  `title || tool` · prose before/after JSON · anything after the closing `}}`.

Prefer a skill row over `exec` when both apply;
device calendar → `calendar` with `source":"tool`; work boards → skill.

If the latest user message does not clearly match a catalog row, do **not** route.
Answer the user's message naturally and directly in plain conversational sentences.
Do not mention the catalog, routing, JSON, or that no tool was found.
Do not repeat a previous assistant answer.

**Routing output shape (only when a catalog row clearly fits):**
{{"title":"<exact catalog title>","source":"skill|tool"}}

**Not allowed in routing output:**
extra keys, nested objects, text before or after JSON,
any chat-template marker after the closing `}}`.

## Catalog
Routable options:

{catalog}
"""


DIRECT_TOOL_RUN_PROMPT = """You are a helpful assistant for Samsung Knox.

## Tool calling (this turn)

The same user message is sent again below.
Use the model's native tool-calling format only if that message clearly matches this tool's purpose.
If the message is unrelated, off-topic, or needs a different capability,
reply with one short plain sentence and no tool call.

When the message is in scope: call only the provided tool; no other assistant text.

<tools>
{openAITools}
</tools>
"""


SKILL_RUN_PROMPT = """You are a helpful assistant for Samsung Knox.

## Tool calling only (this turn)

The same user message is sent again below.
If it reasonably matches this skill's scope (see the loaded SKILL.md),
produce only native tool calls — no other prose.
If the message clearly does not match this skill,
reply with one short plain sentence and no tool call.
Do not use read_skill unless the skill text explicitly says to load another file.

## Tool allowlist (mandatory)

In the YAML frontmatter under knoxagent.requires.tools, the skill lists which tools may be used for this procedure.
You MUST call only those tool names, and only if they appear in the native tool declarations.
Do not call any other tool (e.g. device_control when the allowlist is only exec).
If requires.tools is missing from the frontmatter,
use any declared tool that the skill procedure implies.

Rules:
- Use the model's native tool-calling format for every call. Do not invent another format.
- Tool names and parameters must match the native declarations and the allowlist rule above.
- For multiple tools, emit separate native tool calls in the required order.
- For exec.command, use single quotes in the shell for URLs and jq filters when possible
  (e.g. curl -s 'https://example.com' | jq '.x').
- No markdown code fences. No assistant text before or after the tool call(s).
- Derive arguments from the user's message only when it is in scope for this skill;
  use the loaded skill body (e.g. exec with curl|jq when the skill uses HTTP APIs).
- Tool results arrive as tool result messages; this step is only to emit the required call(s).

<tools>
{openAITools}
</tools>

## Skill
{skillBody}
"""


SUMMARIZATION_PROMPT = """Sequential GGUF tool blocks:

<|tool_response|>response:TOOL{value:"TEXT"}<|tool_response|>

One natural sentence per block, same order.
Preserve the language and script used in each block's TEXT;
do not translate it or switch to another language.

Reply with only a JSON string array, no markdown or extra text."""


# =============================================================================
# SECTION 2: CONFIG
# =============================================================================

VAL_RATIO = 0.10
RANDOM_SEED = 42
STRICT_SCHEMA_VALIDATION = True

# Length warning threshold (approx chars; ~4 chars per token).
# Records exceeding this get logged in stats.json so you can spot bloat.
LENGTH_WARN_CHARS = 14000  # ~3500 tokens

TOOL_RESPONSE_OPEN = "<|tool_response|>"
TOOL_RESPONSE_CLOSE = "<|tool_response|>"


# =============================================================================
# SECTION 3: LOADERS
# =============================================================================

def load_jsonl(path):
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return json.loads(text)
    records = []
    for i, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"[WARN] skipping malformed line {i}: {e}")
    return records


def load_tool_schemas(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tools = data["tools"] if isinstance(data, dict) and "tools" in data else data
    schemas = {}
    for t in tools:
        fn = t.get("function", {})
        name = fn.get("name")
        if name:
            schemas[name] = {"type": "function", "function": fn}
    return schemas


def load_tool_catalog(path):
    """Returns list of {title, source: 'tool'} for routable tools."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tools = data["tools"] if isinstance(data, dict) and "tools" in data else data
    return [
        {"title": t["id"], "source": "tool"}
        for t in tools
        if t.get("defaultEnabled", True)
    ]


def load_skills_catalog(path):
    """
    Returns:
      catalog_entries: list of {title, source: 'skill'} for routing
      skill_index: dict {name -> {requires_tools, body, description}}
    """
    if path is None or not Path(path).exists():
        return [], {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    skills = data.get("skills", []) if isinstance(data, dict) else data
    catalog = [{"title": s["name"], "source": "skill"} for s in skills]
    index = {
        s["name"]: {
            "requires_tools": s.get("requires_tools", []),
            "body": s.get("body", ""),
            "description": s.get("description", ""),
        }
        for s in skills
    }
    return catalog, index


def render_catalog(catalog_entries):
    return "\n".join(
        json.dumps(e, separators=(",", ":")) for e in catalog_entries
    )


# =============================================================================
# SECTION 4: SCHEMA VALIDATION
# =============================================================================

def validate_tool_call(tool_call, schemas):
    name = tool_call.get("name")
    if name not in schemas:
        return False, f"unknown tool: {name}"
    args = tool_call.get("arguments") or tool_call.get("input_parameters") or {}
    params = schemas[name]["function"].get("parameters", {})
    for r in params.get("required", []):
        if r not in args:
            return False, f"missing required arg '{r}' for {name}"
    properties = params.get("properties", {})
    for k, v in args.items():
        if k in properties and "enum" in properties[k]:
            if v not in properties[k]["enum"]:
                return False, f"value {v!r} for {name}.{k} not in enum"
    return True, ""


# =============================================================================
# SECTION 5: FIELD EXTRACTORS — edit if your dataset uses different names
# =============================================================================

def extract_user_message(sample):
    for t in sample.get("turns", []):
        if t.get("role") == "user":
            return t.get("content", "")
    return sample.get("scenario", "")


def extract_tools_called(sample):
    for t in sample.get("turns", []):
        if t.get("role") == "assistant" and t.get("tools_called"):
            return t["tools_called"]
    return sample.get("tools_called", [])


def extract_assistant_summary(sample):
    turns = sample.get("turns", [])
    for t in reversed(turns):
        if t.get("role") == "assistant":
            c = t.get("content", "").strip()
            if c and not t.get("tools_called"):
                return c
    for t in reversed(turns):
        if t.get("role") == "assistant":
            return t.get("content", "").strip()
    return ""


def split_into_sentences(prose, n_target):
    parts = re.split(r'(?<=[.!?])\s+', prose.strip())
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) == n_target:
        return parts
    if len(parts) < n_target and n_target > 1:
        pieces = re.split(r',\s+|\s+and\s+', prose.strip())
        pieces = [p.strip().rstrip('.') + '.' for p in pieces if p.strip()]
        if len(pieces) == n_target:
            return pieces
    return parts


# =============================================================================
# SECTION 6: SKILL ROUTING LOGIC
# =============================================================================

def find_matching_skill(target_tools, skill_index):
    """
    Returns the name of the skill whose requires_tools is a superset of
    target_tools, or None if no skill matches.

    Preference: smallest skill that covers target_tools (most specific).
    """
    target_set = set(target_tools)
    candidates = []
    for name, info in skill_index.items():
        required = set(info.get("requires_tools", []))
        if target_set.issubset(required) and required:
            candidates.append((name, len(required)))
    if not candidates:
        return None
    # Smallest required-tools list wins (most specific skill)
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]


def decide_routing(sample, skill_index):
    """
    Returns dict {"title", "source"} or None for unrouted.
    Rules:
      - No target_tools -> None (unrouted)
      - Single target_tool -> route to that tool
      - Multi target_tools -> find skill whose requires_tools covers them;
        if found, route to skill; else route to first tool as fallback.
    """
    target_tools = sample.get("target_tools", [])
    if not target_tools:
        return None
    if len(target_tools) == 1:
        return {"title": target_tools[0], "source": "tool"}

    skill_name = find_matching_skill(target_tools, skill_index)
    if skill_name:
        return {"title": skill_name, "source": "skill"}
    # No skill covers these tools; fall back to first tool
    return {"title": target_tools[0], "source": "tool"}


# =============================================================================
# SECTION 7: RECORD BUILDERS
# =============================================================================

def build_routing_record(sample, catalog_entries, skill_index):
    user_msg = extract_user_message(sample)
    target_json = decide_routing(sample, skill_index)

    if target_json is None:
        assistant_content = sample.get("fallback_response") or \
            "I can help with that — could you tell me a bit more about what you need?"
    else:
        assistant_content = json.dumps(target_json, separators=(",", ":"))

    system = ROUTING_PROMPT.format(catalog=render_catalog(catalog_entries))

    return {
        "task": "route",
        "sample_id": sample.get("id"),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_content},
        ],
    }


def build_toolcall_record(sample, schemas, skill_index):
    user_msg = extract_user_message(sample)
    tools_called = extract_tools_called(sample)
    if not tools_called:
        return None

    if STRICT_SCHEMA_VALIDATION:
        for tc in tools_called:
            ok, reason = validate_tool_call(tc, schemas)
            if not ok:
                return {"_invalid": True, "reason": reason,
                        "sample_id": sample.get("id")}

    used_names = []
    for tc in tools_called:
        n = tc.get("name")
        if n in schemas and n not in used_names:
            used_names.append(n)
    openai_tools = [schemas[n] for n in used_names]
    openai_tools_str = json.dumps(openai_tools, indent=2)

    tool_calls = []
    for tc in tools_called:
        args = tc.get("arguments") or tc.get("input_parameters") or {}
        tool_calls.append({
            "id": f"call_{len(tool_calls)}",
            "type": "function",
            "function": {
                "name": tc["name"],
                "arguments": json.dumps(args, separators=(",", ":")),
            },
        })

    # Decide direct vs skill prompt based on routing decision
    target_tools = sample.get("target_tools", [])
    skill_name = find_matching_skill(target_tools, skill_index) if len(target_tools) > 1 else None

    if skill_name:
        skill_body = skill_index[skill_name]["body"]
        system = SKILL_RUN_PROMPT.format(
            openAITools=openai_tools_str,
            skillBody=skill_body,
        )
    else:
        system = DIRECT_TOOL_RUN_PROMPT.format(openAITools=openai_tools_str)

    return {
        "task": "call",
        "sample_id": sample.get("id"),
        "skill_used": skill_name,  # metadata for debugging
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": "", "tool_calls": tool_calls},
        ],
    }


def build_summarization_record(sample):
    tools_called = extract_tools_called(sample)
    if not tools_called:
        return None

    blocks = []
    for tc in tools_called:
        out = tc.get("output")
        if out is None:
            return None
        if not isinstance(out, str):
            out = json.dumps(out, separators=(",", ":"))
        escaped = out.replace('"', '\\"')
        blocks.append(
            f'{TOOL_RESPONSE_OPEN}response:TOOL{{value:"{escaped}"}}{TOOL_RESPONSE_CLOSE}'
        )
    user_msg = "".join(blocks)

    summary_array = sample.get("summary_array")
    if summary_array is None:
        prose = extract_assistant_summary(sample)
        if not prose:
            return None
        sentences = split_into_sentences(prose, len(tools_called))
        if len(sentences) != len(tools_called):
            return {"_invalid": True, "reason": "sentence count mismatch",
                    "sample_id": sample.get("id")}
        summary_array = sentences

    return {
        "task": "summarize",
        "sample_id": sample.get("id"),
        "messages": [
            {"role": "system", "content": SUMMARIZATION_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant",
             "content": json.dumps(summary_array, ensure_ascii=False)},
        ],
    }


def record_length(record):
    """Approximate char count for size warnings."""
    total = 0
    for m in record["messages"]:
        total += len(m.get("content", ""))
        for tc in m.get("tool_calls", []) or []:
            total += len(tc.get("function", {}).get("arguments", ""))
            total += len(tc.get("function", {}).get("name", ""))
    return total


# =============================================================================
# SECTION 8: MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dataset", required=True)
    ap.add_argument("--tool_schemas", required=True)
    ap.add_argument("--tool_catalog", required=True)
    ap.add_argument("--skills_catalog", default=None,
                    help="Optional skills_catalog.json (from build_skills_catalog.py)")
    ap.add_argument("--output_dir", default="./sft_out")
    args = ap.parse_args()

    random.seed(RANDOM_SEED)

    raw = load_jsonl(args.raw_dataset)
    schemas = load_tool_schemas(args.tool_schemas)
    tool_catalog = load_tool_catalog(args.tool_catalog)
    skill_catalog, skill_index = load_skills_catalog(args.skills_catalog)

    # Combined catalog: skills first (preferred), then tools
    catalog_entries = skill_catalog + tool_catalog

    print(f"Loaded {len(raw)} raw samples")
    print(f"  {len(schemas)} tool schemas")
    print(f"  {len(tool_catalog)} routable tools")
    print(f"  {len(skill_catalog)} routable skills")

    all_records = []
    dropped = defaultdict(int)
    dropped_examples = defaultdict(list)
    long_records = []
    routing_targets = Counter()

    def note_drop(bucket, sid, reason=""):
        dropped[bucket] += 1
        if len(dropped_examples[bucket]) < 5:
            dropped_examples[bucket].append(f"{sid}: {reason}")

    for sample in raw:
        sid = sample.get("id")

        # Routing
        try:
            r = build_routing_record(sample, catalog_entries, skill_index)
            if r:
                all_records.append(r)
                # Track what we routed to
                content = r["messages"][-1]["content"]
                try:
                    target = json.loads(content)
                    routing_targets[f"{target.get('source')}/{target.get('title')}"] += 1
                except Exception:
                    routing_targets["unrouted/prose"] += 1
        except Exception as e:
            note_drop("route_error", sid, str(e))

        # Tool calling
        try:
            r = build_toolcall_record(sample, schemas, skill_index)
            if r is None:
                note_drop("call_no_tools", sid)
            elif r.get("_invalid"):
                note_drop("call_invalid", r["sample_id"], r["reason"])
            else:
                all_records.append(r)
        except Exception as e:
            note_drop("call_error", sid, str(e))

        # Summarization
        try:
            r = build_summarization_record(sample)
            if r is None:
                note_drop("summarize_skipped", sid)
            elif r.get("_invalid"):
                note_drop("summarize_invalid", r["sample_id"], r["reason"])
            else:
                all_records.append(r)
        except Exception as e:
            note_drop("summarize_error", sid, str(e))

    # Check record lengths
    for r in all_records:
        n = record_length(r)
        if n > LENGTH_WARN_CHARS:
            long_records.append({"sample_id": r["sample_id"], "task": r["task"], "chars": n})

    # Stratified train/val split
    by_task = defaultdict(list)
    for r in all_records:
        by_task[r["task"]].append(r)

    train, val = [], []
    for task, recs in by_task.items():
        random.shuffle(recs)
        n_val = max(1, int(len(recs) * VAL_RATIO))
        val.extend(recs[:n_val])
        train.extend(recs[n_val:])

    random.shuffle(train)
    random.shuffle(val)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dir / "val.jsonl", "w", encoding="utf-8") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "raw_samples": len(raw),
        "total_records": len(all_records),
        "train_records": len(train),
        "val_records": len(val),
        "task_distribution": {
            "train": dict(Counter(r["task"] for r in train)),
            "val": dict(Counter(r["task"] for r in val)),
        },
        "routing_targets_top20": dict(routing_targets.most_common(20)),
        "long_records_count": len(long_records),
        "long_records_first10": long_records[:10],
        "dropped": dict(dropped),
        "dropped_examples_first5": {k: v for k, v in dropped_examples.items()},
    }
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n=== STATS ===")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nWrote {len(train)} train + {len(val)} val records to {out_dir}/")


if __name__ == "__main__":
    main()
