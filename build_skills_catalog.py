"""
build_skills_catalog.py

Walks the skills directory (e.g. main/assets/skills/) and builds
skills_catalog.json containing one entry per skill, with frontmatter fields
(name, description, requires_tools) and the full SKILL.md body.

USAGE:
    python build_skills_catalog.py \
        --skills_dir path/to/assets/skills \
        --output skills_catalog.json

Each skill folder is expected to contain a SKILL.md with YAML frontmatter:
    ---
    name: "connectivity"
    description: "..."
    knoxagent:
      key: "connectivity"
      requires: { tools: ["connectivity_status", "connectivity_control"] }
    ---
    <markdown body>
"""

import json
import argparse
import re
from pathlib import Path


def parse_frontmatter(text):
    """
    Very small YAML frontmatter parser. Does NOT pull in PyYAML to keep this
    dependency-free. Handles the patterns seen in your SKILL.md files:
      - name: "value"
      - description: "value"
      - knoxagent:
          key: "value"
          requires: { tools: ["a", "b"] }
    Returns (frontmatter_dict, body_text).
    """
    if not text.startswith("---"):
        return {}, text

    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text

    fm_raw = parts[1]
    body = parts[2].lstrip("\n")

    fm = {}
    current_block = None

    for line in fm_raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        # Top-level scalar: `name: "..."` or `description: "..."`
        m = re.match(r'^(\w+)\s*:\s*"(.*)"\s*$', line)
        if m and not line.startswith(" "):
            fm[m.group(1)] = m.group(2)
            current_block = None
            continue

        # Top-level block start: `knoxagent:`
        m = re.match(r'^(\w+)\s*:\s*$', line)
        if m and not line.startswith(" "):
            current_block = m.group(1)
            fm[current_block] = {}
            continue

        # Indented child: ` key: "..."` or ` requires: { tools: [...] }`
        if current_block and line.startswith((" ", "\t")):
            child = stripped
            # key: "value"
            m = re.match(r'^(\w+)\s*:\s*"(.*)"\s*$', child)
            if m:
                fm[current_block][m.group(1)] = m.group(2)
                continue
            # requires: { tools: ["a", "b"] }
            m = re.match(r'^(\w+)\s*:\s*\{(.*)\}\s*$', child)
            if m:
                key = m.group(1)
                inner = m.group(2)
                # Parse `tools: ["a", "b"]` inside
                m2 = re.search(r'tools\s*:\s*\[([^\]]*)\]', inner)
                if m2:
                    items = re.findall(r'"([^"]+)"', m2.group(1))
                    fm[current_block][key] = {"tools": items}
                continue

    return fm, body


def extract_skill_info(skill_md_path):
    """Returns dict with name, description, requires_tools, body. None on failure."""
    try:
        text = skill_md_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"[WARN] could not read {skill_md_path}: {e}")
        return None

    fm, body = parse_frontmatter(text)
    if not fm.get("name"):
        # Fall back to folder name
        fm["name"] = skill_md_path.parent.name

    knoxagent = fm.get("knoxagent", {}) or {}
    requires = knoxagent.get("requires", {}) or {}
    requires_tools = requires.get("tools", []) if isinstance(requires, dict) else []

    return {
        "name": fm.get("name", skill_md_path.parent.name),
        "description": fm.get("description", ""),
        "requires_tools": requires_tools,
        "body": text,  # full SKILL.md including frontmatter — what inference loads
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skills_dir", required=True,
                    help="Path to skills folder (e.g. main/assets/skills)")
    ap.add_argument("--output", default="skills_catalog.json")
    args = ap.parse_args()

    skills_dir = Path(args.skills_dir)
    if not skills_dir.exists():
        raise SystemExit(f"Skills dir not found: {skills_dir}")

    skills = []
    skipped = []

    for child in sorted(skills_dir.iterdir()):
        if not child.is_dir():
            continue
        skill_md = child / "SKILL.md"
        if not skill_md.exists():
            skipped.append(f"{child.name}: no SKILL.md")
            continue
        info = extract_skill_info(skill_md)
        if info is None:
            skipped.append(f"{child.name}: parse failed")
            continue
        skills.append(info)
        print(f"  {info['name']:25s} requires_tools={info['requires_tools']}")

    out = {"version": 1, "skills": skills}
    Path(args.output).write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nWrote {len(skills)} skills to {args.output}")
    if skipped:
        print(f"\nSkipped {len(skipped)}:")
        for s in skipped:
            print(f"  - {s}")


if __name__ == "__main__":
    main()
