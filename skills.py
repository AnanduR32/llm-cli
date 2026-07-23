"""
Skills = markdown files in ./skills/, each with a YAML-ish header describing
what it's for, and body content that's only loaded into context when the
model actually asks for it. This mirrors how Claude's own skill system works:
the model sees a short catalog of what's available up front, and pulls in
the full instructions only when relevant -- keeps the system prompt small
until a skill is actually needed.

Skill file format (see skills/example_skill.md):

    ---
    name: git-commit-style
    description: Conventions to follow when writing git commit messages
    ---
    <full instructions here>
"""

import os
import re
import config

SKILL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "load_skill",
        "description": "Load the full instructions for a named skill when its topic becomes relevant to the current task.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The skill name, from the catalog"}
            },
            "required": ["name"],
        },
    },
}


def _parse_skill_file(path: str) -> dict:
    with open(path, "r") as f:
        raw = f.read()
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.DOTALL)
    if not match:
        return {"name": os.path.basename(path), "description": "", "body": raw}
    header, body = match.groups()
    meta = {}
    for line in header.strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return {"name": meta.get("name", os.path.basename(path)),
            "description": meta.get("description", ""),
            "body": body.strip()}


def load_catalog(skills_dir: str = config.SKILLS_DIR) -> list[dict]:
    """Returns [{name, description}] for every skill file, without full body."""
    if not os.path.isdir(skills_dir):
        return []
    catalog = []
    for fname in sorted(os.listdir(skills_dir)):
        if fname.endswith(".md"):
            parsed = _parse_skill_file(os.path.join(skills_dir, fname))
            catalog.append({"name": parsed["name"], "description": parsed["description"]})
    return catalog


def load_skill_body(name: str, skills_dir: str = config.SKILLS_DIR) -> str:
    """Returns the full instructions for one skill by name."""
    if not os.path.isdir(skills_dir):
        return f"No skills directory found at {skills_dir}"
    for fname in os.listdir(skills_dir):
        if fname.endswith(".md"):
            parsed = _parse_skill_file(os.path.join(skills_dir, fname))
            if parsed["name"] == name:
                return parsed["body"]
    return f"No skill named '{name}' found."


def catalog_as_prompt_block(skills_dir: str = config.SKILLS_DIR) -> str:
    """A short block to inject into the system prompt listing available skills."""
    catalog = load_catalog(skills_dir)
    if not catalog:
        return ""
    lines = ["Available skills (call load_skill to pull in full instructions when relevant):"]
    for s in catalog:
        lines.append(f"- {s['name']}: {s['description']}")
    return "\n".join(lines)
