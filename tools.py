"""
Tools the model can call. Two parts per tool:
  1. A schema (OpenAI function-calling format) describing it to the model.
  2. A Python function that actually does the work.

Add new tools by adding to both TOOL_SCHEMAS and TOOL_FUNCTIONS with matching names.
Memory tools (remember/recall) are wired up in agent.py since they need
access to the VectorMemory instance.
"""

import os
import subprocess
import difflib
from pathlib import Path
import config

# The directory the agent/server was launched FROM (not where agent.py itself
# lives -- config.PROJECT_DIR is that). Captured once at import time so it
# can't drift even if something changes the process's cwd mid-session.
WORKING_ROOT = os.path.abspath(os.getcwd())


def _resolve_within_root(path: str) -> str | None:
    """
    Resolves `path` against WORKING_ROOT and confirms the result doesn't
    escape it (blocks '..' traversal and absolute paths pointing elsewhere).
    Returns the resolved absolute path if safe, None if it would escape.
    """
    candidate = os.path.abspath(os.path.join(WORKING_ROOT, path))
    try:
        if os.path.commonpath([WORKING_ROOT, candidate]) != WORKING_ROOT:
            return None
    except ValueError:
        # commonpath raises on Windows if paths are on different drives --
        # different drive always means "outside the root".
        return None
    return candidate


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files in a directory, optionally recursively. Use this to see what's in a codebase before reading specific files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory to list, default '.'"},
                    "recursive": {"type": "boolean", "description": "List all files in subdirectories too"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_edit",
            "description": "Show a unified diff between a file's current content and proposed new content, WITHOUT writing anything to disk. Always use this before write_file when suggesting changes to existing code, so the change can be reviewed first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "new_content": {"type": "string", "description": "The full proposed new content of the file"},
                },
                "required": ["path", "new_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a text file from disk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file on disk, overwriting it if it exists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command and return stdout/stderr. Use with care -- this executes directly on the host machine.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"}
                },
                "required": ["command"],
            },
        },
    },
]


def read_file(path: str) -> str:
    resolved = _resolve_within_root(path)
    if resolved is None:
        return (f"ERROR: refused -- '{path}' resolves outside the working directory "
                f"({WORKING_ROOT}). Access is restricted to files under it.")
    try:
        with open(resolved, "r") as f:
            return f.read()
    except Exception as e:
        return f"ERROR: {e}"


def list_directory(path: str = ".", recursive: bool = False) -> str:
    resolved = _resolve_within_root(path)
    if resolved is None:
        return (f"ERROR: refused -- '{path}' resolves outside the working directory "
                f"({WORKING_ROOT}). Access is restricted to files under it.")
    try:
        p = Path(resolved)
        if recursive:
            entries = sorted(
                str(f.relative_to(p)) for f in p.rglob("*")
                if f.is_file() and "__pycache__" not in f.parts
            )
        else:
            entries = sorted(f.name for f in p.iterdir())
        return "\n".join(entries) if entries else "(empty directory)"
    except Exception as e:
        return f"ERROR: {e}"


def propose_edit(path: str, new_content: str) -> str:
    resolved = _resolve_within_root(path)
    if resolved is None:
        return (f"ERROR: refused -- '{path}' resolves outside the working directory "
                f"({WORKING_ROOT}). Access is restricted to files under it.")
    try:
        old_content = ""
        if os.path.exists(resolved):
            with open(resolved, "r") as f:
                old_content = f.read()
        diff = difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"{path} (current)",
            tofile=f"{path} (proposed)",
        )
        diff_text = "".join(diff)
        return diff_text if diff_text else "No changes -- proposed content matches current content exactly."
    except Exception as e:
        return f"ERROR: {e}"


def write_file(path: str, content: str) -> str:
    resolved = _resolve_within_root(path)
    if resolved is None:
        return (f"ERROR: refused -- '{path}' resolves outside the working directory "
                f"({WORKING_ROOT}). Access is restricted to files under it.")
    if resolved in config.PROTECTED_FILES:
        return (f"ERROR: refused to write to '{path}' -- this is one of the agent's "
                f"own core files. Edit it manually outside the agent if you really mean to.")
    try:
        os.makedirs(os.path.dirname(resolved) or ".", exist_ok=True)
        with open(resolved, "w") as f:
            f.write(content)
        return f"Wrote {len(content)} chars to {path}"
    except Exception as e:
        return f"ERROR: {e}"


def run_shell(command: str) -> str:
    # Heuristic guards -- NOT airtight (see README), but block the common cases.
    lower = command.lower()

    # 1. Protected-file targeting, as before.
    dangerous_ops = [">", "rm ", "del ", "move ", "mv ", "remove-item", "erase "]
    for protected_path in config.PROTECTED_FILES:
        fname = os.path.basename(protected_path).lower()
        if fname in lower and any(op in lower for op in dangerous_ops):
            return (f"ERROR: refused -- this command appears to target the protected "
                    f"core file '{fname}' with a write/delete operation. "
                    f"Run it manually outside the agent if intended.")

    # 2. Obvious attempts to leave the working directory.
    if ".." in command:
        return (f"ERROR: refused -- command contains '..', which could escape the "
                f"working directory ({WORKING_ROOT}). Rephrase without parent-directory "
                f"references.")

    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=60,
            cwd=WORKING_ROOT, encoding="utf-8", errors="replace"
        )
        return f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    except Exception as e:
        return f"ERROR: {e}"


TOOL_FUNCTIONS = {
    "read_file": read_file,
    "write_file": write_file,
    "run_shell": run_shell,
    "list_directory": list_directory,
    "propose_edit": propose_edit,
}