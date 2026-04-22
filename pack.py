#!/usr/bin/env python3
"""Pack the claude-proxy-plugin directory into a .difypkg file (ZIP format)."""
import os
import zipfile

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(os.path.dirname(PLUGIN_DIR), "claude-proxy.difypkg")

# Patterns to exclude (same as .gitignore defaults)
EXCLUDES = {
    "__pycache__",
    ".git",
    ".DS_Store",
    "*.pyc",
    "*.pyo",
    ".difyignore",
    "pack.py",
}

def should_exclude(path: str) -> bool:
    parts = path.replace("\\", "/").split("/")
    for part in parts:
        if part in EXCLUDES or part.endswith(".pyc") or part.endswith(".pyo"):
            return True
    return False

with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as zf:
    # Daemon parses ZIP comment as JSON; empty string causes JSON parse error → 400.
    # Provide a valid empty JSON object so the decoder can be created successfully.
    zf.comment = b'{"signature":"","time":""}'

    for root, dirs, files in os.walk(PLUGIN_DIR):
        # Skip excluded dirs in-place so os.walk won't descend into them
        dirs[:] = [d for d in dirs if not should_exclude(d)]
        for file in files:
            abs_path = os.path.join(root, file)
            rel_path = os.path.relpath(abs_path, PLUGIN_DIR).replace("\\", "/")
            if not should_exclude(rel_path):
                zf.write(abs_path, rel_path)
                print(f"  + {rel_path}")

print(f"\nPackaged → {OUTPUT}")
size_kb = os.path.getsize(OUTPUT) / 1024
print(f"Size: {size_kb:.1f} KB")
