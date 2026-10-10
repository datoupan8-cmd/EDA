"""Check that a teammate's branch touches only its assigned module files."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SCOPES = {
    "component": ("pcb/component/", "pcb/text/", "configs/component/", "tests/component/", "docs/component/", "models/component/"),
    "pin": ("pcb/pin/", "configs/pin/", "tests/pin/", "docs/pin/", "models/pin/"),
    "wire_topology": ("pcb/wire_stage/", "pcb/topology_stage/", "configs/wire_topology/",
                      "tests/wire/", "tests/topology/", "docs/wire_topology/", "models/wire_topology/"),
}


def check_paths(owner: str, paths):
    return sorted(p for p in paths if not any(p.startswith(prefix) for prefix in SCOPES[owner]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--owner", choices=tuple(SCOPES), required=True)
    p.add_argument("--base", default="HEAD", help="HEAD for local edits; integration branch for a branch review")
    p.add_argument("--branch", action="store_true", help="Review committed branch changes from the merge base (for PR checks)")
    p.add_argument("--git", default="git")
    args = p.parse_args()
    def git(*cmd):
        return subprocess.check_output([args.git, *cmd], cwd=ROOT).decode("utf-8").splitlines()
    if args.branch:
        paths = set(git("diff", "--name-only", f"{args.base}...HEAD"))
    else:
        paths = set(git("diff", "--name-only", args.base)) | set(git("ls-files", "--others", "--exclude-standard"))
    invalid = check_paths(args.owner, paths)
    if invalid:
        print("Files outside your module (coordinate shared changes separately):")
        print("\n".join(invalid))
        return 1
    print(f"PASS: {args.owner}, {len(paths)} changed files, all within assigned scope")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
