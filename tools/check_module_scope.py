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

# This explicitly named release branch carries both published owners. Ordinary
# feature/component-* and feature/pin-* branches keep their strict single owner.
SCOPES["component_pin_release"] = SCOPES["component"] + SCOPES["pin"]
SHARED_PATHS = {
    "component_pin_release": frozenset({
        ".gitattributes", ".gitignore", "README.md", "THIRD_PARTY_NOTICES.md",
        "requirements.txt", "models/SHA256SUMS.json",
        "tests/contracts/test_team_module_slots.py",
        "tools/benchmark_component_ocr_decode.py",
        "tools/evaluate_component_local_joint.py",
        "tools/evaluate_component_ocr_ablation.py", "tools/evaluate_component_p01.py",
        ".github/workflows/team-modules.yml", "tools/check_module_scope.py",
        "tests/contracts/test_module_scope.py",
    }),
}


def check_paths(owner: str, paths):
    shared = SHARED_PATHS.get(owner, frozenset())
    return sorted(p for p in paths if p not in shared and
                  not any(p.startswith(prefix) for prefix in SCOPES[owner]))


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
