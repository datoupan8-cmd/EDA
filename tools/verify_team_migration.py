"""Verify migrated algorithm definitions against the two original Git commits."""
from __future__ import annotations
import argparse
import ast
import copy
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


class IgnoreImports(ast.NodeTransformer):
    def visit_Import(self, node):
        return None
    def visit_ImportFrom(self, node):
        return None


def definitions(source):
    result = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            result[node.name] = ast.dump(IgnoreImports().visit(copy.deepcopy(node)), include_attributes=False)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--git", default="git")
    p.add_argument("--output", default="runs/team_migration_verification.json")
    a = p.parse_args()
    manifest = json.loads((ROOT / "docs/team_migration_manifest.json").read_text(encoding="utf-8"))
    rows = []
    for item in manifest["migrations"]:
        source = subprocess.check_output([a.git, "show", f"{item['source_ref']}:{item['source']}"], cwd=ROOT).decode()
        original = definitions(source)
        migrated = definitions((ROOT / item['destination']).read_text(encoding="utf-8"))
        names = item.get("symbols", list(original))
        changed = [name for name in names if original.get(name) != migrated.get(name)]
        rows.append({"source": item['source'], "destination": item['destination'],
                     "compared_definitions": len(names), "changed": changed, "pass": not changed})
    result = {"method": "exact AST of every migrated function/class, excluding import statements",
              "scope_note": "Resource paths/import plumbing are checked by runtime parity separately",
              "pass": all(r['pass'] for r in rows), "files": rows}
    path = ROOT / a.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pass": result['pass'], "files": len(rows),
                      "definitions": sum(r['compared_definitions'] for r in rows),
                      "failures": [r for r in rows if not r['pass']]}, ensure_ascii=False))
    return int(not result['pass'])


if __name__ == "__main__":
    raise SystemExit(main())
