#!/usr/bin/env python3
"""Render the PP+MTP KB status dashboard from state.yml.

This is the ONLY way to read current state. There is no committed STATUS.md (a
rendered file would drift). Run on demand:

    VIRTUAL_ENV="$(pwd)/.venv" .venv/bin/python docs/superpowers/tools/build_status.py
    # or, to drop a throwaway file for a human to skim (gitignored/untracked anyway):
    ... build_status.py --write docs/superpowers/STATUS.generated.md

Source of truth: docs/superpowers/state.yml. Edit that, never the output.
Durable understanding lives in the bricks (research/pp-mtp/*); state.yml only points.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE = HERE.parent / "state.yml"

STATUS_ICON = {
    "done": "✅",
    "gate": "🔓",
    "in_progress": "🔄",
    "todo": "⬜",
    "blocked": "⛔",
}


def load_state() -> dict:
    try:
        import yaml  # PyYAML; present in the vllm .venv
    except ModuleNotFoundError:
        sys.exit(
            "PyYAML not found. Run via the vllm venv:\n"
            '  VIRTUAL_ENV="$(pwd)/.venv" .venv/bin/python '
            "docs/superpowers/tools/build_status.py"
        )
    with STATE.open() as f:
        return yaml.safe_load(f)


def render(state: dict) -> str:
    out: list[str] = []
    w = out.append

    w("# STATUS — PP + MTP spec-decode (generated; do not edit)")
    w("")
    w(f"> Generated from `state.yml` by `tools/build_status.py`. "
      f"Last source update: **{state.get('updated', '?')}**. "
      f"Edit `state.yml`, re-run the script — never hand-edit this output.")
    w("")

    w("## NEXT ACTION")
    w("")
    w(state.get("next_action", "_(unset)_").strip())
    w("")

    w("## Deliverables")
    w("")
    w("| | ID | What | Verify | Last result | PR |")
    w("|---|---|---|---|---|---|")
    for d in state.get("deliverables", []):
        icon = STATUS_ICON.get(d.get("status", ""), d.get("status", ""))
        verify = f"`{d['verify']}`" if d.get("verify") and d["verify"] != "-" else "—"
        lv = f" _(verified {d['last_verified']})_" if d.get("last_verified") else ""
        w(f"| {icon} | **{d['id']}** | {d.get('title','')} | {verify} | "
          f"{d.get('last_result','')}{lv} | {d.get('pr','')} |")
    w("")

    gc = state.get("green_check")
    if gc:
        w("**Green check (local, before any GPU work):**")
        w("")
        w("```bash")
        for line in gc:
            w(line)
        w("```")
        w("")

    if state.get("uncommitted_set"):
        w("**Uncommitted set:** " + state["uncommitted_set"].strip())
        w("")

    w("## Open questions")
    w("")
    w("| # | Question | Brick | Status |")
    w("|---|---|---|---|")
    for q in state.get("questions", []):
        w(f"| {q['id']} | {q.get('text','')} | {q.get('brick','')} | {q.get('status','')} |")
    w("")

    # Pointers from gate/in_progress deliverables = "where the live front is"
    fronts = [d for d in state.get("deliverables", [])
              if d.get("status") in ("gate", "in_progress")]
    if fronts:
        w("## Live front — read these bricks")
        w("")
        for d in fronts:
            if d.get("pointer"):
                w(f"- **{d['id']}** → `{d['pointer']}`")
        w("")

    w("---")
    w("_Runs: `runs.md`. Cold archive: `archive/` "
      "(full pre-image in ~/Nextcloud/Obsidian/.../pp-mtp-kb/). "
      "Teaching ladder (separate surface): `landing/`._")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", metavar="PATH", help="also write the dashboard to PATH")
    args = ap.parse_args()
    text = render(load_state())
    if args.write:
        Path(args.write).write_text(text)
        print(f"wrote {args.write}", file=sys.stderr)
    print(text)


if __name__ == "__main__":
    main()
