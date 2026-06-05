#!/usr/bin/env python3
"""check_refs.py — validate every `path:line` code reference in the landing docs.

Turns "code-referenced" from a claim into a test: for each `vllm/...py:NNN` (or
`tests/...py:NNN`) reference found in the given files, confirm the file exists and the
line number is in range. Stdlib only, no deps. Run from the repo root.

    python docs/superpowers/landing/check_refs.py            # check landing/*.md + *.html
    python docs/superpowers/landing/check_refs.py FILE...    # check specific files

Exit code 0 if all refs resolve, 1 otherwise. Line numbers drift as the tree changes —
this is the cheap guard that catches it before the docs silently rot.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# repo root = three levels up from docs/superpowers/landing/check_refs.py
ROOT = Path(__file__).resolve().parents[3]

# match `vllm/a/b.py:123` or `tests/a/b.py:123`, allowing a leading ~ on the line
REF = re.compile(r"\b((?:vllm|tests)/[\w/().\-]+\.py):~?(\d+)")
# match a bare `gpu_model_runner.py:123` (resolved via the basename index below)
REF_BARE = re.compile(r"\b([\w\-]+\.py):~?(\d+)")


def build_basename_index() -> dict[str, list[str]]:
    """basename -> [relative paths] for every .py under vllm/ and tests/."""
    index: dict[str, list[str]] = {}
    for top in ("vllm", "tests"):
        base = ROOT / top
        if not base.exists():
            continue
        for p in base.rglob("*.py"):
            index.setdefault(p.name, []).append(str(p.relative_to(ROOT)))
    return index


def line_count(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


BASENAMES = build_basename_index()


def check_file(doc: Path) -> list[str]:
    """Return a list of human-readable problems for one doc file."""
    problems: list[str] = []
    seen: set[tuple[str, int]] = set()
    text = doc.read_text(encoding="utf-8", errors="replace")

    def validate(rel: str, line: int) -> None:
        target = ROOT / rel
        if not target.exists():
            problems.append(f"  MISSING FILE  {rel}:{line}")
            return
        if line == 1:  # `:1` is our "whole file" convention
            return
        n = line_count(target)
        if line > n:
            problems.append(f"  OUT OF RANGE  {rel}:{line}  (file has {n} lines)")

    # 1) fully-pathed refs (authoritative)
    pathed: set[str] = set()
    for m in REF.finditer(text):
        rel, line = m.group(1), int(m.group(2))
        pathed.add(Path(rel).name)
        if (rel, line) in seen:
            continue
        seen.add((rel, line))
        validate(rel, line)

    # 2) bare `basename.py:NNN` refs — resolve via the index. These docs are about
    #    V1, and the repo has duplicate basenames across v0/v1, so prefer the unique
    #    `/v1/` candidate; if still ambiguous, skip silently (only report real breaks,
    #    never "couldn't check").
    for m in REF_BARE.finditer(text):
        name, line = m.group(1), int(m.group(2))
        paths = BASENAMES.get(name)
        if not paths:
            continue  # not a vllm/tests source file (likely a doc/tool name)
        if len(paths) == 1:
            rel = paths[0]
        else:
            v1 = [p for p in paths if "/v1/" in p]
            if len(v1) != 1:
                continue  # genuinely ambiguous — skip (path it fully if you want it checked)
            rel = v1[0]
        if (rel, line) in seen:
            continue
        seen.add((rel, line))
        validate(rel, line)
    return problems


def main(argv: list[str]) -> int:
    here = Path(__file__).resolve().parent
    if argv:
        files = [Path(a).resolve() for a in argv]
    else:
        files = sorted(here.glob("*.md")) + sorted(here.glob("*.html"))

    total_refs = 0
    total_bad = 0
    for doc in files:
        if not doc.exists():
            print(f"skip (not found): {doc}")
            continue
        refs = len(set(REF.findall(doc.read_text(encoding="utf-8", errors="replace"))))
        total_refs += refs
        problems = check_file(doc)
        total_bad += len(problems)
        status = "OK" if not problems else f"{len(problems)} BAD"
        print(f"{doc.name:28s} {refs:4d} refs  [{status}]")
        for p in problems:
            print(p)

    print(f"\n{total_refs} unique refs checked, {total_bad} broken.")
    return 0 if total_bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
