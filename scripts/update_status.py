#!/usr/bin/env python3
"""Update docs/STATUS.md milestone status.

Usage:
  python scripts/update_status.py M0 --tests "pytest tests/test_trees.py" --notes "Phase 0 green"
"""

import argparse
from datetime import date
from pathlib import Path
import re
import sys

STATUS_PATH = Path("docs/STATUS.md")

MILESTONES = {
    "M0": "M0 Foundation",
    "M1": "M1 MSC Core",
    "M2": "M2 Graph Partitioning",
    "M3": "M3 Branch Lengths",
    "M4": "M4 EM Loop",
    "M5": "M5 CLI",
    "M6": "M6 Validation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update milestone status in docs/STATUS.md")
    parser.add_argument("milestone", choices=MILESTONES.keys(), help="Milestone ID (e.g., M0)")
    parser.add_argument("--date", default=str(date.today()), help="Date achieved (YYYY-MM-DD)")
    parser.add_argument("--tests", required=True, help="Tests/evidence command(s)")
    parser.add_argument("--notes", default="", help="Short notes")
    parser.add_argument("--complete", action="store_true", help="Mark as complete")
    return parser.parse_args()


def update_status(text: str, milestone: str, achieved_date: str, tests: str, notes: str, complete: bool) -> str:
    title = MILESTONES[milestone]

    # Find the milestone block
    pattern = rf"^## {re.escape(title)}\n(?P<body>(?:.+\n)+?)(?=^## |\Z)"
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise ValueError(f"Milestone section not found: {title}")

    body = match.group("body")

    # Update completion checkbox
    if complete:
        body = re.sub(r"^- \[.\] Complete\n", "- [x] Complete\n", body, flags=re.MULTILINE)

    # Update date
    body = re.sub(r"^(- Date achieved:) .*\n", rf"\1 {achieved_date}\n", body, flags=re.MULTILINE)

    # Update tests
    body = re.sub(r"^\s*- Tests: .*\n", f"  - Tests: {tests}\n", body, flags=re.MULTILINE)

    # Update notes
    notes_value = notes if notes else ""
    body = re.sub(r"^\s*- Notes: .*\n", f"  - Notes: {notes_value}\n", body, flags=re.MULTILINE)

    updated = text[: match.start("body")] + body + text[match.end("body"):]
    return updated


def main() -> int:
    args = parse_args()

    if not STATUS_PATH.exists():
        print(f"Missing {STATUS_PATH}.", file=sys.stderr)
        return 2

    text = STATUS_PATH.read_text(encoding="utf-8")
    updated = update_status(
        text,
        milestone=args.milestone,
        achieved_date=args.date,
        tests=args.tests,
        notes=args.notes,
        complete=args.complete,
    )
    STATUS_PATH.write_text(updated, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
