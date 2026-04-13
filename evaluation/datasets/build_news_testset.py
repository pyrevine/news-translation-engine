"""Build a custom Korean→English news test set (~100 pairs).

**This is a skeleton.** Actual test set construction requires manual curation:

1. Select source articles from CC-licensed or fair-use-compatible sources
   (e.g., Creative Commons news feeds, research-permitted archives).
2. For each article, pick 1-3 informative sentences.
3. Produce reference English translation (human-written or human-verified).
4. Save to `evaluation/datasets/news_testset_v1.jsonl` (gitignored).
5. Save metadata (source URL, date, category) to
   `evaluation/datasets/news_testset_v1.meta.json` (committed).

The actual .jsonl must NOT be committed — see `.gitignore`.

This script provides:
- A `load_news_testset()` loader matching the FLORES loader interface.
- A `validate_testset()` checker (schema, length, duplicates).
- A CLI for validation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PATH = Path("evaluation/datasets/news_testset_v1.jsonl")


@dataclass(frozen=True)
class NewsPair:
    id: str
    ko: str
    en: str
    source: str
    date: str
    category: str


REQUIRED_FIELDS = {"id", "ko", "en", "source", "date", "category"}


def load_news_testset(path: Path = DEFAULT_PATH) -> list[NewsPair]:
    """Load the custom news test set from JSONL."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. See module docstring for construction steps."
        )
    pairs: list[NewsPair] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pairs.append(NewsPair(**{k: row[k] for k in REQUIRED_FIELDS}))
    return pairs


def validate_testset(pairs: list[NewsPair]) -> list[str]:
    """Run sanity checks. Returns list of warning/error messages (empty if clean)."""
    msgs: list[str] = []
    ids = [p.id for p in pairs]
    if len(set(ids)) != len(ids):
        msgs.append(f"duplicate ids: {len(ids) - len(set(ids))}")
    kos = [p.ko for p in pairs]
    if len(set(kos)) != len(kos):
        msgs.append(f"duplicate korean sentences: {len(kos) - len(set(kos))}")
    for p in pairs:
        if not (5 <= len(p.ko) <= 500):
            msgs.append(f"{p.id}: ko length out of range ({len(p.ko)})")
        if not (5 <= len(p.en) <= 500):
            msgs.append(f"{p.id}: en length out of range ({len(p.en)})")
        ratio = len(p.en) / max(len(p.ko), 1)
        if not (0.3 <= ratio <= 5.0):
            msgs.append(f"{p.id}: en/ko length ratio {ratio:.2f} out of [0.3, 5.0]")
    return msgs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=DEFAULT_PATH)
    args = ap.parse_args()

    pairs = load_news_testset(args.path)
    print(f"Loaded {len(pairs)} pairs from {args.path}")
    warnings = validate_testset(pairs)
    if warnings:
        print(f"\n{len(warnings)} issues:")
        for w in warnings:
            print(f"  - {w}")
        return 1
    print("OK — all checks pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
