"""Build the Stage 1 ko→en SFT dataset from vetted public corpora.

Sources (confirmed by data/processed/stage1_sources.json, 2026-04-14):
  1. lemon-mint/korean_english_parallel_wiki_augmented_v1  — wiki prose, {korean, english, score}
  2. lemon-mint/korean_parallel_sentences_v1.1             — general prose, {korean, english}
  3. Helsinki-NLP/opus-100 en-ko                           — mixed/colloquial, translation.{ko, en}

Pipeline
  1. Stream each source, project into {ko, en, source, score?}
  2. Language sanity via unicode heuristic (has Hangul / no Hangul + has Latin)
  3. Length + length-ratio filter (char-based, ratio computed en/ko)
  4. Deduplicate on NFC-normalized Korean (first-writer-wins — priority order above,
     so higher-quality sources shadow duplicates in lower-quality ones)
  5. Optional: drop bottom-N% of rows that ship a score field (only affects scored rows)
  6. Train/val split (95/5, fixed seed)
  7. Emit JSONL (gitignored) + stats JSON (committable)

Usage
  uv run python data/scripts/build_stage1_dataset.py
  uv run python data/scripts/build_stage1_dataset.py --max-per-source 200000
  uv run python data/scripts/build_stage1_dataset.py --max-per-source 1000  # smoke test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
LATIN_RE = re.compile(r"[A-Za-z]")

PROGRESS_EVERY = 50_000


@dataclass
class SourceSpec:
    name: str
    hf_path: str
    hf_config: str | None = None
    split: str = "train"
    ko_field: str = "ko"
    en_field: str = "en"
    score_field: str | None = None


SOURCES: list[SourceSpec] = [
    SourceSpec(
        name="lemon-mint-wiki-augmented-v1",
        hf_path="lemon-mint/korean_english_parallel_wiki_augmented_v1",
        ko_field="korean",
        en_field="english",
        score_field="score",
    ),
    SourceSpec(
        name="lemon-mint-parallel-v1.1",
        hf_path="lemon-mint/korean_parallel_sentences_v1.1",
        ko_field="korean",
        en_field="english",
    ),
    SourceSpec(
        name="opus-100-en-ko",
        hf_path="Helsinki-NLP/opus-100",
        hf_config="en-ko",
        ko_field="translation.ko",
        en_field="translation.en",
    ),
]


@dataclass
class FilterCfg:
    char_min: int = 10
    char_max_ko: int = 800
    char_max_en: int = 1600
    ratio_min: float = 0.3
    ratio_max: float = 5.0


@dataclass
class SourceStats:
    loaded: int = 0
    passed: int = 0
    rejected: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "passed": self.passed,
            "rejected": dict(sorted(self.rejected.items())),
        }


def get_field(row: dict[str, Any], path: str) -> Any:
    cur: Any = row
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def is_ko(text: str) -> bool:
    return bool(HANGUL_RE.search(text))


def is_en(text: str) -> bool:
    return bool(LATIN_RE.search(text)) and not HANGUL_RE.search(text)


def normalize_for_dedup(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip().lower())


def length_check(ko: str, en: str, cfg: FilterCfg) -> str | None:
    if len(ko) < cfg.char_min or len(ko) > cfg.char_max_ko:
        return "len_ko"
    if len(en) < cfg.char_min or len(en) > cfg.char_max_en:
        return "len_en"
    ratio = len(en) / max(len(ko), 1)
    if ratio < cfg.ratio_min or ratio > cfg.ratio_max:
        return "ratio"
    return None


def iter_source(src: SourceSpec, limit: int | None) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(
        src.hf_path, src.hf_config, split=src.split, streaming=True
    )
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield row


def process_source(
    src: SourceSpec,
    limit: int | None,
    seen_hashes: set[str],
    cfg: FilterCfg,
) -> tuple[list[dict[str, Any]], SourceStats, list[float]]:
    stats = SourceStats()
    kept: list[dict[str, Any]] = []
    scores: list[float] = []

    for row in iter_source(src, limit):
        stats.loaded += 1

        ko_raw = get_field(row, src.ko_field)
        en_raw = get_field(row, src.en_field)
        if not isinstance(ko_raw, str) or not isinstance(en_raw, str):
            stats.reject("schema")
            continue
        ko, en = ko_raw.strip(), en_raw.strip()
        if not ko or not en:
            stats.reject("empty")
            continue
        if not is_ko(ko):
            stats.reject("not_ko")
            continue
        if not is_en(en):
            stats.reject("not_en")
            continue

        len_reason = length_check(ko, en, cfg)
        if len_reason:
            stats.reject(len_reason)
            continue

        key = hashlib.md5(normalize_for_dedup(ko).encode("utf-8")).hexdigest()
        if key in seen_hashes:
            stats.reject("dup")
            continue
        seen_hashes.add(key)

        entry: dict[str, Any] = {"ko": ko, "en": en, "source": src.name}
        if src.score_field:
            score = get_field(row, src.score_field)
            if isinstance(score, (int, float)):
                entry["score"] = float(score)
                scores.append(float(score))

        kept.append(entry)
        stats.passed += 1

        if stats.loaded % PROGRESS_EVERY == 0:
            print(
                f"  ...{stats.loaded:,} loaded, {stats.passed:,} kept",
                file=sys.stderr,
            )

    return kept, stats, scores


def score_distribution(scores: list[float]) -> dict[str, float]:
    if not scores:
        return {}
    s = sorted(scores)
    n = len(s)
    return {
        "n": n,
        "min": round(s[0], 4),
        "p10": round(s[n // 10], 4),
        "median": round(s[n // 2], 4),
        "p90": round(s[n * 9 // 10], 4),
        "max": round(s[-1], 4),
    }


def apply_score_filter(
    rows: list[dict[str, Any]], drop_bottom_pct: float
) -> tuple[list[dict[str, Any]], int]:
    """Drop bottom N% of rows that carry a score field. Unscored rows are unaffected."""
    if drop_bottom_pct <= 0:
        return rows, 0
    scored = [r for r in rows if "score" in r]
    unscored = [r for r in rows if "score" not in r]
    if not scored:
        return rows, 0
    scores_sorted = sorted(r["score"] for r in scored)
    cutoff_idx = int(len(scores_sorted) * drop_bottom_pct / 100)
    cutoff = scores_sorted[cutoff_idx] if cutoff_idx < len(scores_sorted) else scores_sorted[-1]
    kept_scored = [r for r in scored if r["score"] >= cutoff]
    dropped = len(scored) - len(kept_scored)
    return kept_scored + unscored, dropped


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--max-per-source", type=int, default=None,
        help="cap rows streamed per source (default: full; useful for smoke tests)",
    )
    ap.add_argument(
        "--drop-bottom-score-pct", type=float, default=0.0,
        help="drop bottom N%% of rows with a score field (default: 0 — keep all)",
    )
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--char-min", type=int, default=10)
    ap.add_argument("--char-max-ko", type=int, default=800)
    ap.add_argument("--char-max-en", type=int, default=1600)
    ap.add_argument("--ratio-min", type=float, default=0.3)
    ap.add_argument("--ratio-max", type=float, default=5.0)
    args = ap.parse_args()

    cfg = FilterCfg(
        char_min=args.char_min,
        char_max_ko=args.char_max_ko,
        char_max_en=args.char_max_en,
        ratio_min=args.ratio_min,
        ratio_max=args.ratio_max,
    )
    random.seed(args.seed)

    all_rows: list[dict[str, Any]] = []
    per_source_stats: dict[str, Any] = {}
    per_source_scores: dict[str, dict[str, float]] = {}
    seen_hashes: set[str] = set()

    for src in SOURCES:
        print(f"\nProcessing {src.name} ({src.hf_path})...", file=sys.stderr)
        rows, stats, scores = process_source(src, args.max_per_source, seen_hashes, cfg)
        per_source_stats[src.name] = stats.to_dict()
        if scores:
            per_source_scores[src.name] = score_distribution(scores)
        all_rows.extend(rows)
        print(
            f"  total loaded={stats.loaded:,} passed={stats.passed:,}",
            file=sys.stderr,
        )

    all_rows, dropped_by_score = apply_score_filter(all_rows, args.drop_bottom_score_pct)

    random.shuffle(all_rows)
    n_val = int(len(all_rows) * args.val_fraction)
    val = all_rows[:n_val]
    train = all_rows[n_val:]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "stage1_train.jsonl"
    val_path = out_dir / "stage1_val.jsonl"
    stats_path = out_dir / "stage1_stats.json"

    write_jsonl(train_path, train)
    write_jsonl(val_path, val)

    src_counts: dict[str, int] = {}
    for r in train + val:
        src_counts[r["source"]] = src_counts.get(r["source"], 0) + 1

    stats_payload = {
        "config": {
            "seed": args.seed,
            "max_per_source": args.max_per_source,
            "val_fraction": args.val_fraction,
            "drop_bottom_score_pct": args.drop_bottom_score_pct,
            "char_min": cfg.char_min,
            "char_max_ko": cfg.char_max_ko,
            "char_max_en": cfg.char_max_en,
            "ratio_min": cfg.ratio_min,
            "ratio_max": cfg.ratio_max,
        },
        "per_source": per_source_stats,
        "score_distribution": per_source_scores,
        "final_source_distribution": dict(sorted(src_counts.items())),
        "total": {
            "before_score_filter": len(all_rows) + dropped_by_score,
            "after_score_filter": len(all_rows),
            "dropped_by_score_filter": dropped_by_score,
            "train": len(train),
            "val": len(val),
        },
    }
    stats_path.write_text(json.dumps(stats_payload, ensure_ascii=False, indent=2))

    print("\nDone.", file=sys.stderr)
    print(f"  train: {len(train):,} → {train_path}", file=sys.stderr)
    print(f"  val:   {len(val):,} → {val_path}", file=sys.stderr)
    print(f"  stats: {stats_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
