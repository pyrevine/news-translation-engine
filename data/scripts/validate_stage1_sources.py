"""Validate candidate parallel corpora for Stage 1 ko→en SFT.

Loads each candidate via ``datasets.load_dataset`` (streaming, so large
sources don't blow up the disk), probes the row schema, samples a handful
of pairs, and applies a coarse ko/en language sanity check. Writes a
summary JSON so downstream scripts can rely on a vetted source list.

Run locally (CPU only, ~1–3 min per source on a decent network):

    uv run python data/scripts/validate_stage1_sources.py
    uv run python data/scripts/validate_stage1_sources.py --limit 20
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
LATIN_RE = re.compile(r"[A-Za-z]")


@dataclass
class Candidate:
    name: str
    hf_path: str
    hf_config: str | None = None
    split: str = "train"
    license: str = "unknown"
    notes: str = ""


CANDIDATES: list[Candidate] = [
    Candidate(
        name="opus-news-commentary",
        hf_path="Helsinki-NLP/news_commentary",
        hf_config="en-ko",
        license="CC-BY-NC-SA-4.0 (News Commentary)",
        notes="News domain; stable OPUS source",
    ),
    Candidate(
        name="opus-100-en-ko",
        hf_path="Helsinki-NLP/opus-100",
        hf_config="en-ko",
        license="mixed (per OPUS-100)",
        notes="Mixed-domain benchmark corpus",
    ),
    Candidate(
        name="opus-paracrawl-en-ko",
        hf_path="Helsinki-NLP/opus_paracrawl",
        hf_config="en-ko",
        license="CC0 (ParaCrawl)",
        notes="Large, noisier — needs aggressive filtering",
    ),
    Candidate(
        name="lemon-mint-wiki-augmented-v1",
        hf_path="lemon-mint/korean_english_parallel_wiki_augmented_v1",
        license="unknown",
        notes="Personal HF dataset — may be missing or gated",
    ),
    Candidate(
        name="lemon-mint-parallel-v1.1",
        hf_path="lemon-mint/korean_parallel_sentences_v1.1",
        license="unknown",
        notes="Personal HF dataset — may be missing or gated",
    ),
]


def probe_ko_en(row: dict[str, Any]) -> tuple[str | None, str | None, str]:
    """Heuristically resolve ko/en text fields, returning (ko, en, schema_desc)."""
    if isinstance(row.get("translation"), dict):
        t = row["translation"]
        if "ko" in t and "en" in t:
            return t.get("ko"), t.get("en"), "translation.{ko,en}"
    if "ko" in row and "en" in row:
        return row["ko"], row["en"], "{ko,en}"
    if "korean" in row and "english" in row:
        return row["korean"], row["english"], "{korean,english}"
    if "src" in row and "tgt" in row:
        return row["src"], row["tgt"], "{src,tgt}"
    return None, None, f"unknown_schema: keys={list(row.keys())[:6]}"


def is_ko(text: str | None) -> bool:
    return bool(text) and bool(HANGUL_RE.search(text))


def is_en(text: str | None) -> bool:
    return bool(text) and bool(LATIN_RE.search(text)) and not HANGUL_RE.search(text)


def get_num_rows(cand: Candidate) -> tuple[int | None, str | None]:
    """Read row count from dataset metadata without downloading the payload."""
    try:
        from datasets import load_dataset_builder

        builder = load_dataset_builder(
            cand.hf_path, cand.hf_config, trust_remote_code=True
        )
        splits = builder.info.splits or {}
        sp = splits.get(cand.split)
        return (sp.num_examples if sp else None), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def validate_one(cand: Candidate, limit: int) -> dict[str, Any]:
    from datasets import load_dataset

    info: dict[str, Any] = {
        "name": cand.name,
        "hf_path": cand.hf_path,
        "hf_config": cand.hf_config,
        "license": cand.license,
        "notes": cand.notes,
    }

    num_rows, num_rows_err = get_num_rows(cand)
    info["num_rows"] = num_rows
    if num_rows_err:
        info["num_rows_error"] = num_rows_err

    try:
        ds = load_dataset(
            cand.hf_path,
            cand.hf_config,
            split=cand.split,
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as exc:
        info["status"] = "load_failed"
        info["error"] = f"{type(exc).__name__}: {exc}"
        return info

    samples: list[dict[str, Any]] = []
    columns: list[str] | None = None
    schema_desc = "?"
    sanity_pass = 0
    try:
        for i, row in enumerate(ds):
            if i == 0:
                columns = list(row.keys())
            ko, en, schema_desc = probe_ko_en(row)
            ok = is_ko(ko) and is_en(en)
            if ok:
                sanity_pass += 1
            samples.append({"ko": ko, "en": en, "ok": ok})
            if i + 1 >= limit:
                break
    except Exception as exc:
        info["status"] = "iter_failed"
        info["error"] = f"{type(exc).__name__}: {exc}"
        info["samples"] = samples
        return info

    info["columns"] = columns
    info["schema"] = schema_desc
    info["samples"] = samples
    info["sanity_pass_rate"] = round(sanity_pass / max(len(samples), 1), 2)
    if not samples:
        info["status"] = "empty"
    elif info["sanity_pass_rate"] >= 0.5:
        info["status"] = "ok"
    else:
        info["status"] = "low_sanity"
    return info


def print_report(results: list[dict[str, Any]]) -> None:
    for r in results:
        header = f"{r['name']} ({r['hf_path']}"
        if r.get("hf_config"):
            header += f"/{r['hf_config']}"
        header += ")"
        print(f"\n=== {header} ===")
        print(f"  status: {r.get('status')}")
        if r.get("error"):
            print(f"  error:  {r['error']}")
        print(f"  license: {r.get('license')}")
        print(f"  notes:   {r.get('notes')}")
        if r.get("num_rows") is not None:
            print(f"  rows:    {r['num_rows']:,}")
        if r.get("columns"):
            print(f"  columns: {r['columns']}")
            print(f"  schema:  {r.get('schema')}")
        for s in r.get("samples", [])[:3]:
            mark = "OK" if s["ok"] else "NG"
            ko = (s["ko"] or "")[:90].replace("\n", " ")
            en = (s["en"] or "")[:90].replace("\n", " ")
            print(f"    [{mark}] ko: {ko}")
            print(f"         en: {en}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=10, help="samples per candidate")
    ap.add_argument(
        "--output", default="data/processed/stage1_sources.json",
        help="summary JSON path (metadata only — safe to commit)",
    )
    args = ap.parse_args()

    results: list[dict[str, Any]] = []
    for cand in CANDIDATES:
        print(f"Validating {cand.name}...", file=sys.stderr)
        try:
            results.append(validate_one(cand, args.limit))
        except Exception as exc:
            print(traceback.format_exc(), file=sys.stderr)
            results.append({
                "name": cand.name,
                "hf_path": cand.hf_path,
                "status": "validator_crash",
                "error": f"{type(exc).__name__}: {exc}",
            })

    print_report(results)

    selected = [r["name"] for r in results if r.get("status") == "ok"]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"candidates": results, "selected": selected},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\nWrote {out}", file=sys.stderr)
    print(f"Selected (status=ok): {selected}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
