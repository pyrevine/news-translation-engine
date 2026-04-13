"""FLORES-200 Korean↔English loader.

FLORES-200 (devtest: 1012 pairs) is the standard held-out multilingual benchmark.

Downloads the official tarball from Meta's public CDN on first use and caches
it under ``~/.cache/news-translation-engine/flores200/``. No HF authentication
needed.

The original ``facebook/flores`` and ``Muennighoff/flores200`` HF datasets are
script-based and deprecated in ``datasets>=4.0``. The maintained
``openlanguagedata/flores_plus`` is gated (requires access request). Downloading
the tarball directly sidesteps both issues.
"""

from __future__ import annotations

import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
CACHE_DIR = Path.home() / ".cache" / "news-translation-engine" / "flores200"
LANG_CODES = {"ko": "kor_Hang", "en": "eng_Latn"}


@dataclass(frozen=True)
class Pair:
    id: str
    ko: str
    en: str


def _ensure_dataset() -> Path:
    """Download and extract the FLORES-200 tarball if not already present.

    Returns the path to the extracted ``flores200_dataset`` directory.
    """
    extracted = CACHE_DIR / "flores200_dataset"
    if extracted.is_dir() and (extracted / "devtest").is_dir():
        return extracted

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tarball = CACHE_DIR / "flores200_dataset.tar.gz"
    if not tarball.exists():
        print(f"Downloading FLORES-200 (~25MB) from {FLORES_URL} ...")
        urllib.request.urlretrieve(FLORES_URL, tarball)

    print(f"Extracting {tarball} ...")
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(CACHE_DIR, filter="data")

    if not (extracted / "devtest").is_dir():
        raise RuntimeError(f"Unexpected tarball layout under {CACHE_DIR}")
    return extracted


def _read_lang_file(root: Path, split: str, lang: str) -> list[str]:
    """Read one language/split file. Each line is a sentence."""
    path = root / split / f"{LANG_CODES[lang]}.{split}"
    if not path.exists():
        raise FileNotFoundError(path)
    return [line.rstrip("\n") for line in path.read_text().splitlines()]


def load_flores_ko_en(split: str = "devtest") -> list[Pair]:
    """Load Korean-English parallel pairs from FLORES-200.

    Args:
        split: ``"dev"`` (997 pairs) or ``"devtest"`` (1012 pairs, standard
            test set).

    Returns:
        List of ``Pair`` objects. Sentences are aligned by line index.
    """
    if split not in {"dev", "devtest"}:
        raise ValueError(f"split must be 'dev' or 'devtest', got {split!r}")

    root = _ensure_dataset()
    ko = _read_lang_file(root, split, "ko")
    en = _read_lang_file(root, split, "en")
    if len(ko) != len(en):
        raise RuntimeError(f"FLORES row mismatch: ko={len(ko)} en={len(en)}")

    return [Pair(id=str(i), ko=k, en=e) for i, (k, e) in enumerate(zip(ko, en, strict=True))]


if __name__ == "__main__":
    pairs = load_flores_ko_en("devtest")
    print(f"Loaded {len(pairs)} pairs")
    print(f"First: id={pairs[0].id} ko={pairs[0].ko[:60]!r}")
    print(f"       en={pairs[0].en[:60]!r}")
