#!/usr/bin/env python3
"""
common.py

Shared, production-grade text normalization, similarity, and streaming I/O utilities
for the Amazon ML Challenge 2026 Business Entity Resolution pipeline.

Preserves exact NFKC + casefold + punctuation stripping + whitespace collapsing
semantics established during dataset profiling and pair analysis.
"""

from __future__ import annotations

import csv
import re
import shutil
import sys
import unicodedata
from pathlib import Path
from typing import Dict, Generator, List, Optional, Set, Tuple

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)
_DIGIT_RE = re.compile(r"\b\d+\b", flags=re.UNICODE)


def normalize_text(s: Optional[str]) -> str:
    """
    Unicode NFKC normalize, casefold, strip punctuation, collapse whitespace.
    Purely algorithmic -- no external dictionaries or lookups.
    Matches tools/analyze_training_pairs.py verbatim.
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.casefold()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def first_token(s_normalized: str) -> str:
    """
    Extracts the first whitespace-delimited token from a normalized string.
    Matches tools/analyze_training_pairs.py verbatim.
    """
    parts = s_normalized.split(" ")
    return parts[0] if parts and parts[0] != "" else ""


def tokenize_address(s_normalized: str) -> Set[str]:
    """Extracts unique non-empty tokens from a normalized address string."""
    if not s_normalized:
        return set()
    return {t for t in s_normalized.split(" ") if t != ""}


def extract_digits(s_normalized: str) -> Set[str]:
    """Extracts unique numeric digit tokens from a string."""
    if not s_normalized:
        return set()
    return set(_DIGIT_RE.findall(s_normalized))


def levenshtein_sim(s1: str, s2: str) -> float:
    """
    Computes normalized Levenshtein similarity score in [0.0, 1.0].
    sim = 1.0 - LevenshteinDistance / max(len(s1), len(s2))
    Fast DP implementation requiring 2 rows of memory.
    """
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    if len1 < len2:
        s1, s2 = s2, s1
        len1, len2 = len2, len1

    prev_row = list(range(len2 + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1] * (len2 + 1)
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row[j + 1] = min(
                curr_row[j] + 1,        # insertion
                prev_row[j + 1] + 1,    # deletion
                prev_row[j] + cost      # substitution
            )
        prev_row = curr_row

    dist = prev_row[len2]
    max_len = max(len1, len2)
    return max(0.0, min(1.0, 1.0 - (dist / max_len)))


def jaccard_sim(set1: Set[str], set2: Set[str]) -> float:
    """Computes Jaccard similarity over two token sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    if intersection == 0:
        return 0.0
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def token_sort_sim(s1: str, s2: str) -> float:
    """Computes Levenshtein similarity over alphabetically sorted tokens."""
    if not s1 or not s2:
        return 0.0
    sorted1 = " ".join(sorted(s1.split()))
    sorted2 = " ".join(sorted(s2.split()))
    return levenshtein_sim(sorted1, sorted2)


def digit_jaccard_sim(s1: str, s2: str) -> float:
    """Computes Jaccard similarity over extracted numeric digit tokens."""
    d1 = extract_digits(s1)
    d2 = extract_digits(s2)
    return jaccard_sim(d1, d2)


def stream_tsv_rows(
    path: Path,
    expected_columns: List[str],
    encoding: str = "utf-8",
    max_rows: Optional[int] = None
) -> Generator[Dict[str, str], None, None]:
    """
    Memory-safe line-by-line streaming reader for TSV files.
    Validates column headers and field counts.
    Supports max_rows for small-sample smoke testing.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path.resolve()}")

    with open(path, "r", encoding=encoding, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        try:
            raw_header = next(reader)
        except StopIteration:
            raise ValueError(f"File is completely empty: {path.resolve()}")

        header = [col.strip().lstrip("\ufeff") for col in raw_header]

        for col in expected_columns:
            if col not in header:
                raise KeyError(
                    f"Required column '{col}' missing from header in {path.name}. "
                    f"Found columns: {header}"
                )

        col_indices = {col: header.index(col) for col in expected_columns}
        expected_len = len(header)
        yielded_rows = 0

        for line_num, fields in enumerate(reader, start=2):
            if max_rows is not None and yielded_rows >= max_rows:
                break

            if not fields or (len(fields) == 1 and fields[0] == ""):
                continue

            if len(fields) != expected_len:
                raise ValueError(
                    f"Malformed row at line {line_num} in {path.name}. "
                    f"Expected {expected_len} fields, got {len(fields)}."
                )

            yielded_rows += 1
            yield {col: fields[col_indices[col]] for col in expected_columns}


def check_free_disk_space(path: Path, min_bytes: int = 3_221_225_472) -> int:
    """
    Verifies that target directory has at least min_bytes free (default 3 GiB).
    Returns available free bytes or raises RuntimeError.
    """
    target = path if path.exists() else path.parent
    if not target.exists():
        target.mkdir(parents=True, exist_ok=True)
    _total, _used, free = shutil.disk_usage(target)
    if free < min_bytes:
        raise RuntimeError(
            f"Insufficient disk space on {target.resolve()}. "
            f"Free: {format_bytes(free)}, Required minimum: {format_bytes(min_bytes)}"
        )
    return free


def format_bytes(num_bytes: int) -> str:
    """Formats byte counts into human-readable strings."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"