#!/usr/bin/env python3
"""
profile_dataset.py

Read-only, streaming PROFILING tool for the Amazon ML Challenge 2026
"Business Entity Resolution" dataset. This is a statistics/integrity
report generator only -- it performs no cleaning, no ML, no matching.

This script has NOT been executed against the real dataset. It only runs
when you copy it into your project and execute it on the machine where
the dataset actually lives. Nothing in reports/dataset_profile.* is real
until you run it and open those files yourself.

WHAT THIS SCRIPT DOES
----------------------
- Recursively locates the expected TRAIN/TEST files under the project
  root (does not assume a fixed path).
- Streams each file line-by-line. Per-file aggregate counters/sets are
  kept in memory (this is unavoidable for exact duplicate-detection and
  cross-file ID-reference checks); the raw rows themselves are never
  held in memory or loaded into a DataFrame.
- For each source file (train_source1/2/3, test_source1/2/3): basic
  column statistics, entity_id integrity (prefix, duplicates, empties),
  text-quality signals for business_name/business_address, and country
  distribution.
- For train_ground_truth.tsv: row/duplicate/empty checks, comma-parsed
  match-list statistics, and cross-file referential integrity against
  train_source1/2/3.
- Cross-source duplicate-value diagnostics (exact string duplicates only
  -- this is NOT entity matching and is not interpreted as such).
- Train vs. test structural comparison.
- Writes reports/dataset_profile.json and reports/dataset_profile.md,
  with findings explicitly separated into:
    1. confirmed issues (integrity violations actually observed)
    2. observed statistics (neutral facts about the data)
    3. items requiring human/technical review (ambiguous or only
       possibly-noteworthy patterns -- NOT asserted as problems)

WHAT THIS SCRIPT DOES NOT DO
-----------------------------
- It NEVER modifies, renames, moves, deletes, rewrites, or normalizes any
  raw dataset file.
- It performs NO ML, feature engineering, blocking, matching, training,
  or prediction. Exact-string duplicate counts are reported as raw
  counts only and are explicitly labeled "not entity matching".
- It makes NO internet or external API/database calls.
- It does NOT guess column meaning: if an expected column name is not
  found in a file's header, the related statistics are skipped and the
  omission is reported under "items requiring review" rather than
  silently assumed or worked around.
- It does NOT report a data characteristic as a "problem" merely because
  it is unusual (e.g. non-ASCII text, which is expected due to
  transliteration).

USAGE (Windows PowerShell)
---------------------------
    cd D:\\Documents\\PARTH\\AmazonMLchallenge
    python tools\\profile_dataset.py

Optional flags:
    --root <path>                 Directory to search from (default: parent
                                   of the "tools" folder this script lives in)
    --encoding NAME                Encoding to use when reading files
                                   (default: utf-8)
    --max-examples N               Max example values/IDs stored per
                                   violation category in the report
                                   (default: 50). Counts are always exact;
                                   this only caps how many examples are
                                   listed.
    --top-n-values N               How many most-frequent values to list
                                   for country / duplicate-name / etc.
                                   tables (default: 25).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Expected dataset shape (per the official problem statement / prior audit).
# Used only to locate files by name and to know which column names to look
# for -- never to assume a row or column is present or well-formed.
# ---------------------------------------------------------------------------

SOURCE_FILES = {
    "train_source1.tsv": ("train", "1"),
    "train_source2.tsv": ("train", "2"),
    "train_source3.tsv": ("train", "3"),
    "test_source1.tsv": ("test", "1"),
    "test_source2.tsv": ("test", "2"),
    "test_source3.tsv": ("test", "3"),
}
GROUND_TRUTH_FILE = "train_ground_truth.tsv"
ALL_EXPECTED_FILES = list(SOURCE_FILES.keys()) + [GROUND_TRUTH_FILE]

EXPECTED_PREFIX = {"1": "S1-", "2": "S2-", "3": "S3-"}

EXPECTED_SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]

# Candidate ground-truth column names. The script looks for an exact match
# first; if none is found it searches for a plausible match and flags the
# choice explicitly in the report rather than silently assuming.
GT_ID_COL_CANDIDATES = ["source1_entity_id", "source1_id", "entity_id_source1"]
GT_MATCH_COL_CANDIDATES = ["matched_entity_ids", "matched_ids", "matches", "matched_entity_id"]

SKIP_DIR_NAMES = {
    ".git", ".hg", ".svn", "__pycache__", "node_modules",
    ".venv", "venv", "env", ".mypy_cache", ".pytest_cache",
    ".idea", ".vscode",
}


# ---------------------------------------------------------------------------
# Small streaming aggregation helpers
# ---------------------------------------------------------------------------

class LengthStats:
    __slots__ = ("count", "total", "min", "max")

    def __init__(self):
        self.count = 0
        self.total = 0
        self.min = None
        self.max = None

    def add(self, length: int):
        self.count += 1
        self.total += length
        if self.min is None or length < self.min:
            self.min = length
        if self.max is None or length > self.max:
            self.max = length

    def to_dict(self) -> dict:
        mean = (self.total / self.count) if self.count else None
        return {
            "count": self.count,
            "min": self.min,
            "max": self.max,
            "mean": round(mean, 2) if mean is not None else None,
        }


def is_non_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


def top_n_counter(counter: Counter, n: int) -> list:
    return [{"value": v, "count": c} for v, c in counter.most_common(n)]


def _walk_skipping(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        yield dirpath, dirnames, filenames


def find_expected_files(root: Path) -> dict:
    """Return {expected_filename: Path or None} by recursive search."""
    found = {name: None for name in ALL_EXPECTED_FILES}
    for dirpath, _dirnames, filenames in _walk_skipping(root):
        for name in filenames:
            if name in found and found[name] is None:
                found[name] = Path(dirpath) / name
    return found


def col_index(header: list, name: str) -> Optional[int]:
    try:
        return header.index(name)
    except ValueError:
        return None


def find_gt_column(header: list, candidates: list) -> tuple:
    """Return (index_or_None, matched_name_or_None, ambiguous_note_or_None)."""
    for c in candidates:
        idx = col_index(header, c)
        if idx is not None:
            return idx, c, None
    # No exact match among candidates: report ambiguity, do not guess further.
    return None, None, (
        f"No column matching any of {candidates} was found in header {header}. "
        f"Related ground-truth checks were skipped."
    )


# ---------------------------------------------------------------------------
# Source file profiling
# ---------------------------------------------------------------------------

@dataclass
class SourceProfile:
    filename: str
    path: str
    split: str
    source_num: str
    read_error: Optional[str] = None
    encoding_used: Optional[str] = None
    header: Optional[list] = None
    row_count: int = 0  # data rows only (excludes header/blank lines)
    blank_row_count: int = 0

    missing_expected_columns: list = field(default_factory=list)

    entity_id_total: int = 0
    entity_id_empty: int = 0
    entity_id_unique_count: int = 0
    entity_id_duplicate_value_count: int = 0  # distinct values that repeat
    entity_id_duplicate_row_count: int = 0    # extra rows beyond first occurrence
    entity_id_duplicate_examples: list = field(default_factory=list)
    entity_id_prefix_violations: int = 0
    entity_id_prefix_violation_examples: list = field(default_factory=list)

    per_column_empty: dict = field(default_factory=dict)
    per_column_whitespace_only: dict = field(default_factory=dict)

    business_name_length: Optional[dict] = None
    business_address_length: Optional[dict] = None
    business_name_leading_trailing_ws: int = 0
    business_address_leading_trailing_ws: int = 0
    business_name_non_ascii_count: int = 0
    business_address_non_ascii_count: int = 0

    country_distinct_count: int = 0
    country_top_values: list = field(default_factory=list)
    country_missing_or_empty: int = 0

    duplicate_business_name_value_count: int = 0
    duplicate_business_name_row_count: int = 0
    duplicate_business_address_value_count: int = 0
    duplicate_business_address_row_count: int = 0
    duplicate_name_address_value_count: int = 0
    duplicate_name_address_row_count: int = 0

    # Not serialized directly; used for cross-file checks.
    entity_id_set: set = field(default_factory=set, repr=False)


def profile_source_file(path: Path, split: str, source_num: str, encoding: str,
                         max_examples: int, top_n: int) -> SourceProfile:
    prof = SourceProfile(filename=path.name, path=str(path), split=split, source_num=source_num)

    header = None
    header_len = None
    idx = {}

    entity_id_counts = Counter()
    country_counts = Counter()
    name_counts = Counter()
    address_counts = Counter()
    name_address_counts = Counter()

    empty_counts = defaultdict(int)
    whitespace_counts = defaultdict(int)

    name_len = LengthStats()
    addr_len = LengthStats()

    expected_prefix = EXPECTED_PREFIX.get(source_num)

    try:
        with open(path, "r", encoding=encoding, newline="") as f:
            prof.encoding_used = encoding
            for raw_line in f:
                line = raw_line.rstrip("\r\n")
                if line == "":
                    prof.blank_row_count += 1
                    continue

                fields = line.split("\t")

                if header is None:
                    header = fields
                    header_len = len(fields)
                    prof.header = header
                    idx["entity_id"] = col_index(header, "entity_id")
                    idx["business_name"] = col_index(header, "business_name")
                    idx["business_address"] = col_index(header, "business_address")
                    idx["country"] = col_index(header, "country")
                    prof.missing_expected_columns = [
                        c for c in EXPECTED_SOURCE_COLUMNS if idx.get(c) is None
                    ]
                    for c in header:
                        empty_counts[c] = 0
                        whitespace_counts[c] = 0
                    continue

                # Skip rows that structurally don't match the header field
                # count -- these are already reported by the audit script
                # and must not be silently reinterpreted here.
                if len(fields) != header_len:
                    continue

                prof.row_count += 1

                for col_name, col_idx in zip(header, range(header_len)):
                    val = fields[col_idx]
                    if val == "":
                        empty_counts[col_name] += 1
                    elif val.strip() == "":
                        whitespace_counts[col_name] += 1

                if idx.get("entity_id") is not None:
                    eid = fields[idx["entity_id"]]
                    prof.entity_id_total += 1
                    if eid == "":
                        prof.entity_id_empty += 1
                    else:
                        entity_id_counts[eid] += 1
                        prof.entity_id_set.add(eid)
                        if expected_prefix and not eid.startswith(expected_prefix):
                            prof.entity_id_prefix_violations += 1
                            if len(prof.entity_id_prefix_violation_examples) < max_examples:
                                prof.entity_id_prefix_violation_examples.append(eid)

                if idx.get("business_name") is not None:
                    name = fields[idx["business_name"]]
                    name_len.add(len(name))
                    if name != name.strip() and name != "":
                        prof.business_name_leading_trailing_ws += 1
                    if is_non_ascii(name):
                        prof.business_name_non_ascii_count += 1
                    if name != "":
                        name_counts[name] += 1

                if idx.get("business_address") is not None:
                    addr = fields[idx["business_address"]]
                    addr_len.add(len(addr))
                    if addr != addr.strip() and addr != "":
                        prof.business_address_leading_trailing_ws += 1
                    if is_non_ascii(addr):
                        prof.business_address_non_ascii_count += 1
                    if addr != "":
                        address_counts[addr] += 1

                if idx.get("business_name") is not None and idx.get("business_address") is not None:
                    name = fields[idx["business_name"]]
                    addr = fields[idx["business_address"]]
                    if name != "" and addr != "":
                        name_address_counts[(name, addr)] += 1

                if idx.get("country") is not None:
                    country = fields[idx["country"]]
                    if country.strip() == "":
                        prof.country_missing_or_empty += 1
                    else:
                        country_counts[country] += 1

    except UnicodeDecodeError as e:
        prof.read_error = f"UnicodeDecodeError with encoding='{encoding}': {e}"
        return prof
    except OSError as e:
        prof.read_error = f"OS error reading file: {e}"
        return prof

    prof.entity_id_unique_count = len(entity_id_counts)
    dup_values = [ (v, c) for v, c in entity_id_counts.items() if c > 1 ]
    prof.entity_id_duplicate_value_count = len(dup_values)
    prof.entity_id_duplicate_row_count = sum(c - 1 for _v, c in dup_values)
    prof.entity_id_duplicate_examples = [v for v, _c in dup_values[:max_examples]]

    prof.per_column_empty = dict(empty_counts)
    prof.per_column_whitespace_only = dict(whitespace_counts)

    prof.business_name_length = name_len.to_dict() if idx.get("business_name") is not None else None
    prof.business_address_length = addr_len.to_dict() if idx.get("business_address") is not None else None

    prof.country_distinct_count = len(country_counts)
    prof.country_top_values = top_n_counter(country_counts, top_n)

    name_dups = [(v, c) for v, c in name_counts.items() if c > 1]
    prof.duplicate_business_name_value_count = len(name_dups)
    prof.duplicate_business_name_row_count = sum(c - 1 for _v, c in name_dups)

    addr_dups = [(v, c) for v, c in address_counts.items() if c > 1]
    prof.duplicate_business_address_value_count = len(addr_dups)
    prof.duplicate_business_address_row_count = sum(c - 1 for _v, c in addr_dups)

    na_dups = [(v, c) for v, c in name_address_counts.items() if c > 1]
    prof.duplicate_name_address_value_count = len(na_dups)
    prof.duplicate_name_address_row_count = sum(c - 1 for _v, c in na_dups)

    # Stash raw counters temporarily for cross-file use (country sets etc.)
    prof._country_set = set(country_counts.keys())  # type: ignore[attr-defined]

    return prof


# ---------------------------------------------------------------------------
# Ground-truth profiling
# ---------------------------------------------------------------------------

@dataclass
class GroundTruthProfile:
    filename: str
    path: str
    read_error: Optional[str] = None
    encoding_used: Optional[str] = None
    header: Optional[list] = None
    row_count: int = 0
    blank_row_count: int = 0

    id_column_used: Optional[str] = None
    match_column_used: Optional[str] = None
    column_detection_note: Optional[str] = None

    source1_id_total: int = 0
    source1_id_empty: int = 0
    source1_id_duplicate_value_count: int = 0
    source1_id_duplicate_row_count: int = 0
    source1_id_duplicate_examples: list = field(default_factory=list)

    singleton_row_count: int = 0  # empty match list
    match_count_distribution: dict = field(default_factory=dict)  # {num_matches: num_rows}

    unknown_source1_id_count: int = 0
    unknown_source1_id_examples: list = field(default_factory=list)

    unknown_matched_id_count: int = 0
    unknown_matched_id_examples: list = field(default_factory=list)

    s1_id_used_as_matched_count: int = 0
    s1_id_used_as_matched_examples: list = field(default_factory=list)


def profile_ground_truth(path: Path, encoding: str, max_examples: int,
                          s1_ids: set, s2_ids: set, s3_ids: set) -> GroundTruthProfile:
    prof = GroundTruthProfile(filename=path.name, path=str(path))

    header = None
    header_len = None
    id_idx = None
    match_idx = None

    s1_counts = Counter()
    match_dist = Counter()

    try:
        with open(path, "r", encoding=encoding, newline="") as f:
            prof.encoding_used = encoding
            for raw_line in f:
                line = raw_line.rstrip("\r\n")
                if line == "":
                    prof.blank_row_count += 1
                    continue

                fields = line.split("\t")

                if header is None:
                    header = fields
                    header_len = len(fields)
                    prof.header = header
                    id_idx, id_name, id_note = find_gt_column(header, GT_ID_COL_CANDIDATES)
                    match_idx, match_name, match_note = find_gt_column(header, GT_MATCH_COL_CANDIDATES)
                    prof.id_column_used = id_name
                    prof.match_column_used = match_name
                    notes = [n for n in (id_note, match_note) if n]
                    prof.column_detection_note = " | ".join(notes) if notes else None
                    continue

                if len(fields) != header_len:
                    continue

                prof.row_count += 1

                if id_idx is not None:
                    sid = fields[id_idx]
                    prof.source1_id_total += 1
                    if sid == "":
                        prof.source1_id_empty += 1
                    else:
                        s1_counts[sid] += 1
                        if s1_ids and sid not in s1_ids:
                            prof.unknown_source1_id_count += 1
                            if len(prof.unknown_source1_id_examples) < max_examples:
                                prof.unknown_source1_id_examples.append(sid)

                if match_idx is not None:
                    raw_match = fields[match_idx]
                    matched_ids = [m.strip() for m in raw_match.split(",") if m.strip() != ""]
                    n_matches = len(matched_ids)
                    match_dist[n_matches] += 1
                    if n_matches == 0:
                        prof.singleton_row_count += 1
                    for mid in matched_ids:
                        known = (mid in s2_ids) or (mid in s3_ids)
                        if not known:
                            prof.unknown_matched_id_count += 1
                            if len(prof.unknown_matched_id_examples) < max_examples:
                                prof.unknown_matched_id_examples.append(mid)
                        if s1_ids and mid in s1_ids:
                            prof.s1_id_used_as_matched_count += 1
                            if len(prof.s1_id_used_as_matched_examples) < max_examples:
                                prof.s1_id_used_as_matched_examples.append(mid)

    except UnicodeDecodeError as e:
        prof.read_error = f"UnicodeDecodeError with encoding='{encoding}': {e}"
        return prof
    except OSError as e:
        prof.read_error = f"OS error reading file: {e}"
        return prof

    dup_values = [(v, c) for v, c in s1_counts.items() if c > 1]
    prof.source1_id_duplicate_value_count = len(dup_values)
    prof.source1_id_duplicate_row_count = sum(c - 1 for _v, c in dup_values)
    prof.source1_id_duplicate_examples = [v for v, _c in dup_values[:max_examples]]

    prof.match_count_distribution = {str(k): v for k, v in sorted(match_dist.items())}

    return prof


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def build_report(root: Path, expected_files: dict, source_profiles: dict,
                  gt_profile: Optional[GroundTruthProfile], args) -> tuple:
    timestamp = datetime.now(timezone.utc).isoformat()

    confirmed_issues = []
    observed_statistics = []
    needs_review = []

    missing_files = [name for name, p in expected_files.items() if p is None]
    if missing_files:
        confirmed_issues.append(
            f"Expected files not found under search root: {missing_files}"
        )

    for fname, prof in source_profiles.items():
        if prof.read_error:
            confirmed_issues.append(f"{fname}: read error - {prof.read_error}")
            continue
        if prof.missing_expected_columns:
            needs_review.append(
                f"{fname}: expected columns not found in header: {prof.missing_expected_columns}"
            )
        if prof.entity_id_empty:
            confirmed_issues.append(f"{fname}: {prof.entity_id_empty} row(s) with empty entity_id")
        if prof.entity_id_duplicate_value_count:
            confirmed_issues.append(
                f"{fname}: {prof.entity_id_duplicate_value_count} duplicate entity_id value(s) "
                f"across {prof.entity_id_duplicate_row_count} extra row(s)"
            )
        if prof.entity_id_prefix_violations:
            confirmed_issues.append(
                f"{fname}: {prof.entity_id_prefix_violations} entity_id value(s) do not start "
                f"with expected prefix '{EXPECTED_PREFIX.get(prof.source_num)}'"
            )
        observed_statistics.append(
            f"{fname}: {prof.row_count} data rows, {prof.entity_id_unique_count} unique entity_id, "
            f"{prof.country_distinct_count} distinct country values"
        )
        if prof.business_name_non_ascii_count:
            observed_statistics.append(
                f"{fname}: {prof.business_name_non_ascii_count} business_name value(s) contain "
                f"non-ASCII characters (expected due to transliteration; not treated as an error)"
            )
        if prof.duplicate_business_name_value_count or prof.duplicate_business_address_value_count:
            needs_review.append(
                f"{fname}: {prof.duplicate_business_name_value_count} exact-duplicate business_name "
                f"value(s) and {prof.duplicate_business_address_value_count} exact-duplicate "
                f"business_address value(s) observed (raw string duplicates only -- NOT entity "
                f"matching; may or may not indicate the same real-world business)"
            )

    if gt_profile:
        if gt_profile.read_error:
            confirmed_issues.append(f"{gt_profile.filename}: read error - {gt_profile.read_error}")
        else:
            if gt_profile.column_detection_note:
                needs_review.append(f"{gt_profile.filename}: {gt_profile.column_detection_note}")
            if gt_profile.source1_id_empty:
                confirmed_issues.append(
                    f"{gt_profile.filename}: {gt_profile.source1_id_empty} row(s) with empty source1 id"
                )
            if gt_profile.source1_id_duplicate_value_count:
                confirmed_issues.append(
                    f"{gt_profile.filename}: {gt_profile.source1_id_duplicate_value_count} duplicate "
                    f"source1 id value(s) across {gt_profile.source1_id_duplicate_row_count} extra row(s)"
                )
            if gt_profile.unknown_source1_id_count:
                confirmed_issues.append(
                    f"{gt_profile.filename}: {gt_profile.unknown_source1_id_count} source1 id value(s) "
                    f"not found in train_source1.tsv"
                )
            if gt_profile.unknown_matched_id_count:
                confirmed_issues.append(
                    f"{gt_profile.filename}: {gt_profile.unknown_matched_id_count} matched id "
                    f"reference(s) not found in train_source2.tsv or train_source3.tsv"
                )
            if gt_profile.s1_id_used_as_matched_count:
                confirmed_issues.append(
                    f"{gt_profile.filename}: {gt_profile.s1_id_used_as_matched_count} instance(s) "
                    f"where a train_source1 id appears as a matched entity id"
                )
            observed_statistics.append(
                f"{gt_profile.filename}: {gt_profile.row_count} data rows, "
                f"{gt_profile.singleton_row_count} row(s) with an empty match list"
            )
    else:
        needs_review.append("train_ground_truth.tsv was not found; ground-truth checks were skipped.")

    # Train vs test structural comparison (only for source pairs that exist)
    train_test_notes = []
    for num in ("1", "2", "3"):
        train_p = source_profiles.get(f"train_source{num}.tsv")
        test_p = source_profiles.get(f"test_source{num}.tsv")
        if train_p and test_p and not train_p.read_error and not test_p.read_error:
            if train_p.header != test_p.header:
                confirmed_issues.append(
                    f"train_source{num}.tsv and test_source{num}.tsv have different headers: "
                    f"{train_p.header} vs {test_p.header}"
                )
            train_countries = getattr(train_p, "_country_set", set())
            test_countries = getattr(test_p, "_country_set", set())
            only_in_test = sorted(test_countries - train_countries)
            if only_in_test:
                needs_review.append(
                    f"source{num}: countries appearing in test but not in train: {only_in_test}"
                )
            france_in_test = "France" in test_countries
            train_test_notes.append(
                f"source{num}: 'France' present in test country values: {france_in_test}"
            )

    observed_statistics.extend(train_test_notes)

    files_section = {}
    for fname, prof in source_profiles.items():
        d = prof.__dict__.copy()
        d.pop("entity_id_set", None)
        d.pop("_country_set", None)
        files_section[fname] = d

    gt_section = gt_profile.__dict__.copy() if gt_profile else None

    json_report = {
        "audit_metadata": {
            "generated_at_utc": timestamp,
            "search_root": str(root.resolve()),
            "script": "tools/profile_dataset.py",
            "note": (
                "This report reflects only what this script read from disk "
                "on the machine where it was executed. No data was modified. "
                "Exact-string duplicate counts are NOT entity-matching results."
            ),
        },
        "expected_files_found": {k: (str(v) if v else None) for k, v in expected_files.items()},
        "source_files": files_section,
        "ground_truth_file": gt_section,
        "findings": {
            "confirmed_issues": confirmed_issues,
            "observed_statistics": observed_statistics,
            "items_requiring_review": needs_review,
        },
    }

    md = []
    md.append("# Dataset Profile Report")
    md.append("")
    md.append(f"- Generated (UTC): {timestamp}")
    md.append(f"- Search root: `{root.resolve()}`")
    md.append(
        "- This report reflects only what was read from disk on the machine "
        "where the script was run. No dataset file was modified, renamed, "
        "moved, or deleted. No ML, matching, or cleaning was performed."
    )
    md.append("")

    md.append("## 1. Confirmed issues (integrity violations actually observed)")
    if confirmed_issues:
        for item in confirmed_issues:
            md.append(f"- {item}")
    else:
        md.append("- None observed by this script.")
    md.append("")

    md.append("## 2. Observed statistics (neutral facts, not judgments)")
    if observed_statistics:
        for item in observed_statistics:
            md.append(f"- {item}")
    else:
        md.append("- None recorded.")
    md.append("")

    md.append("## 3. Items requiring human/technical review")
    if needs_review:
        for item in needs_review:
            md.append(f"- {item}")
    else:
        md.append("- None flagged.")
    md.append("")

    md.append("## Per-file detail")
    for fname, prof in source_profiles.items():
        md.append(f"### `{fname}`")
        if prof.read_error:
            md.append(f"- **READ ERROR**: {prof.read_error}")
            md.append("")
            continue
        md.append(f"- Path: `{prof.path}`")
        md.append(f"- Header: `{prof.header}`")
        md.append(f"- Data rows: {prof.row_count}")
        md.append(f"- Blank rows: {prof.blank_row_count}")
        md.append(f"- Missing expected columns: {prof.missing_expected_columns or 'none'}")
        md.append(
            f"- entity_id: total={prof.entity_id_total}, empty={prof.entity_id_empty}, "
            f"unique={prof.entity_id_unique_count}, duplicate_values={prof.entity_id_duplicate_value_count}, "
            f"duplicate_extra_rows={prof.entity_id_duplicate_row_count}, "
            f"prefix_violations={prof.entity_id_prefix_violations}"
        )
        if prof.entity_id_duplicate_examples:
            md.append(f"  - Duplicate entity_id examples: {prof.entity_id_duplicate_examples}")
        if prof.entity_id_prefix_violation_examples:
            md.append(f"  - Prefix-violation examples: {prof.entity_id_prefix_violation_examples}")
        md.append(f"- business_name length stats: {prof.business_name_length}")
        md.append(f"- business_address length stats: {prof.business_address_length}")
        md.append(
            f"- business_name: leading/trailing-whitespace={prof.business_name_leading_trailing_ws}, "
            f"non_ascii_count={prof.business_name_non_ascii_count}"
        )
        md.append(
            f"- business_address: leading/trailing-whitespace={prof.business_address_leading_trailing_ws}, "
            f"non_ascii_count={prof.business_address_non_ascii_count}"
        )
        md.append(
            f"- Duplicate business_name values: {prof.duplicate_business_name_value_count} "
            f"(extra rows: {prof.duplicate_business_name_row_count}) -- raw string duplicates only"
        )
        md.append(
            f"- Duplicate business_address values: {prof.duplicate_business_address_value_count} "
            f"(extra rows: {prof.duplicate_business_address_row_count}) -- raw string duplicates only"
        )
        md.append(
            f"- Duplicate (name, address) pairs: {prof.duplicate_name_address_value_count} "
            f"(extra rows: {prof.duplicate_name_address_row_count}) -- raw string duplicates only"
        )
        md.append(f"- Distinct country values: {prof.country_distinct_count}")
        md.append(f"- Country missing/empty count: {prof.country_missing_or_empty}")
        if prof.country_top_values:
            md.append(f"- Top country values (up to {args.top_n_values}): {prof.country_top_values}")
        md.append(f"- Per-column empty counts: {prof.per_column_empty}")
        md.append(f"- Per-column whitespace-only counts: {prof.per_column_whitespace_only}")
        md.append("")

    if gt_profile:
        md.append(f"### `{gt_profile.filename}`")
        if gt_profile.read_error:
            md.append(f"- **READ ERROR**: {gt_profile.read_error}")
        else:
            md.append(f"- Path: `{gt_profile.path}`")
            md.append(f"- Header: `{gt_profile.header}`")
            md.append(f"- id column used: {gt_profile.id_column_used}")
            md.append(f"- match column used: {gt_profile.match_column_used}")
            if gt_profile.column_detection_note:
                md.append(f"- Column detection note: {gt_profile.column_detection_note}")
            md.append(f"- Data rows: {gt_profile.row_count}")
            md.append(f"- Blank rows: {gt_profile.blank_row_count}")
            md.append(
                f"- source1 id: total={gt_profile.source1_id_total}, empty={gt_profile.source1_id_empty}, "
                f"duplicate_values={gt_profile.source1_id_duplicate_value_count}, "
                f"duplicate_extra_rows={gt_profile.source1_id_duplicate_row_count}"
            )
            if gt_profile.source1_id_duplicate_examples:
                md.append(f"  - Duplicate source1 id examples: {gt_profile.source1_id_duplicate_examples}")
            md.append(f"- Singleton rows (empty match list): {gt_profile.singleton_row_count}")
            md.append(f"- Match-count distribution (num_matches -> num_rows): {gt_profile.match_count_distribution}")
            md.append(
                f"- Referential integrity: unknown_source1_id={gt_profile.unknown_source1_id_count}, "
                f"unknown_matched_id={gt_profile.unknown_matched_id_count}, "
                f"s1_id_used_as_matched={gt_profile.s1_id_used_as_matched_count}"
            )
            if gt_profile.unknown_source1_id_examples:
                md.append(f"  - Unknown source1 id examples: {gt_profile.unknown_source1_id_examples}")
            if gt_profile.unknown_matched_id_examples:
                md.append(f"  - Unknown matched id examples: {gt_profile.unknown_matched_id_examples}")
            if gt_profile.s1_id_used_as_matched_examples:
                md.append(f"  - S1-id-as-matched examples: {gt_profile.s1_id_used_as_matched_examples}")
        md.append("")

    md.append("## Explicit non-actions taken")
    md.append("- No file was modified, renamed, moved, or deleted.")
    md.append("- No ML, feature engineering, blocking, matching, training, or prediction was performed.")
    md.append("- Exact-string duplicate counts are reported as raw counts only and are NOT interpreted as entity matches.")
    md.append("- No internet or external data/API lookup was performed.")
    md.append("- No column was assumed present; missing expected columns are reported, not worked around silently.")
    md.append("")

    return json_report, "\n".join(md)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only streaming profiler for the Amazon ML Challenge "
                     "2026 Business Entity Resolution dataset."
    )
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--encoding", type=str, default="utf-8")
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--top-n-values", type=int, default=25)
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    if args.root:
        root = Path(args.root).resolve()
    elif script_path.parent.name.lower() == "tools":
        root = script_path.parent.parent
    else:
        root = script_path.parent
        print(
            f"[WARN] Script is not inside a 'tools' folder; using '{root}' "
            f"as the search root. Pass --root to override.",
            file=sys.stderr,
        )

    if not root.exists():
        print(f"[ERROR] Search root does not exist: {root}", file=sys.stderr)
        return 2

    print(f"[INFO] Searching for dataset files under: {root}")
    expected_files = find_expected_files(root)

    for name, p in expected_files.items():
        if p is None:
            print(f"[WARN] Expected file not found: {name}", file=sys.stderr)
        else:
            print(f"[INFO] Found {name} -> {p}")

    source_profiles = {}
    for fname, (split, num) in SOURCE_FILES.items():
        p = expected_files.get(fname)
        if p is None:
            continue
        print(f"[INFO] Profiling {fname} ...")
        prof = profile_source_file(p, split, num, args.encoding, args.max_examples, args.top_n_values)
        source_profiles[fname] = prof
        if prof.read_error:
            print(f"[ERROR] {fname}: {prof.read_error}", file=sys.stderr)
        else:
            print(
                f"[INFO]   -> rows={prof.row_count} unique_entity_id={prof.entity_id_unique_count} "
                f"dup_entity_id_values={prof.entity_id_duplicate_value_count} "
                f"prefix_violations={prof.entity_id_prefix_violations}"
            )

    s1_ids = getattr(source_profiles.get("train_source1.tsv"), "entity_id_set", set()) or set()
    s2_ids = getattr(source_profiles.get("train_source2.tsv"), "entity_id_set", set()) or set()
    s3_ids = getattr(source_profiles.get("train_source3.tsv"), "entity_id_set", set()) or set()

    gt_profile = None
    gt_path = expected_files.get(GROUND_TRUTH_FILE)
    if gt_path is not None:
        print(f"[INFO] Profiling {GROUND_TRUTH_FILE} ...")
        gt_profile = profile_ground_truth(gt_path, args.encoding, args.max_examples, s1_ids, s2_ids, s3_ids)
        if gt_profile.read_error:
            print(f"[ERROR] {GROUND_TRUTH_FILE}: {gt_profile.read_error}", file=sys.stderr)
        else:
            print(
                f"[INFO]   -> rows={gt_profile.row_count} "
                f"unknown_source1_id={gt_profile.unknown_source1_id_count} "
                f"unknown_matched_id={gt_profile.unknown_matched_id_count}"
            )

    json_report, md_report = build_report(root, expected_files, source_profiles, gt_profile, args)

    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "dataset_profile.json"
    md_path = reports_dir / "dataset_profile.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)

    print(f"[INFO] Wrote: {json_path}")
    print(f"[INFO] Wrote: {md_path}")
    print(
        "[INFO] Profiling complete. No files were modified. No ML, matching, "
        "or cleaning was performed. Review the reports before any further step."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())