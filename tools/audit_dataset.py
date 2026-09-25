#!/usr/bin/env python3
"""
audit_dataset.py

Read-only, streaming audit tool for the Amazon ML Challenge 2026
"Business Entity Resolution" dataset.

WHAT THIS SCRIPT DOES
----------------------
- Recursively searches the project for TSV files matching the expected
  dataset filenames (does not assume a fixed path).
- Streams each file line-by-line (never loads the whole file into memory).
- Reports, per file: size, header, column count, data-row count,
  blank-row count, duplicate-header count, malformed-row count, and any
  encoding/read errors.
- Detects rows whose tab-separated field count differs from the header's
  field count (missing tabs / extra tabs), and records their exact
  physical line numbers with a truncated, safe preview.
- Treats commas as ordinary data characters (this dataset is TSV, not CSV) 
  and only splits on the tab character.
- Verifies whether the expected source columns
  (entity_id, business_name, business_address, country) are present in
  each source-like file, and reports ground-truth columns separately
  since its schema is expected to differ.
- Writes reports/dataset_audit.json and reports/dataset_audit.md.

WHAT THIS SCRIPT DOES NOT DO
-----------------------------
- It NEVER writes to, renames, moves, or deletes any original dataset file.
- It performs NO cleaning, normalization, imputation, blocking, matching,
  feature engineering, or modelling of any kind.
- It makes NO internet or external API calls.
- It does NOT guess a delimiter if tabs are absent from a row; a row with
  the wrong field count is flagged as malformed, never "fixed".

USAGE (Windows PowerShell)
---------------------------
    cd D:\\Documents\\PARTH\\AmazonML\\AmazonMLchallenge
    python tools\\audit_dataset.py

Optional flags:
    --root <path>            Directory to search from (default: project root,
                              i.e. the parent of the "tools" folder this
                              script lives in).
    --max-preview-chars N     Max chars kept in a malformed-row preview
                              (default: 200).
    --max-malformed-examples N
                              Max malformed-row examples stored per file in
                              the JSON/MD report (default: 200). All
                              malformed rows are still counted; this only
                              caps how many example lines are recorded.
    --encoding NAME           Encoding to try first (default: utf-8). The
                              script always reports if a file could not be
                              read cleanly with this encoding.

This script makes NO assumption that it has already been run. Nothing in
this file causes any claim of "the dataset is clean" unless the actual
counts in the generated report prove it — read reports/dataset_audit.md
after running.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Field count is bumped up only for csv.reader's internal limits, never used
# to reinterpret delimiters.
csv.field_size_limit(min(sys.maxsize, 2_147_483_647))

# ---------------------------------------------------------------------------
# Expected dataset shape (per the official problem statement). These are
# used only to *identify* candidate files by name and to *report* whether
# expected columns are present -- never to assume a row is well-formed.
# ---------------------------------------------------------------------------

EXPECTED_SOURCE_FILENAMES = [
    "train_source1.tsv",
    "train_source2.tsv",
    "train_source3.tsv",
    "test_source1.tsv",
    "test_source2.tsv",
    "test_source3.tsv",
]

EXPECTED_GROUND_TRUTH_FILENAMES = [
    "train_ground_truth.tsv",
]

EXPECTED_SOURCE_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

# Directories we never want to walk into (huge / irrelevant / vendored).
SKIP_DIR_NAMES = {
    ".git", ".hg", ".svn", "__pycache__", "node_modules",
    ".venv", "venv", "env", ".mypy_cache", ".pytest_cache",
    ".idea", ".vscode",
}


@dataclass
class MalformedRow:
    line_number: int
    expected_fields: int
    actual_fields: int
    preview: str


@dataclass
class FileAudit:
    path: str
    role: str  # "source", "ground_truth", or "unclassified"
    size_bytes: Optional[int] = None
    encoding_used: Optional[str] = None
    encoding_error: Optional[str] = None
    header: Optional[list] = None
    header_field_count: Optional[int] = None
    data_row_count: int = 0
    blank_row_count: int = 0
    duplicate_header_count: int = 0
    malformed_row_count: int = 0
    malformed_row_examples: list = field(default_factory=list)
    missing_expected_columns: list = field(default_factory=list)
    unexpected_extra_columns: list = field(default_factory=list)
    non_tab_delimiter_suspected: bool = False
    read_error: Optional[str] = None
    other_anomalies: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "role": self.role,
            "size_bytes": self.size_bytes,
            "encoding_used": self.encoding_used,
            "encoding_error": self.encoding_error,
            "header": self.header,
            "header_field_count": self.header_field_count,
            "data_row_count": self.data_row_count,
            "blank_row_count": self.blank_row_count,
            "duplicate_header_count": self.duplicate_header_count,
            "malformed_row_count": self.malformed_row_count,
            "malformed_row_examples": [
                {
                    "line_number": m.line_number,
                    "expected_fields": m.expected_fields,
                    "actual_fields": m.actual_fields,
                    "preview": m.preview,
                }
                for m in self.malformed_row_examples
            ],
            "missing_expected_columns": self.missing_expected_columns,
            "unexpected_extra_columns": self.unexpected_extra_columns,
            "non_tab_delimiter_suspected": self.non_tab_delimiter_suspected,
            "read_error": self.read_error,
            "other_anomalies": self.other_anomalies,
        }


def safe_preview(raw_line: str, max_chars: int) -> str:
    """Truncate a raw line for safe display; collapse tabs visibly so the
    preview doesn't look like well-formed columns, and never print more
    than max_chars characters."""
    cleaned = raw_line.replace("\t", "<TAB>")
    if len(cleaned) > max_chars:
        return cleaned[:max_chars] + f"...<truncated, {len(raw_line)} chars total>"
    return cleaned


def classify_role(filename: str) -> str:
    if filename in EXPECTED_SOURCE_FILENAMES:
        return "source"
    if filename in EXPECTED_GROUND_TRUTH_FILENAMES:
        return "ground_truth"
    return "unclassified"


def find_candidate_files(root: Path) -> list:
    """Recursively find files whose name matches one of the expected
    dataset filenames. Does not assume any directory structure."""
    expected_names = set(EXPECTED_SOURCE_FILENAMES) | set(EXPECTED_GROUND_TRUTH_FILENAMES)
    found = []
    for dirpath, dirnames, filenames in _walk_skipping(root):
        for name in filenames:
            if name in expected_names:
                found.append(Path(dirpath) / name)
    return sorted(found)


def find_all_tsv_files(root: Path) -> list:
    """Recursively find every .tsv file, so we can report any that don't
    match an expected filename (an anomaly worth flagging, not assuming)."""
    found = []
    for dirpath, dirnames, filenames in _walk_skipping(root):
        for name in filenames:
            if name.lower().endswith(".tsv"):
                found.append(Path(dirpath) / name)
    return sorted(found)


def _walk_skipping(root: Path):
    import os
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        yield dirpath, dirnames, filenames


def audit_file(path: Path, role: str, encoding: str, max_preview_chars: int,
               max_malformed_examples: int) -> FileAudit:
    audit = FileAudit(path=str(path), role=role)

    try:
        audit.size_bytes = path.stat().st_size
    except OSError as e:
        audit.read_error = f"Could not stat file: {e}"
        return audit

    header = None
    header_field_count = None
    seen_header_line = None
    line_number = 0

    try:
        with open(path, "r", encoding=encoding, newline="") as f:
            audit.encoding_used = encoding
            for raw_line in f:
                line_number += 1
                # Strip only the line terminator, never other whitespace,
                # so we don't mask malformed rows.
                line = raw_line.rstrip("\r\n")

                if line == "":
                    audit.blank_row_count += 1
                    continue

                fields = line.split("\t")

                if header is None:
                    header = fields
                    header_field_count = len(fields)
                    seen_header_line = line
                    audit.header = header
                    audit.header_field_count = header_field_count

                    if header_field_count == 1:
                        # Only one field on the header line at all -> no tabs
                        # found where a multi-column header was expected.
                        audit.non_tab_delimiter_suspected = True
                        audit.other_anomalies.append(
                            "Header line contains no tab characters; "
                            "verify this file is actually tab-separated."
                        )

                    missing = [c for c in EXPECTED_SOURCE_COLUMNS if c not in header]
                    if role == "source":
                        audit.missing_expected_columns = missing
                        audit.unexpected_extra_columns = [
                            c for c in header if c not in EXPECTED_SOURCE_COLUMNS
                        ]
                    elif role == "ground_truth":
                        # Ground truth schema is expected to differ from the
                        # source files; report its actual header as-is
                        # without comparing to EXPECTED_SOURCE_COLUMNS.
                        audit.other_anomalies.append(
                            "Ground-truth file: header reported as-is; "
                            "not compared against source-file column list "
                            "because its structure is expected to differ."
                        )
                    else:
                        audit.other_anomalies.append(
                            "File matched no expected role classification; "
                            "header reported as-is."
                        )
                    continue

                # Duplicate header detection: an identical line to the
                # header reappearing later in the file (possible
                # concatenated-file artifact).
                if line == seen_header_line:
                    audit.duplicate_header_count += 1
                    continue

                audit.data_row_count += 1

                if len(fields) != header_field_count:
                    audit.malformed_row_count += 1
                    if len(audit.malformed_row_examples) < max_malformed_examples:
                        audit.malformed_row_examples.append(
                            MalformedRow(
                                line_number=line_number,
                                expected_fields=header_field_count,
                                actual_fields=len(fields),
                                preview=safe_preview(line, max_preview_chars),
                            )
                        )

    except UnicodeDecodeError as e:
        audit.encoding_error = (
            f"Failed to decode with encoding='{encoding}' at approx byte "
            f"offset reported by decoder: {e}. Another encoding may be "
            f"required; re-run with --encoding to try a different one. "
            f"No assumption about the correct encoding has been made."
        )
        audit.read_error = "Encoding error stopped a full read of this file."
    except OSError as e:
        audit.read_error = f"OS-level error while reading file: {e}"

    return audit


def build_reports(root: Path, audits: list, all_tsv_paths: list,
                   matched_paths: set, args: argparse.Namespace) -> tuple:
    timestamp = datetime.now(timezone.utc).isoformat()

    unmatched_tsv = [str(p) for p in all_tsv_paths if p not in matched_paths]

    missing_expected_files = []
    found_names = {Path(a.path).name for a in audits}
    for name in EXPECTED_SOURCE_FILENAMES + EXPECTED_GROUND_TRUTH_FILENAMES:
        if name not in found_names:
            missing_expected_files.append(name)

    json_report = {
        "audit_metadata": {
            "generated_at_utc": timestamp,
            "search_root": str(root.resolve()),
            "script": "tools/audit_dataset.py",
            "note": (
                "This report reflects only what this script read from disk "
                "on the machine where it was executed. No data was modified."
            ),
        },
        "expected_files_not_found": missing_expected_files,
        "tsv_files_found_but_not_expected_by_name": unmatched_tsv,
        "files": [a.to_dict() for a in audits],
    }

    md_lines = []
    md_lines.append("# Dataset Audit Report")
    md_lines.append("")
    md_lines.append(f"- Generated (UTC): {timestamp}")
    md_lines.append(f"- Search root: `{root.resolve()}`")
    md_lines.append(
        "- This report reflects only what was read from disk on the "
        "machine where the script was run. No dataset file was modified, "
        "renamed, moved, or deleted."
    )
    md_lines.append("")

    md_lines.append("## Expected files not found")
    if missing_expected_files:
        for name in missing_expected_files:
            md_lines.append(f"- `{name}` — NOT FOUND under search root")
    else:
        md_lines.append("- None. All expected filenames were located.")
    md_lines.append("")

    md_lines.append("## TSV files found that do not match an expected filename")
    if unmatched_tsv:
        for p in unmatched_tsv:
            md_lines.append(f"- `{p}`")
    else:
        md_lines.append("- None.")
    md_lines.append("")

    md_lines.append("## Per-file audit")
    for a in audits:
        md_lines.append(f"### `{a.path}`")
        md_lines.append(f"- Role (by filename): **{a.role}**")
        md_lines.append(f"- Size: {a.size_bytes} bytes" if a.size_bytes is not None else "- Size: UNKNOWN (stat failed)")
        if a.read_error:
            md_lines.append(f"- **READ ERROR**: {a.read_error}")
        if a.encoding_error:
            md_lines.append(f"- **ENCODING ERROR**: {a.encoding_error}")
        md_lines.append(f"- Encoding used: {a.encoding_used}")
        md_lines.append(f"- Header field count: {a.header_field_count}")
        md_lines.append(f"- Header: `{a.header}`")
        md_lines.append(f"- Data row count (excludes header/blank/duplicate-header lines): {a.data_row_count}")
        md_lines.append(f"- Blank row count: {a.blank_row_count}")
        md_lines.append(f"- Duplicate header row count: {a.duplicate_header_count}")
        md_lines.append(f"- Malformed row count (field count != header field count): {a.malformed_row_count}")
        if a.non_tab_delimiter_suspected:
            md_lines.append("- **WARNING**: header line contained no tab characters.")
        if a.role == "source":
            md_lines.append(f"- Missing expected columns: {a.missing_expected_columns or 'none'}")
            md_lines.append(f"- Extra/unexpected columns vs. expected list: {a.unexpected_extra_columns or 'none'}")
        if a.other_anomalies:
            md_lines.append("- Other notes:")
            for note in a.other_anomalies:
                md_lines.append(f"  - {note}")
        if a.malformed_row_examples:
            cap_note = ""
            if a.malformed_row_count > len(a.malformed_row_examples):
                cap_note = (
                    f" (showing first {len(a.malformed_row_examples)} of "
                    f"{a.malformed_row_count} total malformed rows)"
                )
            md_lines.append(f"- Malformed row examples{cap_note}:")
            md_lines.append("")
            md_lines.append("  | Line # | Expected fields | Actual fields | Preview |")
            md_lines.append("  |---|---|---|---|")
            for m in a.malformed_row_examples:
                preview_escaped = m.preview.replace("|", "\\|")
                md_lines.append(
                    f"  | {m.line_number} | {m.expected_fields} | {m.actual_fields} | `{preview_escaped}` |"
                )
        md_lines.append("")

    md_lines.append("## Explicit non-actions taken")
    md_lines.append("- No file was modified, renamed, moved, or deleted.")
    md_lines.append("- No malformed row was repaired, dropped, or reinterpreted.")
    md_lines.append("- No missing value was inferred or filled in.")
    md_lines.append("- No delimiter was substituted for tabs, even when tabs were missing.")
    md_lines.append("- No ML, cleaning, feature engineering, blocking, or matching was performed.")
    md_lines.append("- No internet or external data lookup was performed.")
    md_lines.append("")

    return json_report, "\n".join(md_lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only streaming auditor for the Amazon ML Challenge "
                     "2026 Business Entity Resolution dataset."
    )
    parser.add_argument(
        "--root", type=str, default=None,
        help="Directory to search from. Default: parent directory of the "
             "'tools' folder containing this script (i.e. the project root)."
    )
    parser.add_argument("--max-preview-chars", type=int, default=200)
    parser.add_argument("--max-malformed-examples", type=int, default=200)
    parser.add_argument("--encoding", type=str, default="utf-8")
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    if args.root:
        root = Path(args.root).resolve()
    else:
        # Assume this script lives at <project_root>/tools/audit_dataset.py.
        # If it doesn't, fall back to the script's own directory and say so.
        if script_path.parent.name.lower() == "tools":
            root = script_path.parent.parent
        else:
            root = script_path.parent
            print(
                f"[WARN] Script is not inside a 'tools' folder; using "
                f"'{root}' as the search root. Pass --root to override.",
                file=sys.stderr,
            )

    if not root.exists():
        print(f"[ERROR] Search root does not exist: {root}", file=sys.stderr)
        return 2

    print(f"[INFO] Searching for dataset files under: {root}")

    candidate_paths = find_candidate_files(root)
    all_tsv_paths = find_all_tsv_files(root)
    matched_paths = set(candidate_paths)

    if not candidate_paths:
        print(
            "[WARN] No files matching the expected dataset filenames were "
            "found under the search root. Nothing will be audited. Check "
            "that you are running this from the correct project root, or "
            "pass --root explicitly.",
            file=sys.stderr,
        )

    audits = []
    for p in candidate_paths:
        role = classify_role(p.name)
        print(f"[INFO] Auditing ({role}): {p}")
        audit = audit_file(
            p, role, args.encoding, args.max_preview_chars,
            args.max_malformed_examples,
        )
        audits.append(audit)
        if audit.read_error:
            print(f"[ERROR] {p}: {audit.read_error}", file=sys.stderr)
        else:
            print(
                f"[INFO]   -> rows={audit.data_row_count} "
                f"malformed={audit.malformed_row_count} "
                f"blank={audit.blank_row_count} "
                f"dup_header={audit.duplicate_header_count}"
            )

    json_report, md_report = build_reports(root, audits, all_tsv_paths, matched_paths, args)

    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    json_path = reports_dir / "dataset_audit.json"
    md_path = reports_dir / "dataset_audit.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)

    print(f"[INFO] Wrote: {json_path}")
    print(f"[INFO] Wrote: {md_path}")
    print(
        "[INFO] Audit complete. This script performed no cleaning, "
        "modelling, or file modification of any kind. Review the reports "
        "above before making any preprocessing decisions."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())