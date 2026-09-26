#!/usr/bin/env python3
"""
evaluate_f05.py

Macro F0.5 evaluator for Amazon ML Challenge Entity Resolution.
Evaluates precision, recall, macro F0.5, and singleton accuracy across Source 1 entities.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union


def read_match_tsv(path: Union[str, Path], encoding: str = "utf-8") -> Dict[str, Set[str]]:
    """
    Reads a TSV mapping Source-1 entity IDs to matched/candidate entity IDs.

    Header handling semantics:
      1. Pass 1: Explicit matching for Source-1 and Matched/Candidate columns.
      2. Pass 2: Fallback matching for generic "entity_id" or "s1" columns only if no explicit
         Source-1 column was identified in Pass 1.
      3. CRITICAL: "matched_entity_ids" and "candidate_entity_ids" MUST NEVER trigger the
         generic "entity_id" fallback.

    Returns: Dict[s1_id, Set[matched_or_candidate_ids]]
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Match TSV file not found: {path.resolve()}")

    results: Dict[str, Set[str]] = {}

    with open(path, "r", encoding=encoding, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        try:
            raw_header = next(reader)
        except StopIteration:
            return results

        header = [c.strip().lstrip("\ufeff").casefold() for c in raw_header]

        s1_col = None
        match_col = None

        # Pass 1: Explicit / Exact header matching
        for idx, col_clean in enumerate(header):
            if col_clean in ("source1_entity_id", "source1 id", "source1", "source1_id", "s1_id", "s1"):
                s1_col = idx
            elif (
                col_clean in ("matched_entity_ids", "candidate_entity_ids", "matched_ids", "candidate_ids")
                or "matched" in col_clean
                or "candidate" in col_clean
            ):
                match_col = idx

        # Pass 2: Fallback for generic "entity_id" if explicit Source-1 column was not found
        # Explicitly ignore any column containing "matched" or "candidate"
        if s1_col is None:
            for idx, col_clean in enumerate(header):
                if idx != match_col and "matched" not in col_clean and "candidate" not in col_clean:
                    if col_clean == "entity_id" or col_clean.startswith("source1") or col_clean.startswith("s1"):
                        s1_col = idx
                        break

        # Pass 3: Default indices if headers remain unassigned
        if s1_col is None:
            s1_col = 0
        if match_col is None:
            match_col = 1 if len(header) > 1 and s1_col != 1 else 0

        for fields in reader:
            if not fields or (len(fields) == 1 and fields[0] == ""):
                continue

            s1_id = fields[s1_col].strip() if len(fields) > s1_col else ""
            if not s1_id:
                continue

            matches_raw = fields[match_col].strip() if len(fields) > match_col else ""
            if matches_raw:
                match_set = {m.strip() for m in matches_raw.split(",") if m.strip()}
            else:
                match_set = set()

            results[s1_id] = match_set

    return results


def compute_per_s1_f05(
    gt_matches: Dict[str, Set[str]],
    pred_matches: Dict[str, Set[str]],
    val_s1_ids: List[str],
) -> Dict[str, float]:
    """
    Computes per-S1 Macro F0.5 score across all validation Source-1 entities.
    """
    sum_f05 = 0.0
    sum_prec = 0.0
    sum_rec = 0.0
    singleton_correct = 0
    singleton_total = 0

    total_s1 = len(val_s1_ids)
    if total_s1 == 0:
        return {
            "macro_f05": 0.0,
            "mean_precision": 0.0,
            "mean_recall": 0.0,
            "singleton_accuracy": 0.0,
        }

    for s1_id in val_s1_ids:
        gt_set = gt_matches.get(s1_id, set())
        pred_set = pred_matches.get(s1_id, set())

        if len(gt_set) == 0:
            # Ground truth singleton
            singleton_total += 1
            if len(pred_set) == 0:
                singleton_correct += 1
                sum_prec += 1.0
                sum_rec += 1.0
                sum_f05 += 1.0
            else:
                sum_prec += 0.0
                sum_rec += 1.0
                sum_f05 += 0.0
        else:
            tp = len(pred_set & gt_set)
            fp = len(pred_set - gt_set)
            fn = len(gt_set - pred_set)

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            denom = 0.25 * prec + rec
            f05 = (1.25 * prec * rec / denom) if denom > 0 else 0.0

            sum_prec += prec
            sum_rec += rec
            sum_f05 += f05

    macro_f05 = sum_f05 / total_s1
    mean_prec = sum_prec / total_s1
    mean_rec = sum_rec / total_s1
    singleton_acc = (singleton_correct / singleton_total) if singleton_total > 0 else 1.0

    return {
        "macro_f05": macro_f05,
        "mean_precision": mean_prec,
        "mean_recall": mean_rec,
        "singleton_accuracy": singleton_acc,
    }


def run_self_tests() -> None:
    """Executes all 8 original metric self-tests and the header parsing regression test."""
    print("Running evaluate_f05.py self-tests...")

    # --- Test 1: Perfect match non-singleton ---
    gt = {"S1_1": {"S2_A", "S3_B"}}
    pred = {"S1_1": {"S2_A", "S3_B"}}
    res = compute_per_s1_f05(gt, pred, ["S1_1"])
    assert math.isclose(res["macro_f05"], 1.0), f"Test 1 failed: {res}"
    assert math.isclose(res["mean_precision"], 1.0), f"Test 1 failed: {res}"
    assert math.isclose(res["mean_recall"], 1.0), f"Test 1 failed: {res}"

    # --- Test 2: Perfect match singleton ---
    gt = {"S1_1": set()}
    pred = {"S1_1": set()}
    res = compute_per_s1_f05(gt, pred, ["S1_1"])
    assert math.isclose(res["macro_f05"], 1.0), f"Test 2 failed: {res}"
    assert math.isclose(res["singleton_accuracy"], 1.0), f"Test 2 failed: {res}"

    # --- Test 3: False positive on singleton ---
    gt = {"S1_1": set()}
    pred = {"S1_1": {"S2_X"}}
    res = compute_per_s1_f05(gt, pred, ["S1_1"])
    assert math.isclose(res["macro_f05"], 0.0), f"Test 3 failed: {res}"
    assert math.isclose(res["singleton_accuracy"], 0.0), f"Test 3 failed: {res}"

    # --- Test 4: Partial match ---
    gt = {"S1_1": {"S2_A", "S2_B"}}
    pred = {"S1_1": {"S2_A", "S2_C"}}  # TP=1, FP=1, FN=1 -> prec=0.5, rec=0.5
    res = compute_per_s1_f05(gt, pred, ["S1_1"])
    # prec=0.5, rec=0.5 -> F0.5 = 1.25 * 0.25 / (0.125 + 0.5) = 0.3125 / 0.625 = 0.5
    assert math.isclose(res["macro_f05"], 0.5), f"Test 4 failed: {res}"

    # --- Test 5: Zero recall ---
    gt = {"S1_1": {"S2_A"}}
    pred = {"S1_1": set()}
    res = compute_per_s1_f05(gt, pred, ["S1_1"])
    assert math.isclose(res["macro_f05"], 0.0), f"Test 5 failed: {res}"

    # --- Test 6: Mixture of singletons and non-singletons ---
    gt = {"S1_1": {"S2_A"}, "S1_2": set()}
    pred = {"S1_1": {"S2_A"}, "S1_2": set()}
    res = compute_per_s1_f05(gt, pred, ["S1_1", "S1_2"])
    assert math.isclose(res["macro_f05"], 1.0), f"Test 6 failed: {res}"
    assert math.isclose(res["singleton_accuracy"], 1.0), f"Test 6 failed: {res}"

    # --- Test 7: Zero validation entities ---
    res = compute_per_s1_f05({}, {}, [])
    assert res["macro_f05"] == 0.0, f"Test 7 failed: {res}"

    # --- Test 8: F0.5 weighting verification ---
    # GT has 1 match, Pred has 2 matches (1 TP, 1 FP) -> prec = 0.5, rec = 1.0
    # F0.5 = (1.25 * 0.5 * 1.0) / (0.25 * 0.5 + 1.0) = 0.625 / 1.125 = 5/9 ≈ 0.555555...
    gt = {"S1_1": {"S2_A"}}
    pred = {"S1_1": {"S2_A", "S2_B"}}
    res = compute_per_s1_f05(gt, pred, ["S1_1"])
    expected_f05 = 5.0 / 9.0
    assert math.isclose(res["macro_f05"], expected_f05), f"Test 8 failed: {res}"

    # --- Test 9: Header parsing regression test ---
    test_tsv_content = "source1_entity_id\tmatched_entity_ids\nS1-A\tS2-X,S3-Y\nS1-B\t\n"
    with tempfile.NamedTemporaryFile("w+", encoding="utf-8", suffix=".tsv", delete=False) as tmp:
        tmp.write(test_tsv_content)
        tmp_path = Path(tmp.name)

    try:
        parsed = read_match_tsv(tmp_path)
        assert set(parsed.keys()) == {"S1-A", "S1-B"}, f"Test 9 failed keys: {set(parsed.keys())}"
        assert parsed["S1-A"] == {"S2-X", "S3-Y"}, f"Test 9 failed S1-A: {parsed['S1-A']}"
        assert parsed["S1-B"] == set(), f"Test 9 failed S1-B: {parsed['S1-B']}"
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    print("[SUCCESS] All 9 self-tests passed cleanly!")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate F0.5 score for Entity Resolution")
    parser.add_argument("--gt", type=Path, help="Path to ground truth TSV")
    parser.add_argument("--pred", type=Path, help="Path to predictions TSV")
    parser.add_argument("--self-test", action="store_true", help="Run internal self-tests")

    args = parser.parse_args()

    if args.self_test:
        run_self_tests()
        return 0

    if not args.gt or not args.pred:
        parser.print_help()
        return 1

    gt_matches = read_match_tsv(args.gt)
    pred_matches = read_match_tsv(args.pred)

    val_s1_ids = list(gt_matches.keys())
    res = compute_per_s1_f05(gt_matches, pred_matches, val_s1_ids)

    print(f"Macro F0.5:         {res['macro_f05']:.6f}")
    print(f"Mean Precision:     {res['mean_precision']:.6f}")
    print(f"Mean Recall:        {res['mean_recall']:.6f}")
    print(f"Singleton Accuracy: {res['singleton_accuracy']:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())