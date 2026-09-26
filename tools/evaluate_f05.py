#!/usr/bin/env python3
"""
evaluate_f05.py

Official local evaluator for the Amazon ML Challenge Macro F0.5 metric.
Computes exact per-Source-1 entity Precision, Recall, and F0.5 scores
and macro-averages across all validation entities (including singletons).

Pure standard library implementation with zero external dependencies.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set, Union


def compute_per_s1_f05(
    gt_matches: Dict[str, Union[Set[str], List[str]]],
    pred_matches: Dict[str, Union[Set[str], List[str]]],
    s1_ids: Iterable[str],
) -> Dict[str, Union[float, int]]:
    """
    Computes Macro F0.5 per challenge metric specification.

    Args:
        gt_matches: Map of s1_id -> collection of true match IDs
        pred_matches: Map of s1_id -> collection of predicted match IDs
        s1_ids: Iterable of all S1 entity IDs to evaluate

    Returns:
        Dict containing:
          - macro_f05: Macro-averaged F0.5 across all validation S1 entities
          - mean_precision: Mean precision across all validation S1 entities
          - mean_recall: Mean recall across all validation S1 entities
          - singleton_count: Total number of true singleton S1 entities
          - singleton_correct: Number of singletons correctly predicted as empty
          - singleton_accuracy: Accuracy on true singletons (correct / count)
          - non_singleton_count: Total number of non-singleton S1 entities
          - non_singleton_f05: Mean F0.5 across non-singleton S1 entities
    """
    total_f05 = 0.0
    total_precision = 0.0
    total_recall = 0.0

    singleton_count = 0
    singleton_correct = 0
    non_singleton_count = 0
    non_singleton_f05_sum = 0.0

    s1_id_list = list(s1_ids)
    n_val = len(s1_id_list)

    if n_val == 0:
        return {
            "macro_f05": 0.0,
            "mean_precision": 0.0,
            "mean_recall": 0.0,
            "singleton_count": 0,
            "singleton_correct": 0,
            "singleton_accuracy": 0.0,
            "non_singleton_count": 0,
            "non_singleton_f05": 0.0,
        }

    for s1_id in s1_id_list:
        raw_gt = gt_matches.get(s1_id, set())
        raw_pred = pred_matches.get(s1_id, set())

        g_i = set(raw_gt) if raw_gt else set()
        m_i = set(raw_pred) if raw_pred else set()

        if len(g_i) == 0:
            # True Singleton Entity
            singleton_count += 1
            if len(m_i) == 0:
                f05_i = 1.0
                p_i = 1.0
                r_i = 1.0
                singleton_correct += 1
            else:
                f05_i = 0.0
                p_i = 0.0
                r_i = 1.0
        else:
            # Non-Singleton Entity
            non_singleton_count += 1
            if len(m_i) == 0:
                f05_i = 0.0
                p_i = 1.0
                r_i = 0.0
            else:
                tp = len(g_i & m_i)
                p_i = tp / len(m_i)
                r_i = tp / len(g_i)

                denom = (0.25 * p_i) + r_i
                if denom > 0:
                    f05_i = (1.25 * p_i * r_i) / denom
                else:
                    f05_i = 0.0

            non_singleton_f05_sum += f05_i

        total_f05 += f05_i
        total_precision += p_i
        total_recall += r_i

    return {
        "macro_f05": round(total_f05 / n_val, 6),
        "mean_precision": round(total_precision / n_val, 6),
        "mean_recall": round(total_recall / n_val, 6),
        "singleton_count": singleton_count,
        "singleton_correct": singleton_correct,
        "singleton_accuracy": round(singleton_correct / singleton_count, 6) if singleton_count > 0 else 0.0,
        "non_singleton_count": non_singleton_count,
        "non_singleton_f05": round(non_singleton_f05_sum / non_singleton_count, 6) if non_singleton_count > 0 else 0.0,
    }


def read_match_tsv(file_path: Path) -> Dict[str, Set[str]]:
    """
    Reads a TSV prediction or ground-truth file line-by-line.
    Maps s1_id -> Set of matched IDs.
    """
    matches: Dict[str, Set[str]] = {}
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path.resolve()}")

    with open(file_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if not header:
            return matches

        s1_col = 0
        match_col = 1
        for idx, col in enumerate(header):
            col_clean = col.strip().lstrip("\ufeff").casefold()
            if "source1" in col_clean or "entity_id" in col_clean:
                s1_col = idx
            elif "match" in col_clean or "candidate" in col_clean:
                match_col = idx

        for fields in reader:
            if not fields or len(fields) <= s1_col:
                continue
            s1_id = fields[s1_col].strip()
            if not s1_id:
                continue

            match_str = fields[match_col].strip() if len(fields) > match_col else ""
            if match_str:
                m_ids = {m.strip() for m in match_str.split(",") if m.strip()}
            else:
                m_ids = set()

            matches[s1_id] = m_ids

    return matches


def run_self_tests() -> bool:
    """Executes deterministic self-tests on hardcoded toy examples."""
    print("=" * 60)
    print("RUNNING EVALUATOR SELF-TESTS")
    print("=" * 60)

    # 1. Perfect non-singleton
    res1 = compute_per_s1_f05({"S1": {"A", "B"}}, {"S1": {"A", "B"}}, ["S1"])
    assert res1["macro_f05"] == 1.0, f"Test 1 failed: {res1}"
    print("[PASS] Test 1: Perfect non-singleton -> F0.5 = 1.0")

    # 2. Partial recall with perfect precision (GT={A,B}, Pred={A}) -> F0.5 = 5/6 ~ 0.833333
    res2 = compute_per_s1_f05({"S1": {"A", "B"}}, {"S1": {"A"}}, ["S1"])
    expected_f05_2 = round(5.0 / 6.0, 6)
    assert res2["macro_f05"] == expected_f05_2, f"Test 2 failed: {res2} vs expected {expected_f05_2}"
    print(f"[PASS] Test 2: Partial recall w/ perfect precision -> F0.5 = {res2['macro_f05']}")

    # 3. False positives (GT={A,B}, Pred={A,B,C}) -> F0.5 = 5/7 ~ 0.714286
    res3 = compute_per_s1_f05({"S1": {"A", "B"}}, {"S1": {"A", "B", "C"}}, ["S1"])
    expected_f05_3 = round(5.0 / 7.0, 6)
    assert res3["macro_f05"] == expected_f05_3, f"Test 3 failed: {res3} vs expected {expected_f05_3}"
    print(f"[PASS] Test 3: False positives -> F0.5 = {res3['macro_f05']}")

    # 4. True singleton correctly empty
    res4 = compute_per_s1_f05({"S1": set()}, {"S1": set()}, ["S1"])
    assert res4["macro_f05"] == 1.0 and res4["singleton_accuracy"] == 1.0, f"Test 4 failed: {res4}"
    print("[PASS] Test 4: True singleton correctly empty -> F0.5 = 1.0")

    # 5. False merge on singleton
    res5 = compute_per_s1_f05({"S1": set()}, {"S1": {"A"}}, ["S1"])
    assert res5["macro_f05"] == 0.0 and res5["singleton_accuracy"] == 0.0, f"Test 5 failed: {res5}"
    print("[PASS] Test 5: False merge on singleton -> F0.5 = 0.0")

    # 6. Non-singleton with zero prediction
    res6 = compute_per_s1_f05({"S1": {"A"}}, {"S1": set()}, ["S1"])
    assert res6["macro_f05"] == 0.0, f"Test 6 failed: {res6}"
    print("[PASS] Test 6: Non-singleton with zero prediction -> F0.5 = 0.0")

    # 7. Duplicate prediction IDs (GT={A}, Pred={A, A})
    res7 = compute_per_s1_f05({"S1": {"A"}}, {"S1": ["A", "A"]}, ["S1"])
    assert res7["macro_f05"] == 1.0, f"Test 7 failed: {res7}"
    print("[PASS] Test 7: Duplicate prediction IDs -> F0.5 = 1.0")

    # 8. Combined Multi-Entity Test (1, 4, 5 combined)
    gt_multi = {"S1-Pass": {"A", "B"}, "S1-SingPass": set(), "S1-SingFail": set()}
    pred_multi = {"S1-Pass": {"A", "B"}, "S1-SingPass": set(), "S1-SingFail": {"X"}}
    res8 = compute_per_s1_f05(gt_multi, pred_multi, ["S1-Pass", "S1-SingPass", "S1-SingFail"])
    expected_f05_8 = round((1.0 + 1.0 + 0.0) / 3.0, 6)  # 0.666667
    assert res8["macro_f05"] == expected_f05_8, f"Test 8 failed: {res8}"
    assert res8["singleton_count"] == 2 and res8["singleton_correct"] == 1
    print(f"[PASS] Test 8: Combined multi-entity macro average -> Macro F0.5 = {res8['macro_f05']}")

    print("=" * 60)
    print("ALL 8 SELF-TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Local Macro F0.5 Evaluator for Amazon ML Challenge 2026."
    )
    parser.add_argument("--gt-path", type=Path, help="Path to ground truth TSV file")
    parser.add_argument("--pred-path", type=Path, help="Path to predictions TSV file")
    parser.add_argument("--s1-ids-path", type=Path, help="Optional text file listing validation S1 IDs (one per line)")
    parser.add_argument("--self-test", action="store_true", help="Run internal self-test suite and exit")

    args = parser.parse_args()

    if args.self_test or (not args.gt_path and not args.pred_path):
        success = run_self_tests()
        return 0 if success else 1

    if not args.gt_path or not args.pred_path:
        print("[ERROR] Both --gt-path and --pred-path are required for evaluation.", file=sys.stderr)
        return 1

    gt_matches = read_match_tsv(args.gt_path)
    pred_matches = read_match_tsv(args.pred_path)

    if args.s1_ids_path and args.s1_ids_path.exists():
        with open(args.s1_ids_path, "r", encoding="utf-8") as f:
            s1_ids = [line.strip() for line in f if line.strip()]
    else:
        # Default validation S1 IDs set to union of keys from GT and Predictions
        s1_ids = sorted(list(set(gt_matches.keys()) | set(pred_matches.keys())))

    results = compute_per_s1_f05(gt_matches, pred_matches, s1_ids)

    print("\n--- MACRO F0.5 EVALUATION RESULTS ---")
    print(f"Evaluated S1 Entities: {len(s1_ids):,}")
    print(f"Macro F0.5 Score:       {results['macro_f05']:.6f}")
    print(f"Mean Precision:         {results['mean_precision']:.6f}")
    print(f"Mean Recall:            {results['mean_recall']:.6f}")
    print(f"Singleton Accuracy:     {results['singleton_accuracy']:.6f} ({results['singleton_correct']:,} / {results['singleton_count']:,})")
    print(f"Non-Singleton F0.5:     {results['non_singleton_f05']:.6f} (Count: {results['non_singleton_count']:,})")
    print("------------------------------------\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())