#!/usr/bin/env python3
"""
benchmark_actual_v1_recall.py

Benchmark script to measure the actual pair recall and candidate set statistics
of the bounded V1 blocker (A + G + E_rare@200 with top_k_broad=15) on a 20,000 S1 sample.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import format_rss, stream_tsv_rows
from evaluate_f05 import read_match_tsv
from v1_pipeline import (
    SOURCE_COLUMNS,
    build_s1_compact_index,
    extract_candidates_for_row,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Actual V1 Bounded Blocker Recall on 20k S1 Sample"
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=None,
        help="Directory containing train_source1.tsv, train_source2.tsv, train_source3.tsv, and train_ground_truth.tsv",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root path of project dataset fallback",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20000,
        help="Number of S1 entities to sample (default: 20000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for sampling S1 entities (default: 1337)",
    )
    parser.add_argument(
        "--top-k-broad",
        type=int,
        default=15,
        help="Top-K broad candidates per noisy row (default: 15)",
    )

    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    root = args.root if args.root else script_dir.parent

    # Path Resolution
    if args.train_dir is not None:
        t_dir = args.train_dir.expanduser().resolve()
        gt_path = t_dir / "train_ground_truth.tsv"
        s1_path = t_dir / "train_source1.tsv"
        s2_path = t_dir / "train_source2.tsv"
        s3_path = t_dir / "train_source3.tsv"
    else:
        gt_path = root / "dataset" / "train" / "train_ground_truth.tsv"
        if not gt_path.exists():
            gt_path = root / "train_ground_truth.tsv"

        s1_path = root / "dataset" / "train" / "train_source1.tsv"
        if not s1_path.exists():
            s1_path = root / "train_source1.tsv"

        s2_path = root / "dataset" / "train" / "train_source2.tsv"
        if not s2_path.exists():
            s2_path = root / "train_source2.tsv"

        s3_path = root / "dataset" / "train" / "train_source3.tsv"
        if not s3_path.exists():
            s3_path = root / "train_source3.tsv"

    print("=" * 70)
    print("[INFO] Starting Actual V1 Bounded Blocker Recall Benchmark")
    print(f"[INFO] Ground Truth Path: {gt_path.resolve()}")
    print(f"[INFO] Source 1 Path:     {s1_path.resolve()}")
    print(f"[INFO] Source 2 Path:     {s2_path.resolve()}")
    print(f"[INFO] Source 3 Path:     {s3_path.resolve()}")
    print(f"[INFO] Sample Size:       {args.sample_size:,} S1 entities (seed={args.seed})")
    print(f"[INFO] Blocker Settings:   A + G + E_rare@200 (top_k_broad={args.top_k_broad})")
    print("=" * 70)

    # 1. Load full Ground Truth
    t0 = time.time()
    print("[INFO] Loading complete ground truth...")
    gt_matches = read_match_tsv(gt_path)
    all_s1_ids = sorted(gt_matches.keys())
    print(f"[INFO] Complete GT loaded: {len(all_s1_ids):,} S1 entities in {time.time() - t0:.2f}s.")

    # 2. Deterministically sample S1 IDs
    rng = random.Random(args.seed)
    if len(all_s1_ids) < args.sample_size:
        sampled_s1_ids = all_s1_ids
    else:
        sampled_s1_ids = rng.sample(all_s1_ids, args.sample_size)

    # 3. Collect ALL GT noisy IDs belonging to those sampled S1 IDs
    noisy_to_target_s1s: Dict[str, Set[str]] = defaultdict(set)
    target_gt_pairs: Set[Tuple[str, str]] = set()

    for s1_id in sampled_s1_ids:
        gt_noisy_set = gt_matches.get(s1_id, set())
        for noisy_id in gt_noisy_set:
            target_gt_pairs.add((s1_id, noisy_id))
            noisy_to_target_s1s[noisy_id].add(s1_id)

    print(
        f"[INFO] Sampled {len(sampled_s1_ids):,} S1 entities. "
        f"Total target GT pairs: {len(target_gt_pairs):,}. "
        f"Unique target noisy entity IDs: {len(noisy_to_target_s1s):,}."
    )

    # 4. Build FULL Source-1 Compact Index
    s1_idx_obj, doc_freq_addr = build_s1_compact_index(s1_path, max_doc_freq=200)
    s1_id_to_idx = {s1_id: idx for idx, s1_id in enumerate(s1_idx_obj.s1_ids)}

    # 5. Stream Source 2 and Source 3
    print("[INFO] Streaming S2/S3 to benchmark bounded candidate generation...")
    t_stream = time.time()

    retrieved_gt_pairs: Set[Tuple[str, str]] = set()
    retrieved_per_s1: Dict[str, Set[str]] = defaultdict(set)
    candidate_counts: List[int] = []

    scanned_rows = 0
    targeted_rows_found = 0

    for src_path in [s2_path, s3_path]:
        print(f"[INFO]   Scanning {src_path.name}...")
        for row in stream_tsv_rows(src_path, SOURCE_COLUMNS):
            scanned_rows += 1
            noisy_id = row["entity_id"]

            if noisy_id in noisy_to_target_s1s:
                targeted_rows_found += 1
                cands = extract_candidates_for_row(
                    row, s1_idx_obj, doc_freq_addr, top_k_broad=args.top_k_broad
                )
                candidate_counts.append(len(cands))

                target_s1_ids = noisy_to_target_s1s[noisy_id]
                for s1_id in target_s1_ids:
                    s1_idx = s1_id_to_idx.get(s1_id)
                    if s1_idx is not None and s1_idx in cands:
                        retrieved_gt_pairs.add((s1_id, noisy_id))
                        retrieved_per_s1[s1_id].add(noisy_id)

            if scanned_rows % 1_000_000 == 0:
                print(
                    f"[INFO]     Scanned {scanned_rows:,} rows... "
                    f"Targeted noisy rows found: {targeted_rows_found:,}, "
                    f"Retrieved GT pairs: {len(retrieved_gt_pairs):,} / {len(target_gt_pairs):,}. "
                    f"RSS: {format_rss()}"
                )

    print(
        f"[INFO] Streaming complete in {time.time() - t_stream:.2f}s. "
        f"Total rows scanned: {scanned_rows:,}. Targeted rows found: {targeted_rows_found:,}."
    )

    # 6. Calculate Metrics
    tot_gt_pairs = len(target_gt_pairs)
    ret_gt_pairs = len(retrieved_gt_pairs)
    pair_recall = (ret_gt_pairs / tot_gt_pairs) if tot_gt_pairs > 0 else 0.0

    # Entity-level full retrieval
    full_retrieval_count = 0
    for s1_id in sampled_s1_ids:
        gt_set = gt_matches.get(s1_id, set())
        ret_set = retrieved_per_s1.get(s1_id, set())
        if gt_set == ret_set:
            full_retrieval_count += 1

    entity_full_retrieval_rate = (
        full_retrieval_count / len(sampled_s1_ids) if sampled_s1_ids else 0.0
    )

    # Candidate set statistics
    if candidate_counts:
        arr_c = np.array(candidate_counts, dtype=np.int32)
        c_mean = float(np.mean(arr_c))
        c_median = float(np.median(arr_c))
        c_p95 = float(np.percentile(arr_c, 95))
        c_p99 = float(np.percentile(arr_c, 99))
        c_max = int(np.max(arr_c))
    else:
        c_mean = c_median = c_p95 = c_p99 = c_max = 0

    print("=" * 70)
    print("BENCHMARK RESULTS: BOUNDED V1 BLOCKER (20,000 S1 Sample)")
    print("=" * 70)
    print(f"Sampled S1 Entities:                     {len(sampled_s1_ids):,}")
    print(f"Total Sampled GT Pairs:                  {tot_gt_pairs:,}")
    print(f"Retrieved Sampled GT Pairs:              {ret_gt_pairs:,}")
    print(f"Actual Bounded Pair Recall:              {pair_recall * 100:.2f}%")
    print("-" * 70)
    print(f"S1 Entities with All GT Pairs Retrieved: {full_retrieval_count:,} / {len(sampled_s1_ids):,}")
    print(f"Entity-Level Full Retrieval Rate:        {entity_full_retrieval_rate * 100:.2f}%")
    print("-" * 70)
    print("Candidate Set Size Statistics (per targeted noisy row):")
    print(f"  Mean:   {c_mean:.2f}")
    print(f"  Median: {c_median:.1f}")
    print(f"  p95:    {c_p95:.1f}")
    print(f"  p99:    {c_p99:.1f}")
    print(f"  Max:    {c_max:,}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())