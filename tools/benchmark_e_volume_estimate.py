#!/usr/bin/env python3
"""
benchmark_e_volume_estimate.py

Fast empirical candidate volume estimation for E-based union blocking rules
using deterministic 0.5% sub-sampling of Source 2 and Source 3 rows.

RULES ESTIMATED:
  1. E:                      any shared normalized address token
  2. C_union_E:              C UNION E
  3. G_union_E:              G UNION E
  4. A_G_union_C_union_E:    A UNION G UNION C UNION E

METADATA CONTEXT:
  C UNION E previously measured at approximately 99.70% sampled pair recall
  on the 20,000-S1 benchmark sample.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Reuse shared common utilities and normalization
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from common import (
        normalize_text,
        first_token,
        tokenize_address,
        stream_tsv_rows,
        format_bytes,
    )
except ImportError as e:
    print(f"[FATAL] Could not import common.py: {e}", file=sys.stderr)
    sys.exit(1)

try:
    from analyze_training_pairs import (
        percentile,
        find_gt_column,
        GT_ID_COL_CANDIDATES,
        GT_MATCH_COL_CANDIDATES,
    )
except ImportError as e:
    print(f"[FATAL] Could not import analyze_training_pairs.py: {e}", file=sys.stderr)
    sys.exit(1)

SCRIPT_VERSION = "1.0.0"
SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]

RULE_KEYS = [
    "E",
    "C_union_E",
    "G_union_E",
    "A_G_union_C_union_E",
]

RULE_LABELS = {
    "E": "E: any shared normalized address token",
    "C_union_E": "C UNION E: (country + first name token) OR shared address token",
    "G_union_E": "G UNION E: (country + first two name tokens) OR shared address token",
    "A_G_union_C_union_E": "A UNION G UNION C UNION E: exact name OR country+2 tokens OR country+1 token OR shared address token",
}


def get_rss_bytes() -> int:
    """Retrieves current process Resident Set Size (RSS) memory in bytes."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform.startswith("linux"):
            return usage * 1024
        return usage
    except Exception:
        return 0


def format_rss() -> str:
    b = get_rss_bytes()
    return format_bytes(b) if b > 0 else "Unknown"


def distribution_summary_list(values: list) -> dict:
    """Computes distribution stats from a list of numbers with clean rounding."""
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    values_sorted = sorted(values)
    n = len(values_sorted)
    raw_mean = sum(values_sorted) / n
    return {
        "count": n,
        "min": round(float(values_sorted[0]), 2),
        "median": round(float(percentile(values_sorted, 50)), 2),
        "mean": round(float(raw_mean), 2),
        "p95": round(float(percentile(values_sorted, 95)), 2),
        "p99": round(float(percentile(values_sorted, 99)), 2),
        "max": round(float(values_sorted[-1]), 2),
    }


def sample_ground_truth_entities(gt_path: Path, encoding: str, sample_size: int, seed: int) -> tuple:
    """Reservoir sample 20,000 S1 entities deterministically from train_ground_truth.tsv."""
    rng = random.Random(seed)
    reservoir = []
    seen_count = 0
    total_rows = 0

    with open(gt_path, "r", encoding=encoding, newline="") as f:
        header = None
        header_len = None
        id_idx = None
        for raw_line in f:
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            fields = line.split("\t")
            if header is None:
                header = fields
                header_len = len(fields)
                id_idx, _, _ = find_gt_column(header, GT_ID_COL_CANDIDATES)
                continue
            if len(fields) != header_len or id_idx is None:
                continue

            total_rows += 1
            s1_id = fields[id_idx]
            if not s1_id:
                continue

            seen_count += 1
            if len(reservoir) < sample_size:
                reservoir.append(s1_id)
            else:
                j = rng.randint(0, seen_count - 1)
                if j < sample_size:
                    reservoir[j] = s1_id

    return reservoir, total_rows


def extract_first_two_tokens(s_normalized: str) -> Optional[str]:
    parts = [t for t in s_normalized.split(" ") if t]
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Candidate volume estimation benchmark via sub-sampling.")
    parser.add_argument("--root", type=Path, default=None, help="Project root path")
    parser.add_argument("--encoding", type=str, default="utf-8")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--sample-rate", type=float, default=0.005, help="Sampling fraction per noisy source (default: 0.005)")
    parser.add_argument("--max-ram-gb", type=float, default=4.0)

    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    root = args.root if args.root else script_dir.parent
    max_ram_bytes = int(args.max_ram_gb * 1024 * 1024 * 1024)

    start_time = time.time()
    rss_baseline = get_rss_bytes()

    print("=" * 70)
    print(f"[INFO] Candidate Volume Estimation Benchmark v{SCRIPT_VERSION}")
    print(f"[INFO] Seed: {args.seed}, Sample Size: {args.sample_size:,}, Sample Rate: {args.sample_rate * 100:.2f}%")
    print(f"[INFO] Baseline Process RSS: {format_bytes(rss_baseline)}")
    print("=" * 70)

    # 1. Sample Ground Truth S1 Entities
    gt_path = root / "dataset" / "train" / "train_ground_truth.tsv"
    if not gt_path.exists():
        gt_path = root / "train_ground_truth.tsv"

    print(f"[INFO] Pass 1: Sampling S1 entities from {gt_path.resolve()}...")
    s1_sample_ids, total_gt_rows = sample_ground_truth_entities(gt_path, args.encoding, args.sample_size, args.seed)
    sample_set = set(s1_sample_ids)
    s1_id_to_idx = {sid: i for i, sid in enumerate(s1_sample_ids)}

    print(f"[INFO]   - Sampled S1 Entities: {len(sample_set):,}")

    # 2. Load S1 records for sample & build sample index
    s1_path = root / "dataset" / "train" / "train_source1.tsv"
    if not s1_path.exists():
        s1_path = root / "train_source1.tsv"

    print(f"[INFO] Pass 2: Loading S1 records and building sample indices from {s1_path.resolve()}...")

    s1_records_A: Dict[str, List[int]] = defaultdict(list)
    s1_records_G: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    s1_records_C: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    s1_records_E: Dict[str, List[int]] = defaultdict(list)

    loaded_s1 = 0
    for row in stream_tsv_rows(s1_path, SOURCE_COLUMNS, encoding=args.encoding):
        sid = row["entity_id"]
        if sid in sample_set:
            s1_idx = s1_id_to_idx[sid]
            country = row["country"].strip()
            norm_name = normalize_text(row["business_name"])
            norm_addr = normalize_text(row["business_address"])

            if norm_name:
                s1_records_A[norm_name].append(s1_idx)

            ftok = first_token(norm_name)
            if country and ftok:
                s1_records_C[(country, ftok)].append(s1_idx)

            first_two = extract_first_two_tokens(norm_name)
            if country and first_two:
                s1_records_G[(country, first_two)].append(s1_idx)

            addr_tokens = tokenize_address(norm_addr)
            if addr_tokens:
                for token in addr_tokens:
                    s1_records_E[token].append(s1_idx)

            loaded_s1 += 1

    rss_after_s1 = get_rss_bytes()
    print(f"[INFO]   - Loaded {loaded_s1:,} S1 sample records. RSS: {format_bytes(rss_after_s1)}")

    # 3. Stream Source 2 and Source 3 using Bernoulli sub-sampling
    s2_path = root / "dataset" / "train" / "train_source2.tsv"
    if not s2_path.exists():
        s2_path = root / "train_source2.tsv"
    s3_path = root / "dataset" / "train" / "train_source3.tsv"
    if not s3_path.exists():
        s3_path = root / "train_source3.tsv"

    # Tracking counters
    # cand_s2[rk][s1_idx] -> count of sampled S2 hits
    cand_s2 = {rk: [0] * len(s1_sample_ids) for rk in RULE_KEYS}
    cand_s3 = {rk: [0] * len(s1_sample_ids) for rk in RULE_KEYS}

    # Per-noisy-row hit distributions
    row_hit_sizes = {rk: [] for rk in RULE_KEYS}

    s2_total_rows = s2_selected_rows = 0
    s3_total_rows = s3_selected_rows = 0

    rng_s2 = random.Random(args.seed)
    rng_s3 = random.Random(args.seed + 1)

    for source_label, src_path, rng in [("Source 2", s2_path, rng_s2), ("Source 3", s3_path, rng_s3)]:
        print(f"[INFO] Pass 3: Streaming {source_label} (Target 0.5% Sample) from {src_path.resolve()}...")
        t_src_start = time.time()
        src_total = 0
        src_selected = 0

        target_cand_dict = cand_s2 if source_label == "Source 2" else cand_s3

        for row in stream_tsv_rows(src_path, SOURCE_COLUMNS, encoding=args.encoding):
            src_total += 1
            if rng.random() >= args.sample_rate:
                continue

            src_selected += 1
            country = row["country"].strip()
            norm_name = normalize_text(row["business_name"])
            norm_addr = normalize_text(row["business_address"])

            ftok = first_token(norm_name)
            first_two = extract_first_two_tokens(norm_name)
            tokens_addr = tokenize_address(norm_addr)

            # Atomic hit lookup against S1 sample index
            hits_A = set(s1_records_A.get(norm_name, [])) if norm_name else set()
            hits_G = set(s1_records_G.get((country, first_two), [])) if (country and first_two) else set()
            hits_C = set(s1_records_C.get((country, ftok), [])) if (country and ftok) else set()

            hits_E = set()
            if tokens_addr:
                for token in tokens_addr:
                    if token in s1_records_E:
                        hits_E.update(s1_records_E[token])

            # Deduplicated Union Sets per sampled noisy record
            rule_hits = {
                "E": hits_E,
                "C_union_E": hits_C | hits_E,
                "G_union_E": hits_G | hits_E,
                "A_G_union_C_union_E": hits_A | hits_G | hits_C | hits_E,
            }

            for rk in RULE_KEYS:
                matched = rule_hits[rk]
                hit_len = len(matched)
                row_hit_sizes[rk].append(hit_len)
                if matched:
                    for s1_idx in matched:
                        target_cand_dict[rk][s1_idx] += 1

            if src_selected % 5_000 == 0:
                cur_rss = get_rss_bytes()
                print(f"[INFO]   Scanned {src_total:,} rows, Selected {src_selected:,} {source_label} rows... (RSS: {format_bytes(cur_rss)})")
                if cur_rss > max_ram_bytes:
                    raise MemoryError(f"Process RSS ({format_bytes(cur_rss)}) exceeded safety ceiling ({format_bytes(max_ram_bytes)}).")

        t_src_elapsed = time.time() - t_src_start
        print(
            f"[INFO] {source_label} Complete: Scanned {src_total:,} total, Selected {src_selected:,} rows "
            f"({(src_selected / src_total) * 100:.3f}% fraction) in {t_src_elapsed:.2f}s. RSS: {format_rss()}"
        )

        if source_label == "Source 2":
            s2_total_rows, s2_selected_rows = src_total, src_selected
        else:
            s3_total_rows, s3_selected_rows = src_total, src_selected

    # 4. Compute Extrapolations & Per-S1 Candidate Distributions
    f_s2 = s2_selected_rows / s2_total_rows if s2_total_rows > 0 else 0.005
    f_s3 = s3_selected_rows / s3_total_rows if s3_total_rows > 0 else 0.005

    rule_results = {}
    for rk in RULE_KEYS:
        sampled_s2_hits = sum(cand_s2[rk])
        sampled_s3_hits = sum(cand_s3[rk])
        total_sampled_hits = sampled_s2_hits + sampled_s3_hits

        # Extrapolated candidate pairs for 20k S1 sample across full S2 & S3
        est_s2_full = sampled_s2_hits / f_s2
        est_s3_full = sampled_s3_hits / f_s3
        est_total_full_s1_sample = est_s2_full + est_s3_full

        est_mean_per_s1 = est_total_full_s1_sample / len(s1_sample_ids)

        # Compute estimated full candidate count for each S1 entity i
        est_counts_per_s1 = [
            (cand_s2[rk][i] / f_s2) + (cand_s3[rk][i] / f_s3)
            for i in range(len(s1_sample_ids))
        ]

        per_row_dist = distribution_summary_list(row_hit_sizes[rk])
        per_s1_dist = distribution_summary_list(est_counts_per_s1)

        rule_results[rk] = {
            "label": RULE_LABELS[rk],
            "sampled_candidate_pair_hits": total_sampled_hits,
            "estimated_full_training_candidate_pairs_20k_sample": round(est_total_full_s1_sample, 0),
            "estimated_mean_candidates_per_s1_entity": round(est_mean_per_s1, 2),
            "per_noisy_row_hit_size_distribution": per_row_dist,
            "approximate_per_s1_candidate_distribution": per_s1_dist,
        }

    # 5. Assemble and Write Reports
    total_elapsed = time.time() - start_time
    total_noisy_scanned = s2_total_rows + s3_total_rows
    total_noisy_selected = s2_selected_rows + s3_selected_rows

    report = {
        "benchmark_metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "script": "tools/benchmark_e_volume_estimate.py",
            "script_version": SCRIPT_VERSION,
            "seed": args.seed,
            "target_sample_rate": args.sample_rate,
            "actual_sample_size_s1": len(s1_sample_ids),
            "gt_total_rows": total_gt_rows,
            "s2_total_rows": s2_total_rows,
            "s2_selected_rows": s2_selected_rows,
            "s2_sampling_fraction": round(f_s2, 6),
            "s3_total_rows": s3_total_rows,
            "s3_selected_rows": s3_selected_rows,
            "s3_sampling_fraction": round(f_s3, 6),
            "total_noisy_rows_scanned": total_noisy_scanned,
            "total_noisy_rows_selected": total_noisy_selected,
            "peak_rss_ram": format_rss(),
            "total_elapsed_seconds": round(total_elapsed, 2),
            "throughput_rows_per_sec": round(total_noisy_scanned / total_elapsed, 0) if total_elapsed > 0 else 0,
            "contextual_recall_metadata": (
                "C UNION E previously measured at approximately 99.70% sampled pair recall "
                "on the 20,000-S1 benchmark sample."
            ),
        },
        "rule_volume_estimates": rule_results,
    }

    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "benchmark_e_volume_estimate.json"
    md_path = reports_dir / "benchmark_e_volume_estimate.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_lines = [
        "# Candidate Volume Estimation Benchmark Report",
        "",
        "## Executive Context",
        "- **Contextual Recall Reference**: C UNION E previously measured at **~99.70%** sampled pair recall on the 20,000-S1 benchmark sample.",
        "- **Purpose**: Determine order-of-magnitude candidate volume for E-based union rules to evaluate computational feasibility.",
        "",
        "## Sub-Sampling Execution Profile",
        f"- Generated (UTC): {report['benchmark_metadata']['generated_at_utc']}",
        f"- Seed: {args.seed}, S1 Entity Sample: {len(s1_sample_ids):,}",
        f"- Source 2: Scanned {s2_total_rows:,} rows, Selected {s2_selected_rows:,} rows ({f_s2 * 100:.3f}%)",
        f"- Source 3: Scanned {s3_total_rows:,} rows, Selected {s3_selected_rows:,} rows ({f_s3 * 100:.3f}%)",
        f"- Total Noisy Rows Scanned: {total_noisy_scanned:,} (Selected: {total_noisy_selected:,})",
        f"- Peak Process RSS Memory: {format_rss()}",
        f"- Total Elapsed Time: {total_elapsed:.2f}s ({report['benchmark_metadata']['throughput_rows_per_sec']:,} rows/s)",
        "",
        "## Estimated Candidate Volume Summary (20,000 S1 Entities)",
        "",
        "| Rule Key | Label | Observed Sampled Pair Hits | Estimated Full Pairs (20k S1) | Estimated Mean Cands / S1 |",
        "|---|---|---|---|---|",
    ]

    for rk in RULE_KEYS:
        res = rule_results[rk]
        md_lines.append(
            f"| `{rk}` | {res['label']} | {res['sampled_candidate_pair_hits']:,} | **{int(res['estimated_full_training_candidate_pairs_20k_sample']):,}** | **{res['estimated_mean_candidates_per_s1_entity']:,.2f}** |"
        )

    md_lines.extend([
        "",
        "## Per-Noisy-Row Hit Size Distributions (How many S1 entities a single noisy record expands to)",
        "",
        "| Rule Key | Min | Median | Mean | P95 | P99 | Max |",
        "|---|---|---|---|---|---|---|",
    ])

    for rk in RULE_KEYS:
        st = rule_results[rk]["per_noisy_row_hit_size_distribution"]
        md_lines.append(
            f"| `{rk}` | {st['min']} | {st['median']:.1f} | {st['mean']:.2f} | {st['p95']:.1f} | {st['p99']:.1f} | {st['max']:,} |"
        )

    md_lines.extend([
        "",
        "## Approximate Per-S1 Candidate Distributions (Estimated full candidate counts across 20k S1 entities)",
        "",
        "| Rule Key | Min | Median | Mean | P95 | P99 | Max |",
        "|---|---|---|---|---|---|---|",
    ])

    for rk in RULE_KEYS:
        st = rule_results[rk]["approximate_per_s1_candidate_distribution"]
        md_lines.append(
            f"| `{rk}` | {st['min']:.1f} | {st['median']:.1f} | {st['mean']:.2f} | {st['p95']:.1f} | {st['p99']:.1f} | {st['max']:,} |"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print("=" * 70)
    print(f"[SUCCESS] Candidate Volume Estimation Benchmark Completed in {total_elapsed:.2f}s.")
    print(f"[INFO] Report Wrote: {json_path.resolve()}")
    print(f"[INFO] Report Wrote: {md_path.resolve()}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())