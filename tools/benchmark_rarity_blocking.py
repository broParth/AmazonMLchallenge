#!/usr/bin/env python3
"""
benchmark_rarity_blocking.py

Empirical benchmark evaluating rarity-aware blocking rules across document
frequency thresholds K in {5, 10, 20, 50, 100, 200}.

RULES BENCHMARKED (per K):
  1. E_rare@K:               Shared normalized address token with doc_freq <= K
  2. N_rare@K:               Same country AND shared normalized name token with doc_freq <= K
  3. A_G_union_CE_rare@K:    A UNION G UNION (country + first_name_token AND shared rare address token)
  4. A_G_union_E_rare@K:     A UNION G UNION E_rare@K
  5. A_G_union_Nrare@K:      A UNION G UNION N_rare@K
  6. A_G_union_Nrare_Erare@K: A UNION G UNION N_rare@K UNION E_rare@K

MEMORY & COMPUTATION GUARANTEES:
  - Document frequency computed over FULL train_source1.tsv.
  - Candidate indices constructed ONLY for the 20,000 sampled S1 entities.
  - Recall evaluated ONLY on target GT noisy records (zero unnecessary expansion).
  - Volume estimated via deterministic 0.5% Bernoulli sampling of S2 and S3.
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
K_THRESHOLDS = [5, 10, 20, 50, 100, 200]

RULE_BASE_KEYS = [
    "E_rare",
    "N_rare",
    "A_G_union_CE_rare",
    "A_G_union_E_rare",
    "A_G_union_Nrare",
    "A_G_union_Nrare_Erare",
]


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
        id_idx = match_idx = None
        for raw_line in f:
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            fields = line.split("\t")
            if header is None:
                header = fields
                header_len = len(fields)
                id_idx, _, _ = find_gt_column(header, GT_ID_COL_CANDIDATES)
                match_idx, _, _ = find_gt_column(header, GT_MATCH_COL_CANDIDATES)
                continue
            if len(fields) != header_len or id_idx is None:
                continue

            total_rows += 1
            s1_id = fields[id_idx]
            if not s1_id:
                continue

            matched_ids = []
            if match_idx is not None:
                raw_match = fields[match_idx]
                matched_ids = [m.strip() for m in raw_match.split(",") if m.strip().startswith(("S2-", "S3-"))]

            seen_count += 1
            entry = (s1_id, matched_ids)
            if len(reservoir) < sample_size:
                reservoir.append(entry)
            else:
                j = rng.randint(0, seen_count - 1)
                if j < sample_size:
                    reservoir[j] = entry

    gt_matches = {s1_id: set(matched) for s1_id, matched in reservoir}
    return gt_matches, total_rows


def extract_first_two_tokens(s_normalized: str) -> Optional[str]:
    parts = [t for t in s_normalized.split(" ") if t]
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Rarity-aware blocking benchmark.")
    parser.add_argument("--root", type=Path, default=None, help="Project root path")
    parser.add_argument("--encoding", type=str, default="utf-8")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--sample-rate", type=float, default=0.005, help="Sampling fraction for volume estimation (default: 0.005)")
    parser.add_argument("--max-ram-gb", type=float, default=4.0)

    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    root = args.root if args.root else script_dir.parent
    max_ram_bytes = int(args.max_ram_gb * 1024 * 1024 * 1024)

    start_time = time.time()
    rss_baseline = get_rss_bytes()

    print("=" * 70)
    print(f"[INFO] Rarity-Aware Blocking Benchmark v{SCRIPT_VERSION}")
    print(f"[INFO] Seed: {args.seed}, Sample S1 Size: {args.sample_size:,}, Volume Sample Rate: {args.sample_rate * 100:.2f}%")
    print(f"[INFO] Baseline Process RSS: {format_bytes(rss_baseline)}")
    print("=" * 70)

    # 1. Sample Ground Truth S1 Entities
    gt_path = root / "dataset" / "train" / "train_ground_truth.tsv"
    if not gt_path.exists():
        gt_path = root / "train_ground_truth.tsv"

    print(f"[INFO] Pass 1: Sampling S1 entities from {gt_path.resolve()}...")
    gt_matches, total_gt_rows = sample_ground_truth_entities(gt_path, args.encoding, args.sample_size, args.seed)

    s1_sample_ids = list(gt_matches.keys())
    sample_set = set(s1_sample_ids)
    singletons = {sid for sid, m in gt_matches.items() if len(m) == 0}
    non_singletons = sample_set - singletons
    all_referenced_true_matches = set().union(*gt_matches.values())

    s1_id_to_idx = {sid: i for i, sid in enumerate(s1_sample_ids)}
    s1_gt_sets = [gt_matches[sid] for sid in s1_sample_ids]

    # Map GT noisy target ID -> Set of matching sampled S1 indices
    gt_noisy_to_s1: Dict[str, Set[int]] = defaultdict(set)
    for s1_idx, gt_set in enumerate(s1_gt_sets):
        for noisy_id in gt_set:
            gt_noisy_to_s1[noisy_id].add(s1_idx)

    print(f"[INFO]   - Sampled S1 Entities: {len(sample_set):,} ({len(singletons):,} singletons, {len(non_singletons):,} non-singletons)")
    print(f"[INFO]   - Ground Truth Match Targets: {len(all_referenced_true_matches):,} unique S2/S3 IDs mapped to sample.")

    # 2. Compute Document Frequency over FULL train_source1.tsv & Build Sample Index
    s1_path = root / "dataset" / "train" / "train_source1.tsv"
    if not s1_path.exists():
        s1_path = root / "train_source1.tsv"

    print(f"[INFO] Pass 2: Computing full-corpus token document frequencies from {s1_path.resolve()}...")

    doc_freq_addr: Counter = Counter()
    doc_freq_name: Counter = Counter()

    # Index containers for the 20,000 sampled S1 entities
    s1_records_A: Dict[str, List[int]] = defaultdict(list)
    s1_records_G: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    s1_records_C: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    s1_records_E_addr: Dict[str, List[int]] = defaultdict(list)
    s1_records_N_name: Dict[str, List[int]] = defaultdict(list)

    s1_country = [""] * len(s1_sample_ids)
    s1_addr_tokens = [set() for _ in range(len(s1_sample_ids))]

    s1_scanned = 0
    s1_sampled_loaded = 0

    for row in stream_tsv_rows(s1_path, SOURCE_COLUMNS, encoding=args.encoding):
        s1_scanned += 1
        sid = row["entity_id"]
        country = row["country"].strip()
        norm_name = normalize_text(row["business_name"])
        norm_addr = normalize_text(row["business_address"])

        addr_tokens = tokenize_address(norm_addr)
        name_tokens = tokenize_address(norm_name)

        # Full S1 Corpus Document Frequency Counters
        for token in addr_tokens:
            doc_freq_addr[token] += 1
        for token in name_tokens:
            doc_freq_name[token] += 1

        # Build Inverted Indexes ONLY for the 20k Sampled Entities
        if sid in sample_set:
            s1_idx = s1_id_to_idx[sid]
            s1_country[s1_idx] = country
            s1_addr_tokens[s1_idx] = addr_tokens

            if norm_name:
                s1_records_A[norm_name].append(s1_idx)

            ftok = first_token(norm_name)
            if country and ftok:
                s1_records_C[(country, ftok)].append(s1_idx)

            first_two = extract_first_two_tokens(norm_name)
            if country and first_two:
                s1_records_G[(country, first_two)].append(s1_idx)

            for token in addr_tokens:
                s1_records_E_addr[token].append(s1_idx)

            for token in name_tokens:
                s1_records_N_name[token].append(s1_idx)

            s1_sampled_loaded += 1

        if s1_scanned % 1_000_000 == 0:
            print(f"[INFO]   Scanned {s1_scanned:,} S1 rows... (RSS: {format_rss()})")

    rss_after_s1 = get_rss_bytes()
    print(f"[INFO] Pass 2 Complete: Scanned {s1_scanned:,} S1 rows, Loaded {s1_sampled_loaded:,} sample records. RSS: {format_bytes(rss_after_s1)}")
    print(f"[INFO]   - Unique Address Tokens: {len(doc_freq_addr):,}, Unique Business Name Tokens: {len(doc_freq_name):,}")

    # Token Rarity Statistics
    surviving_addr_tokens = {K: sum(1 for cnt in doc_freq_addr.values() if cnt <= K) for K in K_THRESHOLDS}
    surviving_name_tokens = {K: sum(1 for cnt in doc_freq_name.values() if cnt <= K) for K in K_THRESHOLDS}

    # 3. Stream Source 2 and Source 3 for Recall and Volume Estimation
    s2_path = root / "dataset" / "train" / "train_source2.tsv"
    if not s2_path.exists():
        s2_path = root / "train_source2.tsv"
    s3_path = root / "dataset" / "train" / "train_source3.tsv"
    if not s3_path.exists():
        s3_path = root / "train_source3.tsv"

    # Trackers for recall: gt_retrieved[rule][K][s1_idx] -> Set[retrieved_noisy_gt_ids]
    gt_retrieved = {
        rule: {K: [set() for _ in range(len(s1_sample_ids))] for K in K_THRESHOLDS}
        for rule in RULE_BASE_KEYS
    }

    # Trackers for volume: cand_volume_s2/s3[rule][K][s1_idx] -> integer hit count
    volume_s2 = {
        rule: {K: [0] * len(s1_sample_ids) for K in K_THRESHOLDS}
        for rule in RULE_BASE_KEYS
    }
    volume_s3 = {
        rule: {K: [0] * len(s1_sample_ids) for K in K_THRESHOLDS}
        for rule in RULE_BASE_KEYS
    }

    s2_total_rows = s2_selected_rows = 0
    s3_total_rows = s3_selected_rows = 0
    referenced_records = {}

    rng_s2 = random.Random(args.seed)
    rng_s3 = random.Random(args.seed + 1)

    for source_label, src_path, rng in [("Source 2", s2_path, rng_s2), ("Source 3", s3_path, rng_s3)]:
        print(f"[INFO] Pass 3: Streaming {source_label} from {src_path.resolve()}...")
        t_src_start = time.time()
        src_total = 0
        src_selected = 0

        vol_dict = volume_s2 if source_label == "Source 2" else volume_s3

        for row in stream_tsv_rows(src_path, SOURCE_COLUMNS, encoding=args.encoding):
            src_total += 1
            noisy_id = row["entity_id"]

            is_gt_target = noisy_id in gt_noisy_to_s1
            is_vol_sampled = rng.random() < args.sample_rate

            if is_vol_sampled:
                src_selected += 1

            if not is_gt_target and not is_vol_sampled:
                continue

            country = row["country"].strip()
            norm_name = normalize_text(row["business_name"])
            norm_addr = normalize_text(row["business_address"])

            if noisy_id in all_referenced_true_matches:
                referenced_records[noisy_id] = True

            ftok = first_token(norm_name)
            first_two = extract_first_two_tokens(norm_name)
            addr_tokens = tokenize_address(norm_addr)
            name_tokens = tokenize_address(norm_name)

            # Atomic lookups against sample index
            hits_A = set(s1_records_A.get(norm_name, [])) if norm_name else set()
            hits_G = set(s1_records_G.get((country, first_two), [])) if (country and first_two) else set()
            hits_C = set(s1_records_C.get((country, ftok), [])) if (country and ftok) else set()

            # Pre-group rare tokens for current record across K thresholds
            addr_tokens_by_k = {
                K: [t for t in addr_tokens if doc_freq_addr.get(t, 0) <= K]
                for K in K_THRESHOLDS
            }
            name_tokens_by_k = {
                K: [t for t in name_tokens if doc_freq_name.get(t, 0) <= K]
                for K in K_THRESHOLDS
            }

            for K in K_THRESHOLDS:
                # 1. E_rare@K
                rare_addr = addr_tokens_by_k[K]
                hits_E_rare = set()
                if rare_addr:
                    for t in rare_addr:
                        if t in s1_records_E_addr:
                            hits_E_rare.update(s1_records_E_addr[t])

                # 2. N_rare@K (Country match + rare name token)
                rare_name = name_tokens_by_k[K]
                hits_N_rare = set()
                if rare_name and country:
                    for t in rare_name:
                        if t in s1_records_N_name:
                            for candidate_idx in s1_records_N_name[t]:
                                if s1_country[candidate_idx] == country:
                                    hits_N_rare.add(candidate_idx)

                # 3. C_and_E_rare@K
                hits_CE_rare = set()
                if hits_C and rare_addr:
                    rare_addr_set = set(rare_addr)
                    for candidate_idx in hits_C:
                        if bool(s1_addr_tokens[candidate_idx] & rare_addr_set):
                            hits_CE_rare.add(candidate_idx)

                # Construct Union Sets
                hits_A_G_union_CE_rare = hits_A | hits_G | hits_CE_rare
                hits_A_G_union_E_rare = hits_A | hits_G | hits_E_rare
                hits_A_G_union_Nrare = hits_A | hits_G | hits_N_rare
                hits_A_G_union_Nrare_Erare = hits_A | hits_G | hits_N_rare | hits_E_rare

                rule_map = {
                    "E_rare": hits_E_rare,
                    "N_rare": hits_N_rare,
                    "A_G_union_CE_rare": hits_A_G_union_CE_rare,
                    "A_G_union_E_rare": hits_A_G_union_E_rare,
                    "A_G_union_Nrare": hits_A_G_union_Nrare,
                    "A_G_union_Nrare_Erare": hits_A_G_union_Nrare_Erare,
                }

                # Evaluation for Recall (GT Target Records)
                if is_gt_target:
                    target_s1_indices = gt_noisy_to_s1[noisy_id]
                    for rk, hits in rule_map.items():
                        matched_targets = target_s1_indices & hits
                        if matched_targets:
                            for s1_idx in matched_targets:
                                gt_retrieved[rk][K][s1_idx].add(noisy_id)

                # Evaluation for Volume Estimation (Bernoulli Sampled Records)
                if is_vol_sampled:
                    for rk, hits in rule_map.items():
                        if hits:
                            for s1_idx in hits:
                                vol_dict[rk][K][s1_idx] += 1

            if src_total % 1_000_000 == 0:
                cur_rss = get_rss_bytes()
                print(f"[INFO]   Scanned {src_total:,} rows... (RSS: {format_bytes(cur_rss)})")
                if cur_rss > max_ram_bytes:
                    raise MemoryError(f"Process RSS ({format_bytes(cur_rss)}) exceeded safety ceiling ({format_bytes(max_ram_bytes)}).")

        t_src_elapsed = time.time() - t_src_start
        print(
            f"[INFO] {source_label} Complete: Scanned {src_total:,} rows, Selected {src_selected:,} for volume estimation "
            f"in {t_src_elapsed:.2f}s. RSS: {format_rss()}"
        )

        if source_label == "Source 2":
            s2_total_rows, s2_selected_rows = src_total, src_selected
        else:
            s3_total_rows, s3_selected_rows = src_total, src_selected

    # 4. Resolve GT Pairs
    resolved_pairs = []
    unresolved_pairs = 0
    for s1_id, matched_set in gt_matches.items():
        s1_idx = s1_id_to_idx[s1_id]
        for mid in matched_set:
            if mid in referenced_records:
                resolved_pairs.append((s1_idx, s1_id, mid))
            else:
                unresolved_pairs += 1

    total_resolved_pairs = len(resolved_pairs)
    print(f"[INFO] GT Pair Resolution: {total_resolved_pairs:,} resolved pairs used for recall denominator ({unresolved_pairs:,} unresolved).")

    # 5. Calculate Metrics per Rule and K
    f_s2 = s2_selected_rows / s2_total_rows if s2_total_rows > 0 else 0.005
    f_s3 = s3_selected_rows / s3_total_rows if s3_total_rows > 0 else 0.005

    rule_results = {}
    for rk in RULE_BASE_KEYS:
        rule_results[rk] = {}
        for K in K_THRESHOLDS:
            # Recall Metrics
            pair_hits = sum(len(gt_retrieved[rk][K][s1_idx]) for s1_idx in range(len(s1_sample_ids)))
            pair_recall = pair_hits / total_resolved_pairs if total_resolved_pairs > 0 else 0.0

            all_retrieved_entities = 0
            non_singleton_considered = 0
            for s1_id in non_singletons:
                s1_idx = s1_id_to_idx[s1_id]
                true_set = gt_matches[s1_id]
                resolved_true_set = {m for m in true_set if m in referenced_records}
                if not resolved_true_set:
                    continue
                non_singleton_considered += 1
                if resolved_true_set.issubset(gt_retrieved[rk][K][s1_idx]):
                    all_retrieved_entities += 1

            entity_recall = all_retrieved_entities / non_singleton_considered if non_singleton_considered > 0 else 0.0

            # Volume Metrics
            sampled_s2_hits = sum(volume_s2[rk][K])
            sampled_s3_hits = sum(volume_s3[rk][K])
            total_sampled_hits = sampled_s2_hits + sampled_s3_hits

            est_s2_full = sampled_s2_hits / f_s2
            est_s3_full = sampled_s3_hits / f_s3
            est_total_full_s1_sample = est_s2_full + est_s3_full
            est_mean_per_s1 = est_total_full_s1_sample / len(s1_sample_ids)

            est_counts_per_s1 = [
                (volume_s2[rk][K][i] / f_s2) + (volume_s3[rk][K][i] / f_s3)
                for i in range(len(s1_sample_ids))
            ]
            per_s1_dist = distribution_summary_list(est_counts_per_s1)

            rule_results[rk][f"K_{K}"] = {
                "rule_key": rk,
                "K": K,
                "pair_hits": pair_hits,
                "total_resolved_pairs": total_resolved_pairs,
                "pair_recall": round(pair_recall, 6),
                "entities_all_retrieved": all_retrieved_entities,
                "entities_considered": non_singleton_considered,
                "entity_level_all_retrieved_rate": round(entity_recall, 6),
                "sampled_candidate_pair_hits": total_sampled_hits,
                "estimated_full_training_candidate_pairs_20k_sample": round(est_total_full_s1_sample, 0),
                "estimated_mean_candidates_per_s1_entity": round(est_mean_per_s1, 2),
                "approximate_per_s1_candidate_distribution": per_s1_dist,
            }

    # 6. Assemble & Write Report
    total_elapsed = time.time() - start_time
    total_noisy_scanned = s2_total_rows + s3_total_rows

    addr_dist = distribution_summary_list(list(doc_freq_addr.values()))
    name_dist = distribution_summary_list(list(doc_freq_name.values()))

    report = {
        "benchmark_metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "script": "tools/benchmark_rarity_blocking.py",
            "script_version": SCRIPT_VERSION,
            "seed": args.seed,
            "sample_size_s1": len(s1_sample_ids),
            "s1_total_scanned": s1_scanned,
            "s2_total_scanned": s2_total_rows,
            "s2_selected_rows": s2_selected_rows,
            "s2_sampling_fraction": round(f_s2, 6),
            "s3_total_scanned": s3_total_rows,
            "s3_selected_rows": s3_selected_rows,
            "s3_sampling_fraction": round(f_s3, 6),
            "resolved_gt_pairs": total_resolved_pairs,
            "unresolved_gt_pairs": unresolved_pairs,
            "peak_rss_ram": format_rss(),
            "total_elapsed_seconds": round(total_elapsed, 2),
            "throughput_rows_per_sec": round((s1_scanned + total_noisy_scanned) / total_elapsed, 0) if total_elapsed > 0 else 0,
        },
        "token_rarity_statistics": {
            "address_tokens": {
                "unique_token_count": len(doc_freq_addr),
                "distribution": addr_dist,
                "surviving_rare_tokens_by_K": surviving_addr_tokens,
                "top_10_frequent_tokens": doc_freq_addr.most_common(10),
            },
            "name_tokens": {
                "unique_token_count": len(doc_freq_name),
                "distribution": name_dist,
                "surviving_rare_tokens_by_K": surviving_name_tokens,
                "top_10_frequent_tokens": doc_freq_name.most_common(10),
            },
        },
        "rarity_rules": rule_results,
    }

    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "benchmark_rarity_blocking.json"
    md_path = reports_dir / "benchmark_rarity_blocking.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Build Markdown Summary
    md_lines = [
        "# Rarity-Aware Blocking Benchmark Report",
        "",
        "## Execution Metadata",
        f"- Generated (UTC): {report['benchmark_metadata']['generated_at_utc']}",
        f"- Seed: {args.seed}, S1 Sample Size: {len(s1_sample_ids):,}",
        f"- Full S1 Records Scanned (Token Document Frequency): {s1_scanned:,}",
        f"- S2 Rows Scanned: {s2_total_rows:,} (Volume Sampled: {s2_selected_rows:,}, {f_s2 * 100:.3f}%)",
        f"- S3 Rows Scanned: {s3_total_rows:,} (Volume Sampled: {s3_selected_rows:,}, {f_s3 * 100:.3f}%)",
        f"- Resolved GT Pairs: {total_resolved_pairs:,} (Unresolved: {unresolved_pairs:,})",
        f"- Peak Process RSS Memory: {format_rss()}",
        f"- Total Elapsed Time: {total_elapsed:.2f}s ({report['benchmark_metadata']['throughput_rows_per_sec']:,} rows/s)",
        "",
        "## Token Rarity Profiles (Full S1 Corpus)",
        f"- Unique Address Tokens: {len(doc_freq_addr):,} (Surviving at K=200: {surviving_addr_tokens[200]:,})",
        f"- Unique Business Name Tokens: {len(doc_freq_name):,} (Surviving at K=200: {surviving_name_tokens[200]:,})",
        f"- Address Token Document Frequency: Mean = {addr_dist['mean']:.2f}, Median = {addr_dist['median']:.1f}, P95 = {addr_dist['p95']:.1f}, Max = {addr_dist['max']:,}",
        f"- Name Token Document Frequency: Mean = {name_dist['mean']:.2f}, Median = {name_dist['median']:.1f}, P95 = {name_dist['p95']:.1f}, Max = {name_dist['max']:,}",
        "",
        "## Rarity-Aware Blocking Rules Performance Comparison",
        "",
        "| Rule | K | GT Pair Recall | Entity All-Retrieved Rate | Est. Mean Cands / S1 | Median | P95 | P99 | Max | Est. Total Candidates (20k S1) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for rk in RULE_BASE_KEYS:
        for K in K_THRESHOLDS:
            res = rule_results[rk][f"K_{K}"]
            st = res["approximate_per_s1_candidate_distribution"]
            md_lines.append(
                f"| `{rk}` | {K} | **{res['pair_recall'] * 100:.2f}%** ({res['pair_hits']:,}/{total_resolved_pairs:,}) | **{res['entity_level_all_retrieved_rate'] * 100:.2f}%** | **{res['estimated_mean_candidates_per_s1_entity']:,.2f}** | {st['median']:.1f} | {st['p95']:.1f} | {st['p99']:.1f} | {st['max']:,} | {int(res['estimated_full_training_candidate_pairs_20k_sample']):,} |"
            )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print("=" * 70)
    print(f"[SUCCESS] Rarity-Aware Blocking Benchmark Completed in {total_elapsed:.2f}s.")
    print(f"[INFO] Report Wrote: {json_path.resolve()}")
    print(f"[INFO] Report Wrote: {md_path.resolve()}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())