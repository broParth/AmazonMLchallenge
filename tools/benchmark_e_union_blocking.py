#!/usr/bin/env python3
"""
benchmark_e_union_blocking.py

Empirical benchmark evaluating E-based union candidate generation rules on a
deterministic sample of 20,000 Source-1 training entities.

RULES BENCHMARKED:
  1. E:                      any shared normalized address token
  2. C_union_E:              C UNION E
  3. G_union_E:              G UNION E
  4. A_G_union_C_union_E:    A UNION G UNION C UNION E

MEMORY GUARANTEES:
  - Inverted indices constructed ONLY for the 20,000 sampled S1 entities.
  - Zero storage of arbitrary candidate string IDs in RAM.
  - Candidate counts tracked via simple integer arrays.
  - Ground-truth hits tracked only against known GT IDs for each sampled S1 entity.
  - Active process RSS monitoring with a hard safety ceiling (--max-ram-gb).
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


def distribution_summary_from_counter(freq_counter: Counter) -> dict:
    """Computes distribution stats from a Counter of integer values with clean rounding."""
    total_count = sum(freq_counter.values())
    if total_count == 0:
        return {"count": 0, "min": 0, "median": 0.0, "mean": 0.0, "p95": 0.0, "p99": 0.0, "max": 0}

    sorted_distinct = sorted(freq_counter.items(), key=lambda x: x[0])
    min_val = sorted_distinct[0][0]
    max_val = sorted_distinct[-1][0]
    total_sum = sum(val * cnt for val, cnt in sorted_distinct)
    mean_val = round(total_sum / total_count, 2)

    def get_pct_val(pct: float) -> float:
        target_k = (total_count - 1) * (pct / 100.0)
        cum = 0
        for val, cnt in sorted_distinct:
            cum += cnt
            if cum - 1 >= target_k:
                return round(float(val), 2)
        return round(float(max_val), 2)

    return {
        "count": total_count,
        "min": min_val,
        "median": get_pct_val(50.0),
        "mean": mean_val,
        "p95": get_pct_val(95.0),
        "p99": get_pct_val(99.0),
        "max": max_val,
    }


def distribution_summary_list(values: list) -> dict:
    """Computes distribution stats from a list of numbers with clean rounding."""
    if not values:
        return {
            "count": 0,
            "min": 0,
            "median": 0.0,
            "mean": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0,
            "total_candidate_pairs": 0,
        }
    values_sorted = sorted(values)
    n = len(values_sorted)
    raw_mean = sum(values_sorted) / n
    return {
        "count": n,
        "min": values_sorted[0],
        "median": round(float(percentile(values_sorted, 50)), 2),
        "mean": round(float(raw_mean), 2),
        "p95": round(float(percentile(values_sorted, 95)), 2),
        "p99": round(float(percentile(values_sorted, 99)), 2),
        "max": values_sorted[-1],
        "total_candidate_pairs": sum(values_sorted),
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
    parser = argparse.ArgumentParser(description="E-Union blocking experiment benchmark.")
    parser.add_argument("--root", type=Path, default=None, help="Project root path")
    parser.add_argument("--encoding", type=str, default="utf-8")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--max-ram-gb", type=float, default=4.0)

    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    root = args.root if args.root else script_dir.parent
    max_ram_bytes = int(args.max_ram_gb * 1024 * 1024 * 1024)

    start_time = time.time()
    rss_baseline = get_rss_bytes()

    print("=" * 70)
    print(f"[INFO] E-Union Blocking Benchmark v{SCRIPT_VERSION}")
    print(f"[INFO] Seed: {args.seed}, Sample Size: {args.sample_size:,}, Max RAM: {args.max_ram_gb} GB")
    print(f"[INFO] Baseline Process RSS: {format_bytes(rss_baseline)}")
    print("=" * 70)

    # 1. Sample Ground Truth
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

    print(f"[INFO]   - Sampled S1 Entities: {len(sample_set):,} ({len(singletons):,} singletons, {len(non_singletons):,} non-singletons)")
    print(f"[INFO]   - Ground Truth Match Targets: {len(all_referenced_true_matches):,} unique S2/S3 IDs")

    # 2. Load S1 records for sample & build sample index
    s1_path = root / "dataset" / "train" / "train_source1.tsv"
    if not s1_path.exists():
        s1_path = root / "train_source1.tsv"

    print(f"[INFO] Pass 2: Loading S1 records and building sample indices from {s1_path.resolve()}...")

    s1_id_to_idx = {sid: i for i, sid in enumerate(s1_sample_ids)}
    s1_gt_sets = [gt_matches[sid] for sid in s1_sample_ids]

    s1_records_A: Dict[str, List[int]] = defaultdict(list)
    s1_records_G: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    s1_records_C: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    s1_records_E: Dict[str, List[int]] = defaultdict(list)

    app_A = [False] * len(s1_sample_ids)
    app_G = [False] * len(s1_sample_ids)
    app_C = [False] * len(s1_sample_ids)
    app_E = [False] * len(s1_sample_ids)

    loaded_s1 = 0
    for row in stream_tsv_rows(s1_path, SOURCE_COLUMNS, encoding=args.encoding):
        sid = row["entity_id"]
        if sid in sample_set:
            s1_idx = s1_id_to_idx[sid]
            country = row["country"].strip()
            norm_name = normalize_text(row["business_name"])
            norm_addr = normalize_text(row["business_address"])

            # Rule A
            if norm_name:
                s1_records_A[norm_name].append(s1_idx)
                app_A[s1_idx] = True

            # Rule C
            ftok = first_token(norm_name)
            if country and ftok:
                s1_records_C[(country, ftok)].append(s1_idx)
                app_C[s1_idx] = True

            # Rule G
            first_two = extract_first_two_tokens(norm_name)
            if country and first_two:
                s1_records_G[(country, first_two)].append(s1_idx)
                app_G[s1_idx] = True

            # Rule E
            addr_tokens = tokenize_address(norm_addr)
            if addr_tokens:
                for token in addr_tokens:
                    s1_records_E[token].append(s1_idx)
                app_E[s1_idx] = True

            loaded_s1 += 1

    rss_after_s1 = get_rss_bytes()
    print(f"[INFO]   - Loaded {loaded_s1:,} S1 sample records. RSS: {format_bytes(rss_after_s1)}")

    # 3. Initialize lightweight candidate counters & GT hit tracking
    candidate_counts: Dict[str, List[int]] = {rk: [0] * len(s1_sample_ids) for rk in RULE_KEYS}
    gt_retrieved: Dict[str, List[Set[str]]] = {rk: [set() for _ in range(len(s1_sample_ids))] for rk in RULE_KEYS}

    hits_E_size_counter = Counter()
    referenced_records = {}

    # 4. Stream Source 2 and Source 3
    s2_path = root / "dataset" / "train" / "train_source2.tsv"
    if not s2_path.exists():
        s2_path = root / "train_source2.tsv"
    s3_path = root / "dataset" / "train" / "train_source3.tsv"
    if not s3_path.exists():
        s3_path = root / "train_source3.tsv"

    s2_rows_scanned = 0
    s3_rows_scanned = 0

    for source_label, src_path in [("Source 2", s2_path), ("Source 3", s3_path)]:
        print(f"[INFO] Pass 3: Streaming {source_label} from {src_path.resolve()}...")
        t_src_start = time.time()
        src_rows = 0

        for row in stream_tsv_rows(src_path, SOURCE_COLUMNS, encoding=args.encoding):
            noisy_id = row["entity_id"]
            country = row["country"].strip()
            norm_name = normalize_text(row["business_name"])
            norm_addr = normalize_text(row["business_address"])

            if noisy_id in all_referenced_true_matches:
                referenced_records[noisy_id] = True

            ftok = first_token(norm_name)
            first_two = extract_first_two_tokens(norm_name)
            tokens_addr = tokenize_address(norm_addr)

            # Atomic hits against S1 sample
            hits_A = set(s1_records_A.get(norm_name, [])) if norm_name else set()
            hits_G = set(s1_records_G.get((country, first_two), [])) if (country and first_two) else set()
            hits_C = set(s1_records_C.get((country, ftok), [])) if (country and ftok) else set()

            hits_E = set()
            if tokens_addr:
                for token in tokens_addr:
                    if token in s1_records_E:
                        hits_E.update(s1_records_E[token])

            # Profile |hits_E| distribution
            hits_E_size_counter[len(hits_E)] += 1

            # Construct exact union sets per noisy record
            hits_C_union_E = hits_C | hits_E
            hits_G_union_E = hits_G | hits_E
            hits_A_G_union_C_union_E = hits_A | hits_G | hits_C | hits_E

            rule_hits = {
                "E": hits_E,
                "C_union_E": hits_C_union_E,
                "G_union_E": hits_G_union_E,
                "A_G_union_C_union_E": hits_A_G_union_C_union_E,
            }

            for rk in RULE_KEYS:
                matched_indices = rule_hits[rk]
                if matched_indices:
                    for s1_idx in matched_indices:
                        candidate_counts[rk][s1_idx] += 1
                        if noisy_id in s1_gt_sets[s1_idx]:
                            gt_retrieved[rk][s1_idx].add(noisy_id)

            src_rows += 1
            if source_label == "Source 2":
                s2_rows_scanned += 1
            else:
                s3_rows_scanned += 1

            if src_rows % 1_000_000 == 0:
                cur_rss = get_rss_bytes()
                print(f"[INFO]   Scanned {src_rows:,} {source_label} rows... (RSS: {format_bytes(cur_rss)})")
                if cur_rss > max_ram_bytes:
                    raise MemoryError(f"Process RSS ({format_bytes(cur_rss)}) exceeded safety ceiling ({format_bytes(max_ram_bytes)}).")

        t_src_elapsed = time.time() - t_src_start
        print(f"[INFO] {source_label} Complete: {src_rows:,} rows in {t_src_elapsed:.2f}s ({src_rows / t_src_elapsed:,.0f} rows/s). RSS: {format_rss()}")

    # 5. Resolve Ground Truth Pairs
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
    print(f"[INFO] Ground Truth Pair Resolution: {total_resolved_pairs:,} resolved pairs used for recall denominator ({unresolved_pairs:,} unresolved).")

    # 6. Evaluate Candidate Counts & Recall Per Rule
    rule_results = {}
    for rk in RULE_KEYS:
        if rk == "E":
            applicable_mask = app_E
        elif rk == "C_union_E":
            applicable_mask = [c or e for c, e in zip(app_C, app_E)]
        elif rk == "G_union_E":
            applicable_mask = [g or e for g, e in zip(app_G, app_E)]
        elif rk == "A_G_union_C_union_E":
            applicable_mask = [a or g or c or e for a, g, c, e in zip(app_A, app_G, app_C, app_E)]
        else:
            applicable_mask = [True] * len(s1_sample_ids)

        applicable_indices = [i for i, is_app in enumerate(applicable_mask) if is_app]
        cand_counts = [candidate_counts[rk][i] for i in applicable_indices]

        cand_stats = distribution_summary_list(cand_counts)

        # Recall Evaluation
        hits = sum(len(gt_retrieved[rk][s1_idx]) for s1_idx in range(len(s1_sample_ids)))
        pair_recall = hits / total_resolved_pairs if total_resolved_pairs > 0 else 0.0

        # Entity-level all-matches retrieved rate (non-singletons)
        all_retrieved_entities = 0
        non_singleton_considered = 0
        for s1_id in non_singletons:
            s1_idx = s1_id_to_idx[s1_id]
            true_set = gt_matches[s1_id]
            resolved_true_set = {m for m in true_set if m in referenced_records}
            if not resolved_true_set:
                continue
            non_singleton_considered += 1
            if resolved_true_set.issubset(gt_retrieved[rk][s1_idx]):
                all_retrieved_entities += 1

        entity_recall = all_retrieved_entities / non_singleton_considered if non_singleton_considered > 0 else 0.0

        rule_results[rk] = {
            "label": RULE_LABELS[rk],
            "applicable_s1_entities": len(applicable_indices),
            "candidate_count_stats": cand_stats,
            "pair_hits": hits,
            "pair_total_resolved": total_resolved_pairs,
            "pair_recall": round(pair_recall, 6),
            "entities_all_retrieved": all_retrieved_entities,
            "entities_considered": non_singleton_considered,
            "entity_level_all_retrieved_rate": round(entity_recall, 6),
        }

    # 7. Assemble & Write Report
    total_elapsed = time.time() - start_time
    hits_E_stats = distribution_summary_from_counter(hits_E_size_counter)

    report = {
        "benchmark_metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "script": "tools/benchmark_e_union_blocking.py",
            "script_version": SCRIPT_VERSION,
            "seed": args.seed,
            "requested_sample_size": args.sample_size,
            "actual_sample_size": len(s1_sample_ids),
            "gt_total_rows": total_gt_rows,
            "s2_rows_scanned": s2_rows_scanned,
            "s3_rows_scanned": s3_rows_scanned,
            "total_noisy_rows_scanned": s2_rows_scanned + s3_rows_scanned,
            "resolved_gt_pairs": total_resolved_pairs,
            "unresolved_gt_pairs": unresolved_pairs,
            "peak_rss_ram": format_rss(),
            "total_elapsed_seconds": round(total_elapsed, 2),
        },
        "hits_E_distribution_profile": hits_E_stats,
        "e_union_rules": rule_results,
    }

    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "benchmark_e_union_blocking.json"
    md_path = reports_dir / "benchmark_e_union_blocking.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_lines = [
        "# E-Union Blocking Benchmark Report",
        "",
        f"- Generated (UTC): {report['benchmark_metadata']['generated_at_utc']}",
        f"- Seed: {args.seed}, Sample S1 Entities: {len(s1_sample_ids):,}",
        f"- S2 Rows Scanned: {s2_rows_scanned:,}, S3 Rows Scanned: {s3_rows_scanned:,}",
        f"- Resolved GT Pairs: {total_resolved_pairs:,} (Unresolved: {unresolved_pairs:,})",
        f"- Peak Process RSS Memory: {format_rss()}",
        f"- Total Elapsed Time: {total_elapsed:.2f}s",
        "",
        "## Rule E Lookup Size Profile (|hits_E| across streamed noisy records)",
        f"- Total Rule E Lookups: {hits_E_stats['count']:,}",
        f"- Min: {hits_E_stats['min']}, Median: {hits_E_stats['median']:.1f}, Mean: {hits_E_stats['mean']:.2f}, P95: {hits_E_stats['p95']:.1f}, P99: {hits_E_stats['p99']:.1f}, Max: {hits_E_stats['max']:,}",
        "",
        "## E-Union Blocking Rules Performance",
        "",
        "| Rule Key | Label | Applicable S1 | Pair Recall (Hits / Resolved GT) | Entity All-Retrieved Rate | Mean Cands | Median Cands | P95 Cands | P99 Cands | Max Cands | Total Candidates |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for rk in RULE_KEYS:
        res = rule_results[rk]
        st = res["candidate_count_stats"]
        md_lines.append(
            f"| `{rk}` | {res['label']} | {res['applicable_s1_entities']:,} | **{res['pair_recall'] * 100:.2f}%** ({res['pair_hits']:,}/{total_resolved_pairs:,}) | **{res['entity_level_all_retrieved_rate'] * 100:.2f}%** | {st['mean']:.2f} | {st['median']:.1f} | {st['p95']:.1f} | {st['p99']:.1f} | {st['max']:,} | {st['total_candidate_pairs']:,} |"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print("=" * 70)
    print(f"[SUCCESS] E-Union Blocking Benchmark Completed in {total_elapsed:.2f}s.")
    print(f"[INFO] Report Wrote: {json_path.resolve()}")
    print(f"[INFO] Report Wrote: {md_path.resolve()}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())