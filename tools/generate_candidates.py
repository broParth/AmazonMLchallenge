#!/usr/bin/env python3
"""
generate_candidates.py

Generates candidate matching pairs using the C UNION E blocking rule:
  - Rule C: (country, first normalized name token)
  - Rule E: Any shared normalized address token

Memory-safe, disk-partitioned architecture designed for 8 GB RAM EC2:
  1. Indexes Source 1 using compact array.array('I') integer offsets.
  2. Streams Source 2 and Source 3 row-by-row, emitting raw (s1_idx, candidate_id)
     hits directly to range-partitioned temp files (zero per-row set allocations).
  3. Sorts each partition file by s1_idx and streams sorted hits line-by-line,
     deduplicating candidate IDs for 1 S1 entity at a time in RAM.
"""

from __future__ import annotations

import argparse
import array
import math
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Generator, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from common import (
        normalize_text,
        first_token,
        tokenize_address,
        stream_tsv_rows,
        check_free_disk_space,
        format_bytes,
    )
except ImportError as e:
    print(f"[FATAL] Failed to import common.py: {e}", file=sys.stderr)
    sys.exit(1)

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def get_rss_bytes() -> int:
    """Retrieves current Resident Set Size (RSS) memory in bytes."""
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


def build_source1_index(
    s1_path: Path,
    encoding: str,
    max_rows: Optional[int] = None
) -> Tuple[List[str], Dict[Tuple[str, str], array.array], Dict[str, array.array]]:
    """
    Streams Source 1 and builds Rule C and Rule E posting lists using array.array('I').
    Logs detailed RSS memory deltas at each stage.
    """
    print(f"[INFO] Indexing Source 1 from: {s1_path.resolve()}")
    if max_rows:
        print(f"[INFO] SMALL-SAMPLE MODE: Capping Source 1 to max {max_rows:,} rows.")

    t0 = time.time()
    rss_start = get_rss_bytes()

    s1_ids: List[str] = []
    index_C: Dict[Tuple[str, str], array.array] = defaultdict(lambda: array.array('I'))
    index_E: Dict[str, array.array] = defaultdict(lambda: array.array('I'))

    count = 0
    for row in stream_tsv_rows(s1_path, SOURCE_COLUMNS, encoding=encoding, max_rows=max_rows):
        s1_id = row["entity_id"]
        country = row["country"].strip()
        norm_name = normalize_text(row["business_name"])
        norm_addr = normalize_text(row["business_address"])

        s1_idx = len(s1_ids)
        s1_ids.append(s1_id)

        # Rule C Key
        fname_token = first_token(norm_name)
        if country and fname_token:
            index_C[(country, fname_token)].append(s1_idx)

        # Rule E Keys
        addr_tokens = tokenize_address(norm_addr)
        for token in addr_tokens:
            index_E[token].append(s1_idx)

        count += 1
        if count % 500_000 == 0:
            print(f"[INFO]   Indexed {count:,} Source-1 rows... (RSS: {format_rss()})")

    elapsed = time.time() - t0
    rss_end = get_rss_bytes()

    c_postings = sum(len(a) for a in index_C.values())
    e_postings = sum(len(a) for a in index_E.values())

    print(f"[INFO] Source-1 Indexing Complete in {elapsed:.2f}s.")
    print(f"[INFO]   - Total S1 Entities:     {len(s1_ids):,}")
    print(f"[INFO]   - Rule C Unique Keys:    {len(index_C):,}, Total Postings: {c_postings:,}")
    print(f"[INFO]   - Rule E Unique Tokens:  {len(index_E):,}, Total Postings: {e_postings:,}")
    print(f"[INFO]   - Start RSS Memory:      {format_bytes(rss_start)}")
    print(f"[INFO]   - Index Peak RSS Memory: {format_bytes(rss_end)} (Delta: +{format_bytes(rss_end - rss_start)})")

    return s1_ids, index_C, index_E


def stream_source_to_partitions(
    source_path: Path,
    source_label: str,
    index_C: Dict[Tuple[str, str], array.array],
    index_E: Dict[str, array.array],
    partition_files: List,
    partition_size: int,
    temp_dir: Path,
    max_temp_disk_bytes: int,
    encoding: str,
    max_rows: Optional[int] = None
) -> int:
    """
    Streams a noisy source row-by-row, queries indices, and writes raw hit lines
    directly to range partition files without building a per-row Set[int].
    """
    print(f"[INFO] Streaming {source_label} from: {source_path.resolve()}")
    if max_rows:
        print(f"[INFO] SMALL-SAMPLE MODE: Capping {source_label} to max {max_rows:,} rows.")

    t0 = time.time()
    row_count = 0
    raw_pair_hits = 0

    for row in stream_tsv_rows(source_path, SOURCE_COLUMNS, encoding=encoding, max_rows=max_rows):
        noisy_id = row["entity_id"]
        country = row["country"].strip()
        norm_name = normalize_text(row["business_name"])
        norm_addr = normalize_text(row["business_address"])

        # Direct Posting List Output for Rule C
        fname_token = first_token(norm_name)
        if country and fname_token:
            c_key = (country, fname_token)
            if c_key in index_C:
                for s1_idx in index_C[c_key]:
                    p_id = s1_idx // partition_size
                    partition_files[p_id].write(f"{s1_idx}\t{noisy_id}\n")
                    raw_pair_hits += 1

        # Direct Posting List Output for Rule E
        addr_tokens = tokenize_address(norm_addr)
        for token in addr_tokens:
            if token in index_E:
                for s1_idx in index_E[token]:
                    p_id = s1_idx // partition_size
                    partition_files[p_id].write(f"{s1_idx}\t{noisy_id}\n")
                    raw_pair_hits += 1

        row_count += 1

        if row_count % 500_000 == 0:
            # Periodically flush partition handles to update disk usage statistics
            for h in partition_files:
                h.flush()

            temp_disk_used = sum(f.stat().st_size for f in temp_dir.glob("partition_*.tmp"))
            print(
                f"[INFO]   Processed {row_count:,} {source_label} rows... "
                f"Raw Hits: {raw_pair_hits:,}, Temp Disk Used: {format_bytes(temp_disk_used)} (RSS: {format_rss()})"
            )

            if temp_disk_used > max_temp_disk_bytes:
                raise RuntimeError(
                    f"Temp disk budget exceeded ({format_bytes(temp_disk_used)} > "
                    f"Limit: {format_bytes(max_temp_disk_bytes)})."
                )
            check_free_disk_space(temp_dir, min_bytes=3_221_225_472)

    elapsed = time.time() - t0
    print(
        f"[INFO] {source_label} Stream Complete: {row_count:,} rows scanned in {elapsed:.2f}s. "
        f"Raw Candidate Hits: {raw_pair_hits:,}. (RSS: {format_rss()})"
    )
    return raw_pair_hits


def sort_partition_file(part_path: Path) -> Path:
    """
    Sorts partition_P.tmp by integer s1_idx using system 'sort' CLI if available,
    falling back to Python in-memory sort if missing.
    """
    if not part_path.exists() or part_path.stat().st_size == 0:
        return part_path

    sorted_path = part_path.with_suffix(".sorted.tmp")

    try:
        subprocess.run(
            ["sort", "-t", "\t", "-k1,1n", str(part_path), "-o", str(sorted_path)],
            check=True,
            capture_output=True,
            text=True
        )
        return sorted_path
    except (subprocess.SubprocessError, FileNotFoundError):
        with open(part_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        lines.sort(key=lambda l: int(l.split("\t")[0]))
        with open(sorted_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        return sorted_path


def iter_grouped_candidates(sorted_path: Path) -> Generator[Tuple[int, Set[str]], None, None]:
    """
    Streams sorted partition file line-by-line, yielding (s1_idx, Set[candidate_ids])
    for one S1 entity at a time.
    """
    if not sorted_path.exists():
        return

    with open(sorted_path, "r", encoding="utf-8") as f:
        current_s1_idx: Optional[int] = None
        current_cands: Set[str] = set()

        for line in f:
            line_str = line.rstrip("\r\n")
            if not line_str:
                continue
            parts = line_str.split("\t")
            if len(parts) != 2:
                continue
            s1_idx = int(parts[0])
            cand_id = parts[1]

            if current_s1_idx is None:
                current_s1_idx = s1_idx
                current_cands.add(cand_id)
            elif s1_idx == current_s1_idx:
                current_cands.add(cand_id)
            else:
                yield current_s1_idx, current_cands
                current_s1_idx = s1_idx
                current_cands = {cand_id}

        if current_s1_idx is not None:
            yield current_s1_idx, current_cands


def merge_partitions_and_write_output(
    output_path: Path,
    temp_dir: Path,
    s1_ids: List[str],
    num_partitions: int,
    partition_size: int,
) -> Tuple[int, int, int]:
    """
    Sequentially sorts partition files and streams candidate sets one S1 entity
    at a time, maintaining $O(1)$ memory overhead.
    """
    print(f"[INFO] Starting Range Partition Merge ({num_partitions} partitions) -> {output_path.resolve()}")
    t0 = time.time()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_entities = len(s1_ids)
    dedup_pairs_count = 0
    singletons_count = 0

    with open(output_path, "w", encoding="utf-8", newline="") as out_f:
        out_f.write("source1_entity_id\tcandidate_entity_ids\n")

        for p_id in range(num_partitions):
            part_path = temp_dir / f"partition_{p_id}.tmp"
            start_s1_idx = p_id * partition_size
            end_s1_idx = min((p_id + 1) * partition_size, total_entities)

            sorted_path = sort_partition_file(part_path)
            grouped_gen = iter_grouped_candidates(sorted_path)
            next_matched = next(grouped_gen, None)

            for target_s1_idx in range(start_s1_idx, end_s1_idx):
                s1_id = s1_ids[target_s1_idx]

                if next_matched is not None and next_matched[0] == target_s1_idx:
                    cands = sorted(next_matched[1])
                    out_f.write(f"{s1_id}\t{','.join(cands)}\n")
                    dedup_pairs_count += len(cands)
                    next_matched = next(grouped_gen, None)
                else:
                    out_f.write(f"{s1_id}\t\n")
                    singletons_count += 1

            # Clean temporary partition files immediately
            if part_path.exists():
                part_path.unlink()
            if sorted_path.exists() and sorted_path != part_path:
                sorted_path.unlink()

            print(
                f"[INFO]   Merged Partition {p_id + 1}/{num_partitions} "
                f"(S1 Index Range [{start_s1_idx:,}..{end_s1_idx - 1:,}]). RSS: {format_rss()}"
            )

    elapsed = time.time() - t0
    print(f"[INFO] Output Generation Complete in {elapsed:.2f}s.")
    print(f"[INFO]   - Total S1 Entities:               {total_entities:,}")
    print(f"[INFO]   - Singletons (0 candidates):       {singletons_count:,}")
    print(f"[INFO]   - Deduplicated Candidate Pairs:   {dedup_pairs_count:,}")

    return total_entities, dedup_pairs_count, singletons_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Memory-safe disk-partitioned candidate generator for Amazon ML Challenge 2026."
    )
    parser.add_argument("--s1-path", type=Path, required=True, help="Path to source1.tsv")
    parser.add_argument("--s2-path", type=Path, required=True, help="Path to source2.tsv")
    parser.add_argument("--s3-path", type=Path, required=True, help="Path to source3.tsv")
    parser.add_argument("--output-path", type=Path, required=True, help="Path for candidate_pairs.tsv output")
    parser.add_argument("--temp-dir", type=Path, default=Path("temp_candidates"), help="Temporary directory for partition files")
    parser.add_argument("--partition-size", type=int, default=100_000, help="S1 entities per range partition (default: 100,000)")
    parser.add_argument("--max-temp-disk-gb", type=float, default=15.0, help="Max temporary disk space limit in GB (default: 15.0)")
    parser.add_argument("--max-rows", type=int, default=None, help="Small-sample smoke-test mode: cap rows read per file")
    parser.add_argument("--encoding", type=str, default="utf-8", help="File encoding (default: utf-8)")

    args = parser.parse_args()
    start_time = time.time()
    max_temp_disk_bytes = int(args.max_temp_disk_gb * 1024 * 1024 * 1024)

    print("=" * 70)
    print(f"[INFO] Starting Candidate Generator (C UNION E Rule)")
    print(f"[INFO] Mode: {'SMALL-SAMPLE SMOKE TEST' if args.max_rows else 'FULL DATASET RUN'}")
    print(f"[INFO] Initial RSS RAM: {format_rss()}")
    print("=" * 70)

    # Disk Space Guard
    check_free_disk_space(args.output_path.parent, min_bytes=5_368_709_120)

    args.temp_dir.mkdir(parents=True, exist_ok=True)
    partition_file_handles = []

    try:
        # Step 1: Index Source 1
        s1_ids, index_C, index_E = build_source1_index(
            args.s1_path, encoding=args.encoding, max_rows=args.max_rows
        )

        num_s1 = len(s1_ids)
        if num_s1 == 0:
            raise ValueError("Source 1 produced 0 valid records.")

        num_partitions = math.ceil(num_s1 / args.partition_size)
        print(f"[INFO] Partition Config: {num_partitions} partitions ({args.partition_size:,} S1 entities/partition).")

        # Open 1 MB buffered write handles for all partitions
        for p_id in range(num_partitions):
            p_path = args.temp_dir / f"partition_{p_id}.tmp"
            handle = open(p_path, "w", encoding="utf-8", buffering=1_048_576)
            partition_file_handles.append(handle)

        # Step 2: Stream Source 2
        s2_hits = stream_source_to_partitions(
            args.s2_path, "Source 2", index_C, index_E,
            partition_file_handles, args.partition_size,
            args.temp_dir, max_temp_disk_bytes,
            encoding=args.encoding, max_rows=args.max_rows
        )

        # Step 3: Stream Source 3
        s3_hits = stream_source_to_partitions(
            args.s3_path, "Source 3", index_C, index_E,
            partition_file_handles, args.partition_size,
            args.temp_dir, max_temp_disk_bytes,
            encoding=args.encoding, max_rows=args.max_rows
        )

        # Flush & close partition file handles
        for handle in partition_file_handles:
            handle.flush()
            handle.close()
        partition_file_handles.clear()

        # Cumulative Disk Usage Report
        temp_disk_bytes = sum(f.stat().st_size for f in args.temp_dir.glob("partition_*.tmp"))
        print(f"[INFO] Total Temp Partition Disk Footprint: {format_bytes(temp_disk_bytes)}")

        # Step 4: Merge Partitions & Generate Output
        total_s1, dedup_pairs, singletons = merge_partitions_and_write_output(
            args.output_path, args.temp_dir, s1_ids, num_partitions, args.partition_size
        )

        total_time = time.time() - start_time
        print("=" * 70)
        print(f"[SUCCESS] Candidate Generation Finished in {total_time:.2f} seconds.")
        print(f"[SUMMARY] Total S1 Entities:        {total_s1:,}")
        print(f"[SUMMARY] Singletons (0 cands):     {singletons:,}")
        print(f"[SUMMARY] Raw S2 Hits:              {s2_hits:,}")
        print(f"[SUMMARY] Raw S3 Hits:              {s3_hits:,}")
        print(f"[SUMMARY] Deduplicated Candidate Pairs: {dedup_pairs:,}")
        print(f"[SUMMARY] Output File:              {args.output_path.resolve()}")
        print("=" * 70)

        return 0

    finally:
        # Guarantee cleanup of all partition handles and temporary files
        for handle in partition_file_handles:
            try:
                handle.close()
            except Exception:
                pass
        if args.temp_dir.exists():
            shutil.rmtree(args.temp_dir, ignore_errors=True)
            print(f"[INFO] Temporary directory cleaned: {args.temp_dir.resolve()}")


if __name__ == "__main__":
    sys.exit(main())