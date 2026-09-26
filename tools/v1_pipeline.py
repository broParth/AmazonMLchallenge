#!/usr/bin/env python3
"""
v1_pipeline.py

Memory-safe, bounded streaming pipeline engine for Amazon ML Challenge V1 Entity Resolution.

ARCHITECTURAL MEMORY & PERFORMANCE GUARANTEES:
  1. Compact S1 storage: Flat string lists + array('I') inverted indexes (~300 MB RAM for 2.2M S1).
  2. Bounded candidate generation: Exact Rule A hits retained; Rule G + E_rare@200 capped to top-K=15.
  3. Scalar blocker recall accounting: Uses scalar counter (O(1) RAM) to track retrieved GT pairs.
  4. Bounded memmap training matrix: Uses np.memmap with bounded random sampling (500k max samples)
     and balanced class allocation.
  5. Disk-backed streaming validation: Validation candidate predictions streamed directly to val_scored.tmp,
     sorted on disk, and evaluated in streaming S1 groups with zero global prediction dict in RAM.
  6. Batched test inference: Test candidate features batched (50,000 pairs/batch) for predict_proba(),
     partitioned across P=16 disk files, sorted on disk, and assembled sequentially per s1_idx.
  7. Designed to remain well within the 8 GiB EC2 memory budget.
"""

from __future__ import annotations

import array
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    check_free_disk_space,
    first_token,
    format_bytes,
    normalize_text,
    stream_tsv_rows,
    tokenize_address,
)
from evaluate_f05 import compute_per_s1_f05
from v1_features import (
    FEATURE_NAMES,
    extract_v1_features,
    fast_heuristic_candidate_score,
)

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
NUM_PARTITIONS = 16


def extract_first_two_tokens(s_normalized: str) -> Optional[str]:
    parts = [t for t in s_normalized.split(" ") if t]
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"
    return None


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


class S1CompactIndex:
    """Memory-efficient flat string storage and inverted index over Source 1 entities."""

    def __init__(self):
        self.s1_ids: List[str] = []
        self.s1_names: List[str] = []
        self.s1_addrs: List[str] = []
        self.s1_countries: List[str] = []
        self.s1_norm_names: List[str] = []
        self.s1_norm_addrs: List[str] = []

        # Inverted indexes -> compact integer arrays
        self.index_A: Dict[str, array.array] = defaultdict(lambda: array.array("I"))
        self.index_G: Dict[Tuple[str, str], array.array] = defaultdict(lambda: array.array("I"))
        self.index_E_rare: Dict[str, array.array] = defaultdict(lambda: array.array("I"))


def build_s1_compact_index(
    s1_path: Path,
    encoding: str = "utf-8",
    max_rows: Optional[int] = None,
    max_doc_freq: int = 200,
) -> Tuple[S1CompactIndex, Counter]:
    """
    Streams Source 1 to:
      1. Compute address token document frequencies over full S1.
      2. Build compact S1 inverted index (flat arrays, zero dict-of-row overhead).
    """
    print(f"[INFO] Phase 1: Computing address document frequencies from {s1_path.resolve()}...")
    t0 = time.time()

    doc_freq_addr: Counter = Counter()

    # Pass 1: Address Token Document Frequency
    s1_count = 0
    for row in stream_tsv_rows(s1_path, SOURCE_COLUMNS, encoding=encoding, max_rows=max_rows):
        s1_count += 1
        norm_addr = normalize_text(row["business_address"])
        addr_tokens = tokenize_address(norm_addr)
        for token in addr_tokens:
            doc_freq_addr[token] += 1

        if s1_count % 1_000_000 == 0:
            print(f"[INFO]   Scanned {s1_count:,} S1 rows for document frequency... (RSS: {format_rss()})")

    print(
        f"[INFO] Document Frequency Pass Complete: {s1_count:,} S1 rows scanned in {time.time() - t0:.2f}s. "
        f"Unique address tokens: {len(doc_freq_addr):,}."
    )

    # Pass 2: Index Construction
    print(f"[INFO] Phase 2: Building compact S1 index...")
    t1 = time.time()
    s1_idx_obj = S1CompactIndex()

    for row in stream_tsv_rows(s1_path, SOURCE_COLUMNS, encoding=encoding, max_rows=max_rows):
        s1_id = row["entity_id"]
        raw_name = row["business_name"].strip()
        raw_addr = row["business_address"].strip()
        country = row["country"].strip()

        norm_name = normalize_text(raw_name)
        norm_addr = normalize_text(raw_addr)

        idx = len(s1_idx_obj.s1_ids)
        s1_idx_obj.s1_ids.append(s1_id)
        s1_idx_obj.s1_names.append(raw_name)
        s1_idx_obj.s1_addrs.append(raw_addr)
        s1_idx_obj.s1_countries.append(country)
        s1_idx_obj.s1_norm_names.append(norm_name)
        s1_idx_obj.s1_norm_addrs.append(norm_addr)

        # Rule A Key
        if norm_name:
            s1_idx_obj.index_A[norm_name].append(idx)

        # Rule G Key
        first_two = extract_first_two_tokens(norm_name)
        if country and first_two:
            s1_idx_obj.index_G[(country, first_two)].append(idx)

        # Rule E_rare@200 Key
        addr_tokens = tokenize_address(norm_addr)
        for token in addr_tokens:
            if doc_freq_addr.get(token, 0) <= max_doc_freq:
                s1_idx_obj.index_E_rare[token].append(idx)

    print(
        f"[INFO] Compact S1 Index Built in {time.time() - t1:.2f}s. "
        f"Total S1 entities indexed: {len(s1_idx_obj.s1_ids):,}. RSS: {format_rss()}"
    )

    return s1_idx_obj, doc_freq_addr


def extract_candidates_for_row(
    row: Dict[str, str],
    s1_idx_obj: S1CompactIndex,
    doc_freq_addr: Counter,
    top_k_broad: int = 15,
    max_doc_freq: int = 200,
) -> Set[int]:
    """
    Generates bounded candidate S1 integer indices for a single noisy S2/S3 row.
    Exact Rule A hits are ALWAYS retained.
    Broad hits (G, E_rare@200) are scored immediately via fast heuristic and capped to top_k_broad.
    """
    country = row["country"].strip()
    norm_name = normalize_text(row["business_name"])
    norm_addr = normalize_text(row["business_address"])

    first_two = extract_first_two_tokens(norm_name)
    addr_tokens = tokenize_address(norm_addr)

    # Rule A hits (Exact Name - ALWAYS RETAINED)
    exact_hits = set(s1_idx_obj.index_A.get(norm_name, [])) if norm_name else set()

    # Broad Hits (Rule G + Rule E_rare@200)
    broad_hits = set()
    if country and first_two:
        if (country, first_two) in s1_idx_obj.index_G:
            broad_hits.update(s1_idx_obj.index_G[(country, first_two)])

    if addr_tokens:
        for token in addr_tokens:
            if doc_freq_addr.get(token, 0) <= max_doc_freq and token in s1_idx_obj.index_E_rare:
                broad_hits.update(s1_idx_obj.index_E_rare[token])

    # Remove exact hits from broad hits to avoid duplicate scoring
    broad_hits -= exact_hits

    # Score and bound broad hits to top_k_broad
    selected_broad_hits = set()
    if broad_hits:
        if len(broad_hits) <= top_k_broad:
            selected_broad_hits = broad_hits
        else:
            scored = []
            for s1_idx in broad_hits:
                sc = fast_heuristic_candidate_score(
                    s1_idx_obj.s1_norm_names[s1_idx],
                    s1_idx_obj.s1_norm_addrs[s1_idx],
                    s1_idx_obj.s1_countries[s1_idx],
                    norm_name,
                    norm_addr,
                    country,
                    is_exact_name_hit=False,
                )
                scored.append((sc, s1_idx))
            scored.sort(key=lambda x: x[0], reverse=True)
            selected_broad_hits = {s1_idx for _score, s1_idx in scored[:top_k_broad]}

    return exact_hits | selected_broad_hits


def stream_training_data_to_disk(
    s1_idx_obj: S1CompactIndex,
    doc_freq_addr: Counter,
    gt_matches: Dict[str, Set[str]],
    source_paths: List[Path],
    work_dir: Path,
    seed: int = 1337,
    top_k_broad: int = 15,
    negatives_per_s1: int = 15,
    encoding: str = "utf-8",
    max_rows: Optional[int] = None,
) -> Tuple[Path, Path, Path, List[int], List[int], float]:
    """
    Streams Source 2 and Source 3 row-by-row to extract features directly to disk:
      - Assigns 80/20 train/validation split per S1 entity deterministically.
      - Writes train features and labels to binary float files (`train_feats.bin`, `train_labels.bin`).
      - Writes validation candidate pairs to TSV (`val_candidates.tsv`).
      - Uses scalar counter for zero-RAM blocker recall accounting.
    """
    print(f"[INFO] Phase 3: Streaming S2/S3 features directly to disk (K_broad={top_k_broad}, K_neg={negatives_per_s1})...")
    t0 = time.time()

    work_dir.mkdir(parents=True, exist_ok=True)
    train_feats_path = work_dir / "train_feats.bin"
    train_labels_path = work_dir / "train_labels.bin"
    val_cand_path = work_dir / "val_candidates.tsv"

    # Deterministic 80/20 Grouped Train/Val Split
    rng = random.Random(seed)
    s1_is_val = [False] * len(s1_idx_obj.s1_ids)
    train_s1_indices = []
    val_s1_indices = []

    for idx, s1_id in enumerate(s1_idx_obj.s1_ids):
        if rng.random() >= 0.80:
            s1_is_val[idx] = True
            val_s1_indices.append(idx)
        else:
            train_s1_indices.append(idx)

    # Track sampled negatives per S1 entity during streaming
    neg_counts_per_s1 = [0] * len(s1_idx_obj.s1_ids)

    # Memory-bounded scalar recall counter
    retrieved_gt_count = 0
    total_gt_pairs_in_corpus = sum(len(gt_matches.get(s1_id, set())) for s1_id in s1_idx_obj.s1_ids)

    train_feats_f = open(train_feats_path, "wb")
    train_labels_f = open(train_labels_path, "wb")
    val_cand_f = open(val_cand_path, "w", encoding="utf-8", newline="")
    val_cand_f.write("s1_idx\ts1_id\tnoisy_id\tis_gt\tfeatures\n")

    total_noisy_rows = 0
    train_pos_written = 0
    train_neg_written = 0
    val_pairs_written = 0

    for src_path in source_paths:
        print(f"[INFO]   Streaming features from {src_path.resolve()}...")
        for row in stream_tsv_rows(src_path, SOURCE_COLUMNS, encoding=encoding, max_rows=max_rows):
            total_noisy_rows += 1
            noisy_id = row["entity_id"]

            candidate_s1_indices = extract_candidates_for_row(
                row, s1_idx_obj, doc_freq_addr, top_k_broad=top_k_broad
            )

            if not candidate_s1_indices:
                continue

            for s1_idx in candidate_s1_indices:
                s1_id = s1_idx_obj.s1_ids[s1_idx]
                s1_row = {
                    "entity_id": s1_id,
                    "business_name": s1_idx_obj.s1_names[s1_idx],
                    "business_address": s1_idx_obj.s1_addrs[s1_idx],
                    "country": s1_idx_obj.s1_countries[s1_idx],
                }

                feats = extract_v1_features(s1_row, row)
                is_gt = 1 if noisy_id in gt_matches.get(s1_id, set()) else 0

                if is_gt == 1:
                    retrieved_gt_count += 1

                if s1_is_val[s1_idx]:
                    # Validation Split Record
                    feat_str = ",".join(f"{f:.6f}" for f in feats)
                    val_cand_f.write(f"{s1_idx}\t{s1_id}\t{noisy_id}\t{is_gt}\t{feat_str}\n")
                    val_pairs_written += 1
                else:
                    # Training Split Record
                    if is_gt == 1:
                        # Retain all positive ground-truth pairs
                        train_feats_f.write(array.array("f", feats).tobytes())
                        train_labels_f.write(array.array("b", [1]).tobytes())
                        train_pos_written += 1
                    else:
                        # Sample hard negative pairs up to negatives_per_s1 limit
                        if neg_counts_per_s1[s1_idx] < negatives_per_s1:
                            train_feats_f.write(array.array("f", feats).tobytes())
                            train_labels_f.write(array.array("b", [0]).tobytes())
                            neg_counts_per_s1[s1_idx] += 1
                            train_neg_written += 1

            if total_noisy_rows % 1_000_000 == 0:
                print(
                    f"[INFO]     Processed {total_noisy_rows:,} noisy rows... "
                    f"Train Pos: {train_pos_written:,}, Train Neg: {train_neg_written:,}, Val Pairs: {val_pairs_written:,}. "
                    f"RSS: {format_rss()}"
                )

    train_feats_f.close()
    train_labels_f.close()
    val_cand_f.close()

    bounded_blocker_recall = (retrieved_gt_count / total_gt_pairs_in_corpus) if total_gt_pairs_in_corpus > 0 else 0.0

    print("=" * 70)
    print(f"[INFO] Bounded Blocker Recall Accounting:")
    print(f"[INFO]   Total Ground Truth Pairs in S1 Corpus:      {total_gt_pairs_in_corpus:,}")
    print(f"[INFO]   Retrieved GT Pairs (Post Top-K Bounding):   {retrieved_gt_count:,}")
    print(f"[INFO]   Observed Bounded Blocker Recall:            {bounded_blocker_recall * 100:.2f}%")
    print(f"[NOTE] Note: The 83.89% pair recall benchmark applies to UNBOUNDED A U G U E_rare@200.")
    print(f"[NOTE] Broad candidate bounding (top-K={top_k_broad}) caps candidates per noisy row to bound RAM,")
    print(f"[NOTE] which may slightly adjust total pair recall on the full corpus.")
    print("=" * 70)

    print(
        f"[INFO] Streaming Feature Pass Finished in {time.time() - t0:.2f}s. "
        f"Total noisy rows: {total_noisy_rows:,}. "
        f"Train Positives: {train_pos_written:,}, Train Negatives: {train_neg_written:,}, "
        f"Val Pairs: {val_pairs_written:,}. RSS: {format_rss()}"
    )

    return train_feats_path, train_labels_path, val_cand_path, train_s1_indices, val_s1_indices, bounded_blocker_recall


def evaluate_single_s1_entity(pred_set: Set[str], gt_set: Set[str]) -> Tuple[float, float, float, Optional[bool]]:
    """Evaluates precision, recall, F0.5, and singleton accuracy for a single S1 entity."""
    if len(gt_set) == 0:
        # Ground truth is singleton
        if len(pred_set) == 0:
            return 1.0, 1.0, 1.0, True
        else:
            return 0.0, 1.0, 0.0, False
    else:
        # Ground truth non-singleton
        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        denom = 0.25 * prec + rec
        f05 = (1.25 * prec * rec / denom) if denom > 0 else 0.0
        return prec, rec, f05, None


def train_and_validate_v1_model(
    train_feats_path: Path,
    train_labels_path: Path,
    val_cand_path: Path,
    gt_matches: Dict[str, Set[str]],
    val_s1_ids: List[str],
    work_dir: Path,
    seed: int = 1337,
    max_train_samples: int = 500_000,
) -> Tuple[object, float, Dict[str, float]]:
    """
    Loads binary training feature arrays from disk via np.memmap with bounded sampling,
    fits a Gradient Boosting classifier, and performs disk-backed streaming validation evaluation.
    """
    print(f"[INFO] Phase 4: Training Matcher Model from disk-backed feature binaries...")

    num_bytes = train_labels_path.stat().st_size
    if num_bytes == 0:
        raise ValueError(f"No training samples found in {train_labels_path.resolve()}")

    num_samples = num_bytes  # int8 labels, 1 byte per sample
    num_features = len(FEATURE_NAMES)

    X_mmap = np.memmap(train_feats_path, dtype=np.float32, mode="r", shape=(num_samples, num_features))
    y_mmap = np.memmap(train_labels_path, dtype=np.int8, mode="r", shape=(num_samples,))

    pos_indices = np.where(y_mmap == 1)[0]
    neg_indices = np.where(y_mmap == 0)[0]

    n_pos_total = len(pos_indices)
    n_neg_total = len(neg_indices)

    rng = np.random.default_rng(seed)

    if num_samples > max_train_samples:
        target_pos = max_train_samples // 2
        target_neg = max_train_samples - target_pos

        # Allocate unused capacity to the other class if one class has fewer samples than target
        if n_pos_total < target_pos:
            actual_pos_count = n_pos_total
            actual_neg_count = min(n_neg_total, max_train_samples - actual_pos_count)
        elif n_neg_total < target_neg:
            actual_neg_count = n_neg_total
            actual_pos_count = min(n_pos_total, max_train_samples - actual_neg_count)
        else:
            actual_pos_count = target_pos
            actual_neg_count = target_neg

        if n_pos_total > actual_pos_count:
            sampled_pos = rng.choice(pos_indices, size=actual_pos_count, replace=False)
        else:
            sampled_pos = pos_indices

        if n_neg_total > actual_neg_count:
            sampled_neg = rng.choice(neg_indices, size=actual_neg_count, replace=False)
        else:
            sampled_neg = neg_indices

        selected_indices = np.concatenate([sampled_pos, sampled_neg])
        rng.shuffle(selected_indices)
    else:
        selected_indices = np.arange(num_samples)

    X_train = np.array(X_mmap[selected_indices], dtype=np.float32)
    y_train = np.array(y_mmap[selected_indices], dtype=np.int8)

    sel_pos_count = int(np.sum(y_train == 1))
    sel_neg_count = int(np.sum(y_train == 0))

    print(f"[INFO] Training Sample Allocation:")
    print(f"[INFO]   Total Positives:    {n_pos_total:,}")
    print(f"[INFO]   Total Negatives:    {n_neg_total:,}")
    print(f"[INFO]   Selected Positives: {sel_pos_count:,}")
    print(f"[INFO]   Selected Negatives: {sel_neg_count:,}")
    print(f"[INFO]   Final Sample Size:  {len(selected_indices):,}")

    t_train = time.time()
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(max_iter=100, random_state=seed, learning_rate=0.1)
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(n_estimators=100, random_state=seed, learning_rate=0.1)

    clf.fit(X_train, y_train)
    print(f"[INFO] Model Training Finished in {time.time() - t_train:.2f}s.")

    # Validation Phase: Stream val_candidates.tsv and write predictions to val_scored.tmp
    print(f"[INFO] Phase 5: Generating scored validation predictions to disk...")
    val_scored_path = work_dir / "val_scored.tmp"

    batch_x = []
    batch_meta = []
    batch_size = 50_000

    with open(val_cand_path, "r", encoding="utf-8") as f_in, open(val_scored_path, "w", encoding="utf-8", newline="") as f_out:
        reader = csv.reader(f_in, delimiter="\t")
        next(reader, None)  # Skip header

        for fields in reader:
            if not fields or len(fields) < 5:
                continue
            s1_id = fields[1]
            noisy_id = fields[2]
            feats = [float(x) for x in fields[4].split(",")]

            batch_x.append(feats)
            batch_meta.append((s1_id, noisy_id))

            if len(batch_x) >= batch_size:
                probs = clf.predict_proba(batch_x)[:, 1]
                for (s_id, n_id), p in zip(batch_meta, probs):
                    f_out.write(f"{s_id}\t{n_id}\t{p:.6f}\n")
                batch_x.clear()
                batch_meta.clear()

        if batch_x:
            probs = clf.predict_proba(batch_x)[:, 1]
            for (s_id, n_id), p in zip(batch_meta, probs):
                f_out.write(f"{s_id}\t{n_id}\t{p:.6f}\n")

    # Sort val_scored.tmp on disk by s1_id for streaming evaluation
    print(f"[INFO] Sorting validation predictions on disk by s1_id...")
    val_scored_sorted_path = work_dir / "val_scored_sorted.tmp"
    subprocess.run(
        ["sort", "-t", "\t", "-k1,1", str(val_scored_path), "-o", str(val_scored_sorted_path)],
        check=True,
    )
    if val_scored_path.exists():
        val_scored_path.unlink()

    # Sweep decision threshold t in [0.10 .. 0.90] at 0.02 step
    print(f"[INFO] Sweeping Decision Thresholds via streaming S1 group evaluation...")
    best_t = 0.50
    best_f05 = -1.0
    best_metrics = {}

    thresholds = [round(0.10 + 0.02 * i, 2) for i in range(41)]
    val_s1_set = set(val_s1_ids)
    total_val_s1_count = len(val_s1_ids)

    for t in thresholds:
        sum_f05 = 0.0
        sum_prec = 0.0
        sum_rec = 0.0
        singleton_correct = 0
        singleton_total = 0

        evaluated_s1_ids: Set[str] = set()

        if val_scored_sorted_path.exists() and val_scored_sorted_path.stat().st_size > 0:
            with open(val_scored_sorted_path, "r", encoding="utf-8") as f_scored:
                curr_s1_id: Optional[str] = None
                curr_pred_set: Set[str] = set()

                for line in f_scored:
                    parts = line.strip().split("\t")
                    if len(parts) < 3:
                        continue
                    s_id, n_id, prob_str = parts[0], parts[1], parts[2]
                    prob = float(prob_str)

                    if curr_s1_id is None:
                        curr_s1_id = s_id

                    if s_id != curr_s1_id:
                        # Evaluate completed s1_id
                        gt_set = gt_matches.get(curr_s1_id, set())
                        p, r, f, is_sing_corr = evaluate_single_s1_entity(curr_pred_set, gt_set)
                        sum_prec += p
                        sum_rec += r
                        sum_f05 += f
                        if is_sing_corr is not None:
                            singleton_total += 1
                            if is_sing_corr:
                                singleton_correct += 1

                        evaluated_s1_ids.add(curr_s1_id)
                        curr_s1_id = s_id
                        curr_pred_set.clear()

                    if prob >= t:
                        curr_pred_set.add(n_id)

                # Evaluate final group in file
                if curr_s1_id is not None:
                    gt_set = gt_matches.get(curr_s1_id, set())
                    p, r, f, is_sing_corr = evaluate_single_s1_entity(curr_pred_set, gt_set)
                    sum_prec += p
                    sum_rec += r
                    sum_f05 += f
                    if is_sing_corr is not None:
                        singleton_total += 1
                        if is_sing_corr:
                            singleton_correct += 1
                    evaluated_s1_ids.add(curr_s1_id)

        # Evaluate all validation S1 entities that had zero candidates retrieved by blocker
        unevaluated_s1 = val_s1_set - evaluated_s1_ids
        for un_s1 in unevaluated_s1:
            gt_set = gt_matches.get(un_s1, set())
            p, r, f, is_sing_corr = evaluate_single_s1_entity(set(), gt_set)
            sum_prec += p
            sum_rec += r
            sum_f05 += f
            if is_sing_corr is not None:
                singleton_total += 1
                if is_sing_corr:
                    singleton_correct += 1

        macro_f05 = sum_f05 / total_val_s1_count if total_val_s1_count > 0 else 0.0
        mean_prec = sum_prec / total_val_s1_count if total_val_s1_count > 0 else 0.0
        mean_rec = sum_rec / total_val_s1_count if total_val_s1_count > 0 else 0.0
        singleton_acc = (singleton_correct / singleton_total) if singleton_total > 0 else 1.0

        if macro_f05 > best_f05:
            best_f05 = macro_f05
            best_t = t
            best_metrics = {
                "macro_f05": macro_f05,
                "mean_precision": mean_prec,
                "mean_recall": mean_rec,
                "singleton_accuracy": singleton_acc,
            }

    if val_scored_sorted_path.exists():
        val_scored_sorted_path.unlink()

    print("=" * 70)
    print(f"[SUCCESS] Threshold Sweep Complete.")
    print(f"[SUMMARY] Optimal Threshold (t*): {best_t:.2f}")
    print(f"[SUMMARY] Validation Macro F0.5:   {best_metrics['macro_f05']:.6f}")
    print(f"[SUMMARY] Validation Precision:    {best_metrics['mean_precision']:.6f}")
    print(f"[SUMMARY] Validation Recall:       {best_metrics['mean_recall']:.6f}")
    print(f"[SUMMARY] Singleton Accuracy:      {best_metrics['singleton_accuracy']:.6f}")
    print("=" * 70)

    return clf, best_t, best_metrics


def sort_partition_file_by_s1_idx(p_path: Path) -> Path:
    """
    Sorts a partition file on disk by integer s1_idx using GNU sort safely via
    subprocess argument lists without constructing a shell command string.
    """
    if not p_path.exists() or p_path.stat().st_size == 0:
        return p_path

    sorted_path = p_path.with_suffix(".sorted.tmp")

    subprocess.run(
        ["sort", "-t", "\t", "-k1,1n", str(p_path), "-o", str(sorted_path)],
        check=True,
    )

    if sorted_path.exists():
        p_path.unlink()
        sorted_path.rename(p_path)

    return p_path


def run_partitioned_test_inference(
    clf: object,
    optimal_t: float,
    s1_idx_obj: S1CompactIndex,
    doc_freq_addr: Counter,
    source_paths: List[Path],
    output_cand_path: Path,
    output_match_path: Path,
    work_dir: Path,
    top_k_broad: int = 15,
    encoding: str = "utf-8",
    max_rows: Optional[int] = None,
    num_partitions: int = NUM_PARTITIONS,
    batch_size: int = 50_000,
) -> None:
    """
    Partitioned Test Streaming Inference:
      1. Opens P=16 disk partition files.
      2. Streams test S2/S3 row-by-row, batches features (50,000 pairs/batch), predicts probabilities,
         and writes candidate/match pairs to partition files.
      3. Sorts each partition file on disk by integer s1_idx.
      4. Streams group-by-s1_idx to emit candidate_pairs.tsv and matching_results.tsv without building global dictionaries in RAM.
    """
    print(f"[INFO] Starting Batched Partitioned Test Streaming Inference (P={num_partitions}, Batch={batch_size:,})...")
    t0 = time.time()

    work_dir.mkdir(parents=True, exist_ok=True)
    partition_handles = []
    partition_paths = []

    for p in range(num_partitions):
        p_path = work_dir / f"test_part_{p}.tmp"
        partition_paths.append(p_path)
        partition_handles.append(open(p_path, "w", encoding="utf-8", newline=""))

    total_noisy_rows = 0

    batch_feats: List[List[float]] = []
    batch_meta: List[Tuple[int, str, str]] = []  # List of (s1_idx, s1_id, noisy_id)

    def flush_inference_batch(
        b_feats: List[List[float]],
        b_meta: List[Tuple[int, str, str]],
    ) -> None:
        if not b_feats:
            return
        probs = clf.predict_proba(b_feats)[:, 1]
        for (s1_idx, s1_id, noisy_id), prob in zip(b_meta, probs):
            is_match = 1 if prob >= optimal_t else 0
            part_idx = s1_idx % num_partitions
            partition_handles[part_idx].write(f"{s1_idx}\t{s1_id}\t{noisy_id}\t{is_match}\n")
        b_feats.clear()
        b_meta.clear()

    for src_path in source_paths:
        print(f"[INFO]   Streaming test records from {src_path.resolve()}...")
        for row in stream_tsv_rows(src_path, SOURCE_COLUMNS, encoding=encoding, max_rows=max_rows):
            total_noisy_rows += 1
            noisy_id = row["entity_id"]

            candidate_s1_indices = extract_candidates_for_row(
                row, s1_idx_obj, doc_freq_addr, top_k_broad=top_k_broad
            )

            if not candidate_s1_indices:
                continue

            for s1_idx in candidate_s1_indices:
                s1_id = s1_idx_obj.s1_ids[s1_idx]
                s1_row = {
                    "entity_id": s1_id,
                    "business_name": s1_idx_obj.s1_names[s1_idx],
                    "business_address": s1_idx_obj.s1_addrs[s1_idx],
                    "country": s1_idx_obj.s1_countries[s1_idx],
                }

                feats = extract_v1_features(s1_row, row)
                batch_feats.append(feats)
                batch_meta.append((s1_idx, s1_id, noisy_id))

                if len(batch_feats) >= batch_size:
                    flush_inference_batch(batch_feats, batch_meta)

            if total_noisy_rows % 1_000_000 == 0:
                print(f"[INFO]     Processed {total_noisy_rows:,} test rows... (RSS: {format_rss()})")

    # Flush final remaining batch
    flush_inference_batch(batch_feats, batch_meta)

    # Close partition handles
    for h in partition_handles:
        h.close()

    print(f"[INFO] Partitioned Candidate Generation Complete in {time.time() - t0:.2f}s. Assembling Output TSVs...")

    output_cand_path.parent.mkdir(parents=True, exist_ok=True)
    output_match_path.parent.mkdir(parents=True, exist_ok=True)

    cand_out_f = open(output_cand_path, "w", encoding="utf-8", newline="")
    match_out_f = open(output_match_path, "w", encoding="utf-8", newline="")

    cand_out_f.write("source1_entity_id\tcandidate_entity_ids\n")
    match_out_f.write("source1_entity_id\tmatched_entity_ids\n")

    total_candidates_emitted = 0
    total_matches_emitted = 0
    singletons_count = 0

    # Process each partition file sequentially
    for p in range(num_partitions):
        p_path = partition_paths[p]
        sort_partition_file_by_s1_idx(p_path)

        current_s1_idx: Optional[int] = None
        current_c_set: Set[str] = set()
        current_m_set: Set[str] = set()

        def flush_group(s1_idx_val: int, c_set: Set[str], m_set: Set[str]) -> Tuple[int, int]:
            s1_id = s1_idx_obj.s1_ids[s1_idx_val]
            cand_list = sorted(c_set)
            match_list = sorted(m_set)
            cand_str = ",".join(cand_list)
            match_str = ",".join(match_list)

            cand_out_f.write(f"{s1_id}\t{cand_str}\n")
            match_out_f.write(f"{s1_id}\t{match_str}\n")
            return len(cand_list), len(match_list)

        # Track emitted S1 indices within this partition to handle any missing singletons
        emitted_partition_s1 = set()

        if p_path.exists() and p_path.stat().st_size > 0:
            with open(p_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                for fields in reader:
                    if not fields or len(fields) < 4:
                        continue
                    s1_idx = int(fields[0])
                    noisy_id = fields[2]
                    is_match = int(fields[3])

                    if current_s1_idx is None:
                        current_s1_idx = s1_idx

                    if s1_idx != current_s1_idx:
                        # Flush completed S1 group immediately
                        c_cnt, m_cnt = flush_group(current_s1_idx, current_c_set, current_m_set)
                        total_candidates_emitted += c_cnt
                        total_matches_emitted += m_cnt
                        emitted_partition_s1.add(current_s1_idx)

                        current_s1_idx = s1_idx
                        current_c_set.clear()
                        current_m_set.clear()

                    current_c_set.add(noisy_id)
                    if is_match == 1:
                        current_m_set.add(noisy_id)

            # Flush final group in file
            if current_s1_idx is not None:
                c_cnt, m_cnt = flush_group(current_s1_idx, current_c_set, current_m_set)
                total_candidates_emitted += c_cnt
                total_matches_emitted += m_cnt
                emitted_partition_s1.add(current_s1_idx)

            p_path.unlink()  # Delete temp file immediately

        # Output rows for any S1 entities assigned to partition p that had zero candidates (singletons)
        for s1_idx in range(p, len(s1_idx_obj.s1_ids), num_partitions):
            if s1_idx not in emitted_partition_s1:
                s1_id = s1_idx_obj.s1_ids[s1_idx]
                cand_out_f.write(f"{s1_id}\t\n")
                match_out_f.write(f"{s1_id}\t\n")
                singletons_count += 1

    cand_out_f.close()
    match_out_f.close()

    print("=" * 70)
    print(f"[SUCCESS] V1 Test Inference Completed.")
    print(f"[SUMMARY] Total S1 Test Entities:    {len(s1_idx_obj.s1_ids):,}")
    print(f"[SUMMARY] Singletons (0 candidates): {singletons_count:,}")
    print(f"[SUMMARY] Total Emitted Candidates:  {total_candidates_emitted:,}")
    print(f"[SUMMARY] Total Emitted Matches:     {total_matches_emitted:,}")
    print(f"[SUMMARY] Wrote Candidate Pairs TSV: {output_cand_path.resolve()}")
    print(f"[SUMMARY] Wrote Matching Results TSV:{output_match_path.resolve()}")
    print("=" * 70)