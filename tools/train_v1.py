#!/usr/bin/env python3
"""
train_v1.py

CLI entrypoint to train V1 Entity Resolution Matcher using bounded disk-backed streaming.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import check_free_disk_space
from evaluate_f05 import read_match_tsv
from v1_pipeline import (
    build_s1_compact_index,
    stream_training_data_to_disk,
    train_and_validate_v1_model,
)

try:
    import joblib
except ImportError:
    joblib = None


def main() -> int:
    parser = argparse.ArgumentParser(description="Train V1 Entity Resolution Pipeline")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=None,
        help="Directory containing train_source1.tsv, train_source2.tsv, train_source3.tsv, and train_ground_truth.tsv",
    )
    parser.add_argument("--root", type=Path, default=None, help="Root path of project dataset fallback")
    parser.add_argument("--work-dir", type=Path, default=Path("tmp_v1_work"), help="Directory for disk-backed intermediate buffers")
    parser.add_argument("--top-k-broad", type=int, default=15, help="Max candidates per noisy row from broad blockers")
    parser.add_argument("--negatives-per-s1", type=int, default=15, help="Hard negatives sampled per S1 entity")
    parser.add_argument("--max-rows", type=int, default=None, help="Small-sample run: cap rows read per file")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--output-model", type=Path, default=Path("v1_model.joblib"), help="Path to save trained model")
    parser.add_argument("--encoding", type=str, default="utf-8")

    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    root = args.root if args.root else script_dir.parent

    # Path Resolution Logic
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
    print(f"[INFO] Starting Amazon ML Challenge V1 Model Training Pipeline")
    print(f"[INFO] Mode: {'SMALL-SAMPLE' if args.max_rows else 'FULL DATASET'}")
    print(f"[INFO] Configuration: Top-K Broad={args.top_k_broad}, K_neg={args.negatives_per_s1}, Seed={args.seed}")
    print(f"[INFO] Resolved Dataset Paths:")
    print(f"[INFO]   Ground Truth: {gt_path.resolve()}")
    print(f"[INFO]   Source 1:     {s1_path.resolve()}")
    print(f"[INFO]   Source 2:     {s2_path.resolve()}")
    print(f"[INFO]   Source 3:     {s3_path.resolve()}")
    print("=" * 70)

    check_free_disk_space(args.work_dir, min_bytes=2_147_483_648)

    # 1. Load Ground Truth Matches
    gt_matches = read_match_tsv(gt_path)

    # 2. Build Compact S1 Index & Address Document Frequencies
    s1_idx_obj, doc_freq_addr = build_s1_compact_index(s1_path, encoding=args.encoding, max_rows=args.max_rows)

    # 3. Stream Features Directly to Disk
    train_feats_p, train_labels_p, val_cand_p, train_indices, val_indices, blocker_recall = stream_training_data_to_disk(
        s1_idx_obj,
        doc_freq_addr,
        gt_matches,
        [s2_path, s3_path],
        args.work_dir,
        seed=args.seed,
        top_k_broad=args.top_k_broad,
        negatives_per_s1=args.negatives_per_s1,
        encoding=args.encoding,
        max_rows=args.max_rows,
    )

    # 4. Train Model & Validate Threshold
    val_s1_ids = [s1_idx_obj.s1_ids[i] for i in val_indices]
    clf, best_t, metrics = train_and_validate_v1_model(
        train_feats_p, train_labels_p, val_cand_p, gt_matches, val_s1_ids, work_dir=args.work_dir, seed=args.seed
    )

    # 5. Save Artifact
    artifact = {
        "model": clf,
        "optimal_threshold": best_t,
        "metrics": metrics,
        "bounded_blocker_recall": blocker_recall,
        "top_k_broad": args.top_k_broad,
        "negatives_per_s1": args.negatives_per_s1,
        "trained_at_utc": str(time.time()),
    }

    if joblib is not None:
        joblib.dump(artifact, args.output_model)
    else:
        import pickle
        with open(args.output_model, "wb") as f:
            pickle.dump(artifact, f)

    print(f"[SUCCESS] Saved model artifact to {args.output_model.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())