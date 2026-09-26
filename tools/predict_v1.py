#!/usr/bin/env python3
"""
predict_v1.py

CLI entrypoint for V1 test inference and submission file generation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import check_free_disk_space
from v1_pipeline import (
    build_s1_compact_index,
    run_partitioned_test_inference,
)

try:
    import joblib
except ImportError:
    joblib = None


def main() -> int:
    parser = argparse.ArgumentParser(description="V1 Test Inference & Submission Generator")
    parser.add_argument("--model-path", type=Path, default=Path("v1_model.joblib"), help="Path to trained model artifact")
    parser.add_argument("--test-dir", type=Path, required=True, help="Path to test dataset directory")
    parser.add_argument("--work-dir", type=Path, default=Path("tmp_v1_predict_work"), help="Directory for partition files")
    parser.add_argument("--output-cand-path", type=Path, default=Path("output/candidate_pairs.tsv"), help="Output path for candidate_pairs.tsv")
    parser.add_argument("--output-match-path", type=Path, default=Path("output/matching_results.tsv"), help="Output path for matching_results.tsv")
    parser.add_argument("--top-k-broad", type=int, default=None, help="Override top-K broad candidates per noisy row")
    parser.add_argument("--max-rows", type=int, default=None, help="Small-sample run: cap rows read per file")
    parser.add_argument("--encoding", type=str, default="utf-8")

    args = parser.parse_args()

    print("=" * 70)
    print(f"[INFO] Starting Amazon ML Challenge V1 Test Inference Pipeline")
    print("=" * 70)

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {args.model_path.resolve()}")

    if joblib is not None:
        artifact = joblib.load(args.model_path)
    else:
        import pickle
        with open(args.model_path, "rb") as f:
            artifact = pickle.load(f)

    clf = artifact["model"]
    optimal_t = float(artifact["optimal_threshold"])
    top_k_broad = args.top_k_broad if args.top_k_broad is not None else int(artifact.get("top_k_broad", 15))

    print(f"[INFO] Loaded Model Artifact: Optimal Threshold t* = {optimal_t:.2f}, Top-K Broad = {top_k_broad}")

    test_s1_path = args.test_dir / "test_source1.tsv"
    test_s2_path = args.test_dir / "test_source2.tsv"
    test_s3_path = args.test_dir / "test_source3.tsv"

    check_free_disk_space(args.work_dir, min_bytes=1_073_741_824)

    # 1. Build Compact Index for Test Source 1
    s1_idx_obj, doc_freq_addr = build_s1_compact_index(test_s1_path, encoding=args.encoding, max_rows=args.max_rows)

    # 2. Run Partitioned Test Streaming Inference
    run_partitioned_test_inference(
        clf,
        optimal_t,
        s1_idx_obj,
        doc_freq_addr,
        [test_s2_path, test_s3_path],
        args.output_cand_path,
        args.output_match_path,
        args.work_dir,
        top_k_broad=top_k_broad,
        encoding=args.encoding,
        max_rows=args.max_rows,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())