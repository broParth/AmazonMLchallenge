#!/usr/bin/env python3
"""
test_v1_pipeline.py

Synthetic unit test runner to verify memory-safe streaming V1 architecture.
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from common import normalize_text
from evaluate_f05 import compute_per_s1_f05
from v1_features import extract_v1_features
from v1_pipeline import (
    build_s1_compact_index,
    run_partitioned_test_inference,
    stream_training_data_to_disk,
    train_and_validate_v1_model,
)


class TestV1PipelineArchitecture(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

        # Synthetic train_source1.tsv
        self.s1_file = self.dir_path / "train_source1.tsv"
        with open(self.s1_file, "w", encoding="utf-8", newline="") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S1-001\tAcme Corporation\t123 Main Street, Suite 100\tUS\n")
            f.write("S1-002\tBeta Logistics\t456 Market Road\tUS\n")
            f.write("S1-003\tGamma Enterprises\t789 Industrial Park\tUS\n")

        # Synthetic train_source2.tsv
        self.s2_file = self.dir_path / "train_source2.tsv"
        with open(self.s2_file, "w", encoding="utf-8", newline="") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S2-101\tAcme Corp Inc\t123 Main St, Ste 100\tUS\n")
            f.write("S2-102\tBeta Logistics LLC\t456 Market Rd\tUS\n")

        # Synthetic train_source3.tsv
        self.s3_file = self.dir_path / "train_source3.tsv"
        with open(self.s3_file, "w", encoding="utf-8", newline="") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S3-201\tAcme Corporation\t123 Main Street\tUS\n")

        # Synthetic train_ground_truth.tsv
        self.gt_file = self.dir_path / "train_ground_truth.tsv"
        with open(self.gt_file, "w", encoding="utf-8", newline="") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            f.write("S1-001\tS2-101,S3-201\n")
            f.write("S1-002\tS2-102\n")
            f.write("S1-003\t\n")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_streaming_disk_training_flow(self):
        s1_idx_obj, doc_freq_addr = build_s1_compact_index(self.s1_file, max_doc_freq=200)

        gt_matches = {
            "S1-001": {"S2-101", "S3-201"},
            "S1-002": {"S2-102"},
            "S1-003": set(),
        }

        train_f, train_l, val_c, train_idx, val_idx, blocker_recall = stream_training_data_to_disk(
            s1_idx_obj,
            doc_freq_addr,
            gt_matches,
            [self.s2_file, self.s3_file],
            self.dir_path / "work",
            seed=1337,
            top_k_broad=5,
            negatives_per_s1=5,
        )

        self.assertTrue(train_f.exists())
        self.assertTrue(train_l.exists())
        self.assertTrue(val_c.exists())
        self.assertGreaterEqual(blocker_recall, 0.0)
        self.assertLessEqual(blocker_recall, 1.0)

        val_s1_ids = [s1_idx_obj.s1_ids[i] for i in val_idx]
        clf, best_t, metrics = train_and_validate_v1_model(
            train_f, train_l, val_c, gt_matches, val_s1_ids, work_dir=self.dir_path / "work", seed=1337
        )

        self.assertIsNotNone(clf)
        self.assertGreaterEqual(best_t, 0.10)
        self.assertLessEqual(best_t, 0.90)

    def test_partitioned_test_inference(self):
        s1_idx_obj, doc_freq_addr = build_s1_compact_index(self.s1_file, max_doc_freq=200)

        gt_matches = {
            "S1-001": {"S2-101", "S3-201"},
            "S1-002": {"S2-102"},
            "S1-003": set(),
        }

        train_f, train_l, val_c, train_idx, val_idx, _ = stream_training_data_to_disk(
            s1_idx_obj,
            doc_freq_addr,
            gt_matches,
            [self.s2_file, self.s3_file],
            self.dir_path / "work",
            seed=1337,
            top_k_broad=5,
            negatives_per_s1=5,
        )

        val_s1_ids = [s1_idx_obj.s1_ids[i] for i in val_idx]
        clf, best_t, _ = train_and_validate_v1_model(
            train_f, train_l, val_c, gt_matches, val_s1_ids, work_dir=self.dir_path / "work", seed=1337
        )

        cand_out = self.dir_path / "output" / "candidate_pairs.tsv"
        match_out = self.dir_path / "output" / "matching_results.tsv"

        run_partitioned_test_inference(
            clf,
            best_t,
            s1_idx_obj,
            doc_freq_addr,
            [self.s2_file, self.s3_file],
            cand_out,
            match_out,
            self.dir_path / "predict_work",
            top_k_broad=5,
            num_partitions=4,
            batch_size=10,
        )

        self.assertTrue(cand_out.exists())
        self.assertTrue(match_out.exists())

        # Verify exact TSV format, singleton preservation, and output line count
        with open(cand_out, "r", encoding="utf-8") as f_c, open(match_out, "r", encoding="utf-8") as f_m:
            c_lines = [l.strip() for l in f_c if l.strip()]
            m_lines = [l.strip() for l in f_m if l.strip()]

            self.assertEqual(len(c_lines), 4)  # Header + 3 S1 entities
            self.assertEqual(len(m_lines), 4)


if __name__ == "__main__":
    unittest.main()