#!/usr/bin/env python3
"""
log_experiment.py

CLI utility to append experiment records to reports/experiments.csv.
Uses standard library packages only.
Automatically creates reports/experiments.csv with the correct header if missing.
Prevents accidental duplicate experiment IDs.
"""

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

COLUMNS = [
    "experiment_id",
    "timestamp_utc",
    "git_commit",
    "blocking_strategy",
    "matcher",
    "model_params",
    "feature_set",
    "decision_threshold",
    "validation_split",
    "validation_f05",
    "validation_precision",
    "validation_recall",
    "singleton_accuracy",
    "candidate_mean",
    "candidate_median",
    "candidate_p95",
    "candidate_p99",
    "candidate_max",
    "public_lb_score",
    "leaderboard_rank",
    "submission_day",
    "submission_number",
    "change_summary",
    "result",
    "notes",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Append an experiment entry to reports/experiments.csv safely."
    )

    parser.add_argument("--experiment-id", type=str, required=True, help="Unique Experiment ID (e.g. EXP-001)")
    parser.add_argument("--timestamp-utc", type=str, default="", help="UTC Timestamp (defaults to current time)")
    parser.add_argument("--git-commit", type=str, default="", help="Git commit hash")
    parser.add_argument("--blocking-strategy", type=str, default="", help="Blocking strategy definition")
    parser.add_argument("--matcher", type=str, default="", help="Matcher algorithm / settings")
    parser.add_argument("--model-params", type=str, default="", help="Model hyperparameters (e.g. num_leaves=31,lr=0.05)")
    parser.add_argument("--feature-set", type=str, default="", help="Feature set description")
    parser.add_argument("--decision-threshold", type=str, default="", help="Decision threshold probability cutoff (e.g. 0.65)")
    parser.add_argument("--validation-split", type=str, default="", help="Validation split definition")
    parser.add_argument("--validation-f05", type=str, default="", help="Local Macro F0.5 score")
    parser.add_argument("--validation-precision", type=str, default="", help="Local Mean Precision")
    parser.add_argument("--validation-recall", type=str, default="", help="Local Mean Recall")
    parser.add_argument("--singleton-accuracy", type=str, default="", help="Accuracy on singleton entities")
    parser.add_argument("--candidate-mean", type=str, default="", help="Mean candidates per S1 entity")
    parser.add_argument("--candidate-median", type=str, default="", help="Median candidates per S1 entity")
    parser.add_argument("--candidate-p95", type=str, default="", help="P95 candidates per S1 entity")
    parser.add_argument("--candidate-p99", type=str, default="", help="P99 candidates per S1 entity")
    parser.add_argument("--candidate-max", type=str, default="", help="Max candidates per S1 entity")
    parser.add_argument("--public-lb-score", type=str, default="", help="Portal public leaderboard score")
    parser.add_argument("--leaderboard-rank", type=str, default="", help="Portal public leaderboard rank")
    parser.add_argument("--submission-day", type=str, default="", help="Submission day index")
    parser.add_argument("--submission-number", type=str, default="", help="Cumulative portal submission number")
    parser.add_argument("--change-summary", type=str, default="", help="Summary of changes in this experiment")
    parser.add_argument("--result", type=str, default="", help="Key result or decision outcome")
    parser.add_argument("--notes", type=str, default="", help="Additional operational notes")
    parser.add_argument("--root", type=Path, default=None, help="Root path of project repository")

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    root = args.root if args.root else script_dir.parent
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "experiments.csv"

    # Auto-generate UTC timestamp if not explicitly provided
    timestamp_val = args.timestamp_utc
    if not timestamp_val:
        timestamp_val = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    exp_id = args.experiment_id.strip()

    # Read existing entries to check for duplicate experiment IDs
    existing_ids = set()
    file_exists = csv_path.exists()

    if file_exists:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                for row in reader:
                    if "experiment_id" in row and row["experiment_id"]:
                        existing_ids.add(row["experiment_id"].strip())

    if exp_id in existing_ids:
        print(f"[ERROR] Experiment ID '{exp_id}' already exists in {csv_path.resolve()}.", file=sys.stderr)
        print("[ERROR] Duplicate experiment IDs are strictly forbidden. Use a new ID (e.g. EXP-xxx).", file=sys.stderr)
        return 1

    # Prepare row dictionary matching exact CSV column headers
    row_data = {
        "experiment_id": exp_id,
        "timestamp_utc": timestamp_val,
        "git_commit": args.git_commit.strip(),
        "blocking_strategy": args.blocking_strategy.strip(),
        "matcher": args.matcher.strip(),
        "model_params": args.model_params.strip(),
        "feature_set": args.feature_set.strip(),
        "decision_threshold": args.decision_threshold.strip(),
        "validation_split": args.validation_split.strip(),
        "validation_f05": args.validation_f05.strip(),
        "validation_precision": args.validation_precision.strip(),
        "validation_recall": args.validation_recall.strip(),
        "singleton_accuracy": args.singleton_accuracy.strip(),
        "candidate_mean": args.candidate_mean.strip(),
        "candidate_median": args.candidate_median.strip(),
        "candidate_p95": args.candidate_p95.strip(),
        "candidate_p99": args.candidate_p99.strip(),
        "candidate_max": args.candidate_max.strip(),
        "public_lb_score": args.public_lb_score.strip(),
        "leaderboard_rank": args.leaderboard_rank.strip(),
        "submission_day": args.submission_day.strip(),
        "submission_number": args.submission_number.strip(),
        "change_summary": args.change_summary.strip(),
        "result": args.result.strip(),
        "notes": args.notes.strip(),
    }

    # Write header if file does not exist or is empty
    write_header = not file_exists or csv_path.stat().st_size == 0

    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row_data)

    print(f"[SUCCESS] Experiment '{exp_id}' logged successfully to {csv_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())