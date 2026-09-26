# Experimentation and Submission Tracking

## Purpose
This directory tracks every pipeline experiment and leaderboard submission for the Amazon ML Challenge entity resolution system. The objective is to maintain strict scientific reproducibility, prevent regression, and systematically correlate local validation metrics with public leaderboard feedback.

## Naming Conventions

### Experiment IDs
- Format: `EXP-001`, `EXP-002`, `EXP-003`, ...
- Rule: Every pipeline iteration (blocking change, feature addition, model retrain, threshold adjustment) MUST be assigned a unique, strictly incrementing experiment ID.

### Submission IDs
- Format: `SUB-001`, `SUB-002`, `SUB-003`, ...
- Rule: A submission ID is assigned ONLY when candidate pairs and prediction files are submitted to the challenge portal. A submission MUST reference the exact `experiment_id` that generated the predictions.

## Recorded Experiment Information (`experiments.csv`)

| Column Name | Description |
| :--- | :--- |
| `experiment_id` | Unique ID (`EXP-001`) |
| `timestamp_utc` | UTC timestamp of experiment run (`YYYY-MM-DDTHH:MM:SSZ`) |
| `git_commit` | Git commit hash at run time (e.g., `a1b2c3d`) |
| `blocking_strategy` | Blocker rule definition (e.g., `C_union_E`, `A_G_union_CandE`) |
| `matcher` | Matching model architecture (e.g., `LightGBM_binary`, `Rule_based`) |
| `model_params` | Hyperparameters used for the model (e.g., `num_leaves=31,lr=0.05,K=15`) |
| `feature_set` | Feature set description (e.g., `15_features_v1`) |
| `decision_threshold` | Probability cutoff $t^*$ used for classification (e.g., `0.65`) |
| `validation_split` | Train/validation split strategy (e.g., `s1_grouped_80_20_seed1337`) |
| `validation_f05` | Local Macro F0.5 score across all validation Source-1 entities |
| `validation_precision` | Mean local precision across validation Source-1 entities |
| `validation_recall` | Mean local recall across validation Source-1 entities |
| `singleton_accuracy` | Accuracy on Source-1 entities with zero true ground-truth matches |
| `candidate_mean` | Mean candidate pairs per Source-1 entity in candidate generation |
| `candidate_median` | Median candidate pairs per Source-1 entity |
| `candidate_p95` | 95th percentile candidate pair count |
| `candidate_p99` | 99th percentile candidate pair count |
| `candidate_max` | Maximum candidate count for a single Source-1 entity |
| `public_lb_score` | Score returned by portal (leave blank if not submitted) |
| `leaderboard_rank` | Portal rank after submission (leave blank if not submitted) |
| `submission_day` | Day index of submission during competition period |
| `submission_number` | Cumulative submission count on portal |
| `change_summary` | One-sentence summary of hypothesis / technical change |
| `result` | Outcome summary (e.g., `Improved local F0.5`, `Rejected due to RAM`) |
| `notes` | Additional operational context |

## Local Validation Policy
Local validation MUST exactly replicate the competition evaluation metric:
1. Compute Precision, Recall, and $F_{0.5}$ separately for each individual Source-1 entity.
2. Include true singletons in the evaluation:
   - True singleton with empty predicted match list = $1.0$ score.
   - True singleton with any predicted match = $0.0$ score.
3. Compute the **Macro-Average $F_{0.5}$** across ALL validation Source-1 entities.
4. Do NOT use globally pooled (micro) precision and recall as the primary decision metric.

## Leaderboard Policy
- Public leaderboard scores and ranks are empirical observations recorded directly from the challenge portal.
- **NEVER** estimate, predict, or fabricate public leaderboard scores or ranks.
- If an experiment is not submitted to the portal, its `public_lb_score` and `leaderboard_rank` must remain blank.

## Reproducibility Requirements
Every logged experiment must record:
- Exact Git commit hash (`git rev-parse --short HEAD`).
- Exact blocking rules and thresholds.
- Exact feature extraction configuration.
- Exact model hyperparameter settings (`model_params`) and decision threshold (`decision_threshold`).
- Measured local validation metrics and candidate volume distributions.

## Submission Discipline
A new leaderboard submission should only be executed after satisfying all five criteria:
1. **Local Validation**: Local Macro $F_{0.5}$ evaluated on the held-out split shows measurable improvement or validates a critical architectural change.
2. **Baseline Comparison**: Metrics outperform or meaningfully complement the current protected baseline.
3. **Singleton Inspection**: False merge rate on singleton entities is explicitly verified and bounded.
4. **Output Verification**: Final prediction files pass all checks via `utils/validate_submission.py`.
5. **Explicit Decision**: The team explicitly agrees to consume a daily submission quota.

*Note: Do not assume the platform automatically retains your best submission unless explicitly verified in official competition rules.*