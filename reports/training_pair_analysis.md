# Training Pair Analysis Report

- Generated (UTC): 2026-09-25T07:01:46.302345+00:00
- Search root: `D:\Documents\PARTH\AmazonML\AmazonMLchallenge`
- Script version: 1.0.0, seed: 1337
- No raw dataset file was modified. No ML training, final matching, or test predictions were performed. This is an evidence-gathering report only.

## Sampling disclosure
- Total true-match pairs in ground truth: 7638365
- Pairs sampled for similarity analysis (seeded reservoir sample): 200000
- Sampled pairs resolvable to actual records: 200000
- Sampled pairs NOT resolvable (record missing from referenced sets): 0
- **All similarity statistics below are estimates from a sample, not the full population.** Increase `--max-pairs-for-similarity` for a larger sample if your machine can handle the runtime.

## A. Dataset facts actually observed
- train_source1.tsv rows indexed: 2206821
- train_source2.tsv rows scanned: 5034616, referenced records kept: 3693619
- train_source3.tsv rows scanned: 5285603, referenced records kept: 3944746
- Ground-truth rows: 2206821
- Singleton Source-1 entities: 123247
- Match-count distribution (num_matches -> num_S1_entities): {'0': 123247, '1': 119157, '2': 375212, '3': 530841, '4': 484115, '5': 321957, '6': 164868, '7': 63968, '8': 18680, '9': 4205, '10': 534, '11': 37}
- S2-only matches: 143029, S3-only: 164498, both: 1776047

## B. True-match characteristics
- Exact raw name equality rate: 0.04639
- Exact normalized name equality rate: 0.219215
- Name Levenshtein similarity distribution: {'count': 200000, 'min': 0.0, 'median': 0.7894736842105263, 'mean': 0.7193, 'p90': 1.0, 'p99': 1.0, 'max': 1.0}
- Name Jaccard (token overlap) distribution: {'count': 200000, 'min': 0.0, 'median': 0.6666666666666666, 'mean': 0.6162, 'p90': 1.0, 'p99': 1.0, 'max': 1.0}
- TF-IDF note: TF-IDF cosine skipped: scikit-learn is not installed in this environment.
- Exact raw address equality rate: 0.021905
- Exact normalized address equality rate: 0.08252
- Address Levenshtein similarity distribution: {'count': 200000, 'min': 0.0, 'median': 0.7105263157894737, 'mean': 0.6389, 'p90': 0.9591836734693877, 'p99': 1.0, 'max': 1.0}
- Address Jaccard distribution: {'count': 200000, 'min': 0.0, 'median': 0.625, 'mean': 0.5967, 'p90': 1.0, 'p99': 1.0, 'max': 1.0}
- Rate of true matches with at least one empty address: 0.04395
- Country equality rate among true matches where both countries are known: 1.0 (n=200000)
- True matches with DIFFERENT country labels on each side: 0 -- country equality alone should not be assumed to guarantee identity, and this dataset shows why.

### Evidence buckets (name/address strength combinations, analytical bins only)
- `name=exact/address=strong`: 12284
- `name=strong/address=dissimilar`: 6281
- `name=weak/address=exact`: 7175
- `name=weak/address=weak`: 34081
- `name=exact/address=dissimilar`: 7329
- `name=exact/address=weak`: 19291
- `name=weak/address=strong`: 22099
- `name=dissimilar/address=dissimilar`: 8203
- `name=strong/address=strong`: 10728
- `name=strong/address=weak`: 16398
- `name=dissimilar/address=weak`: 14829
- `name=weak/address=empty`: 3802
- `name=weak/address=dissimilar`: 13595
- `name=dissimilar/address=exact`: 4274
- `name=strong/address=empty`: 2013
- `name=strong/address=exact`: 2404
- `name=dissimilar/address=strong`: 9588
- `name=dissimilar/address=empty`: 687
- `name=exact/address=exact`: 2651
- `name=exact/address=empty`: 2288
- Thresholds used for binning (NOT final model thresholds): name strong >= 0.85, name weak >= 0.5; address strong >= 0.8, address weak >= 0.4.

## C. Singleton characteristics
- Total singleton Source-1 entities: 123247
- Singleton examples analyzed for duplicate-name/address diagnostics: 50
- Diagnostics below run on 50 of 123247 singleton entities (capped example list from Pass A, not a fresh random sample of the full singleton population). Frequency-based checks SKIPPED (--skip-corpus-frequency).
- Of those, entities whose name has an exact-normalized duplicate somewhere in S2/S3: 0
- Of those, entities whose address has an exact-normalized duplicate somewhere in S2/S3: 0
- **These are diagnostics only.** A name/address duplicate does NOT mean the singleton actually has a hidden match -- correctly predicting a singleton as unmatched is rewarded under macro F0.5, and this section exists only to flag where that could be *harder*.

## D. Positive-vs-negative similarity observations
- Skipped (--skip-corpus-frequency).
- True-match pairs checked: 0
- ... where the matched record's name has other exact-normalized duplicates in its own source: 0
- ... where the matched record's address has other exact-normalized duplicates in its own source: 0
- ... where the matched record's exact (name,address) pair recurs more than once in its own source: 0

- Negative pairs are a CONSTRUCTED, DETERMINISTIC-SEED SAMPLE built only from capped indexes of the training corpora (at most cap_per_key example records per normalized name/address, at most country_sample_cap per country). They do NOT represent the full negative population and are not a random sample of all possible non-match pairs -- see limitations.
- Positive (true match) summary: {'name_levenshtein_sim': {'count': 200000, 'min': 0.0, 'median': 0.7894736842105263, 'mean': 0.7193, 'p90': 1.0, 'p99': 1.0, 'max': 1.0}, 'addr_levenshtein_sim': {'count': 200000, 'min': 0.0, 'median': 0.7105263157894737, 'mean': 0.6389, 'p90': 0.9591836734693877, 'p99': 1.0, 'max': 1.0}, 'name_exact_norm_rate': 0.219215, 'addr_exact_norm_rate': 0.08252, 'country_equal_rate': 1.0}
- Negative category `same_country_random`: {'count_constructed': 20000, 'requested': 20000, 'skipped': False, 'name_levenshtein_sim': {'count': 20000, 'min': 0.0, 'median': 0.1923076923076923, 'mean': 0.2079, 'p90': 0.33333333333333337, 'p99': 0.5757575757575757, 'max': 0.8064516129032258}, 'addr_levenshtein_sim': {'count': 20000, 'min': 0.0, 'median': 0.20656665468918434, 'mean': 0.2036, 'p90': 0.27586206896551724, 'p99': 0.3793103448275862, 'max': 0.6451612903225806}, 'name_exact_norm_rate': 0.0, 'addr_exact_norm_rate': 0.0, 'country_equal_rate': 1.0}
- Negative category `same_name_diff_entity`: {'count_constructed': 0, 'requested': 20000, 'skipped': True, 'reason': 'SKIPPED (--skip-corpus-frequency): this category depends on name_index_capped/address_index_capped, which are only built when --skip-corpus-frequency is NOT set. A count of 0 here is a structural consequence of that flag, not evidence about the data. Re-run without --skip-corpus-frequency to get a real reading for this category.', 'name_levenshtein_sim': None, 'addr_levenshtein_sim': None, 'name_exact_norm_rate': None, 'addr_exact_norm_rate': None, 'country_equal_rate': None}
- Negative category `same_address_diff_entity`: {'count_constructed': 0, 'requested': 20000, 'skipped': True, 'reason': 'SKIPPED (--skip-corpus-frequency): this category depends on name_index_capped/address_index_capped, which are only built when --skip-corpus-frequency is NOT set. A count of 0 here is a structural consequence of that flag, not evidence about the data. Re-run without --skip-corpus-frequency to get a real reading for this category.', 'name_levenshtein_sim': None, 'addr_levenshtein_sim': None, 'name_exact_norm_rate': None, 'addr_exact_norm_rate': None, 'country_equal_rate': None}

## E. Blocking-relevant retrieval experiments
- {'rule': 'exact_normalized_business_name', 'recall': 0.219215, 'candidate_set_size_stats': 'not computed (--skip-corpus-frequency or no data)'}
- {'rule': 'first_token_of_normalized_name', 'recall': 0.72616, 'candidate_set_size_stats': 'not computed (--skip-corpus-frequency or no data)'}
- {'rule': 'country_plus_first_name_token', 'recall': 0.72616, 'candidate_set_size_stats': 'not computed (--skip-corpus-frequency or no data)'}
- {'rule': 'any_shared_address_token', 'recall': 0.95592, 'candidate_set_size_stats': 'not computed (would require a full address-token inverted index over the corpus; not built by this script -- see limitations)', 'note': 'Recall computed over 200000 sampled pairs with at least one non-empty address on both sides.'}
- Recall here means: fraction of SAMPLED true-match pairs where the rule's key computed on the Source-1 record equals the key computed on its true matched record. It is an upper-bound estimate of what a block-then-compare strategy using that key COULD retrieve, not a validated end-to-end blocking recall (it assumes a block is actually searched for that key across the whole source file, which this script does not simulate exhaustively).

## F. Region/country observations
- **US**: {'present_in_sampled_training_pairs': True, 'sampled_pair_count': 119851, 'name_levenshtein_sim': {'count': 119851, 'min': 0.0, 'median': 0.8333333333333334, 'mean': 0.7836, 'p90': 1.0, 'p99': 1.0, 'max': 1.0}, 'addr_levenshtein_sim': {'count': 119851, 'min': 0.0, 'median': 0.7096774193548387, 'mean': 0.64, 'p90': 0.962962962962963, 'p99': 1.0, 'max': 1.0}, 'name_exact_norm_rate': 0.259338678859584}
- **India**: {'present_in_sampled_training_pairs': True, 'sampled_pair_count': 80149, 'name_levenshtein_sim': {'count': 80149, 'min': 0.0, 'median': 0.7037037037037037, 'mean': 0.6232, 'p90': 1.0, 'p99': 1.0, 'max': 1.0}, 'addr_levenshtein_sim': {'count': 80149, 'min': 0.0, 'median': 0.7142857142857143, 'mean': 0.6373, 'p90': 0.9558823529411765, 'p99': 1.0, 'max': 1.0}, 'name_exact_norm_rate': 0.15921596027398968}
- **France**: {'present_in_sampled_training_pairs': False, 'note': "'France' did not appear (by exact string match) in the sampled training pairs' country fields. Not fabricating a breakdown for it. Note this checks exact string equality only -- alternate spellings/codes are not normalized against an external country list."}

## G. Ambiguities and limitations
- Similarity statistics are based on a sample of 200000 of 7638365 total true-match pairs (seed=1337).
- Negative examples are a constructed, capped, deterministic sample -- not a random sample of the true negative population, and not proof of what a model would see across all non-matches.
- Blocking-rule 'recall' estimates assume a rule's key matches exactly between the two sides; actual blocking implementation details (index construction, multi-key blocking, sorted-neighborhood, etc.) are not simulated here.
- Corpus-wide frequency/ambiguity sections (C, D partially) were SKIPPED because --skip-corpus-frequency was set.
- Section D's `same_name_diff_entity` and `same_address_diff_entity` negative categories are ALSO structurally forced to 0 by --skip-corpus-frequency (see the `skipped`/`reason` fields on those entries) -- do not read those zeros as a finding about the data. `same_country_random` is unaffected by this flag.
- Country-field comparisons use exact string equality only; no external country-name normalization or geocoding was used, so distinct strings for the same country (e.g. different codes or spellings) are NOT reconciled and would show as 'not equal'.
- Address token overlap ('any_shared_address_token' blocking rule) reports recall only; candidate-set-size statistics for that rule were not computed because doing so would require a full address-token inverted index over the entire corpus, which this script does not build (see script docstring for the memory rationale).

## H. Recommended questions for the next modelling stage
- Given the observed name/address similarity distributions above, which combination of thresholds on Levenshtein/Jaccard maximizes candidate recall at an acceptable candidate-set size for full-scale blocking (not just the sampled estimate here)?
- How should the pipeline handle the empty-address rate observed among true matches (see section B) -- does address similarity need a distinct 'missing' feature state rather than being scored as dissimilar?
- Given the observed rate of true matches with cross-country labels (section B), should country equality be used as a hard blocking filter or only as a soft feature?
- Given the singleton diagnostics in section C, how should the model or thresholds be tuned to avoid false positives on singletons with duplicate names/addresses elsewhere in S2/S3, given the macro-F0.5 reward for correctly predicting 'no match'?
- Should the final blocking strategy combine multiple keys (e.g. country + name token AND address token overlap) given that no single rule here reached both high recall and a small candidate set in this sampled analysis?
- Does the ambiguity observed in section D (duplicate names/addresses among true matches themselves) suggest that name+address alone will be insufficient and additional fields/features are needed?
