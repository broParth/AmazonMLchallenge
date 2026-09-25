# Dataset Profile Report

- Generated (UTC): 2026-09-25T05:51:29.920890+00:00
- Search root: `D:\Documents\PARTH\AmazonML\AmazonMLchallenge`
- This report reflects only what was read from disk on the machine where the script was run. No dataset file was modified, renamed, moved, or deleted. No ML, matching, or cleaning was performed.

## 1. Confirmed issues (integrity violations actually observed)
- None observed by this script.

## 2. Observed statistics (neutral facts, not judgments)
- train_source1.tsv: 2206821 data rows, 2206821 unique entity_id, 2 distinct country values
- train_source2.tsv: 5034616 data rows, 5034616 unique entity_id, 2 distinct country values
- train_source2.tsv: 764608 business_name value(s) contain non-ASCII characters (expected due to transliteration; not treated as an error)
- train_source3.tsv: 5285603 data rows, 5285603 unique entity_id, 2 distinct country values
- train_source3.tsv: 606737 business_name value(s) contain non-ASCII characters (expected due to transliteration; not treated as an error)
- test_source1.tsv: 1732544 data rows, 1732544 unique entity_id, 3 distinct country values
- test_source1.tsv: 40789 business_name value(s) contain non-ASCII characters (expected due to transliteration; not treated as an error)
- test_source2.tsv: 4887273 data rows, 4887273 unique entity_id, 3 distinct country values
- test_source2.tsv: 928158 business_name value(s) contain non-ASCII characters (expected due to transliteration; not treated as an error)
- test_source3.tsv: 5082316 data rows, 5082316 unique entity_id, 3 distinct country values
- test_source3.tsv: 737515 business_name value(s) contain non-ASCII characters (expected due to transliteration; not treated as an error)
- train_ground_truth.tsv: 2206821 data rows, 123247 row(s) with an empty match list
- source1: 'France' present in test country values: True
- source2: 'France' present in test country values: True
- source3: 'France' present in test country values: True

## 3. Items requiring human/technical review
- train_source1.tsv: 177793 exact-duplicate business_name value(s) and 40089 exact-duplicate business_address value(s) observed (raw string duplicates only -- NOT entity matching; may or may not indicate the same real-world business)
- train_source2.tsv: 239779 exact-duplicate business_name value(s) and 421472 exact-duplicate business_address value(s) observed (raw string duplicates only -- NOT entity matching; may or may not indicate the same real-world business)
- train_source3.tsv: 258276 exact-duplicate business_name value(s) and 382890 exact-duplicate business_address value(s) observed (raw string duplicates only -- NOT entity matching; may or may not indicate the same real-world business)
- test_source1.tsv: 129955 exact-duplicate business_name value(s) and 31396 exact-duplicate business_address value(s) observed (raw string duplicates only -- NOT entity matching; may or may not indicate the same real-world business)
- test_source2.tsv: 223103 exact-duplicate business_name value(s) and 444698 exact-duplicate business_address value(s) observed (raw string duplicates only -- NOT entity matching; may or may not indicate the same real-world business)
- test_source3.tsv: 238156 exact-duplicate business_name value(s) and 404756 exact-duplicate business_address value(s) observed (raw string duplicates only -- NOT entity matching; may or may not indicate the same real-world business)
- source1: countries appearing in test but not in train: ['France']
- source2: countries appearing in test but not in train: ['France']
- source3: countries appearing in test but not in train: ['France']

## Per-file detail
### `train_source1.tsv`
- Path: `D:\Documents\PARTH\AmazonML\AmazonMLchallenge\student_resource\dataset\train\train_source1.tsv`
- Header: `['entity_id', 'business_name', 'business_address', 'country']`
- Data rows: 2206821
- Blank rows: 0
- Missing expected columns: none
- entity_id: total=2206821, empty=0, unique=2206821, duplicate_values=0, duplicate_extra_rows=0, prefix_violations=0
- business_name length stats: {'count': 2206821, 'min': 3, 'max': 105, 'mean': 24.03}
- business_address length stats: {'count': 2206821, 'min': 11, 'max': 256, 'mean': 52.07}
- business_name: leading/trailing-whitespace=0, non_ascii_count=0
- business_address: leading/trailing-whitespace=0, non_ascii_count=554
- Duplicate business_name values: 177793 (extra rows: 667592) -- raw string duplicates only
- Duplicate business_address values: 40089 (extra rows: 76215) -- raw string duplicates only
- Duplicate (name, address) pairs: 0 (extra rows: 0) -- raw string duplicates only
- Distinct country values: 2
- Country missing/empty count: 0
- Top country values (up to 25): [{'value': 'US', 'count': 1323633}, {'value': 'India', 'count': 883188}]
- Per-column empty counts: {'entity_id': 0, 'business_name': 0, 'business_address': 0, 'country': 0}
- Per-column whitespace-only counts: {'entity_id': 0, 'business_name': 0, 'business_address': 0, 'country': 0}

### `train_source2.tsv`
- Path: `D:\Documents\PARTH\AmazonML\AmazonMLchallenge\student_resource\dataset\train\train_source2.tsv`
- Header: `['entity_id', 'business_name', 'business_address', 'country']`
- Data rows: 5034616
- Blank rows: 0
- Missing expected columns: none
- entity_id: total=5034616, empty=0, unique=5034616, duplicate_values=0, duplicate_extra_rows=0, prefix_violations=0
- business_name length stats: {'count': 5034616, 'min': 2, 'max': 104, 'mean': 25.1}
- business_address length stats: {'count': 5034616, 'min': 0, 'max': 249, 'mean': 46.23}
- business_name: leading/trailing-whitespace=0, non_ascii_count=764608
- business_address: leading/trailing-whitespace=0, non_ascii_count=478453
- Duplicate business_name values: 239779 (extra rows: 632607) -- raw string duplicates only
- Duplicate business_address values: 421472 (extra rows: 528388) -- raw string duplicates only
- Duplicate (name, address) pairs: 24875 (extra rows: 25688) -- raw string duplicates only
- Distinct country values: 2
- Country missing/empty count: 0
- Top country values (up to 25): [{'value': 'US', 'count': 3016817}, {'value': 'India', 'count': 2017799}]
- Per-column empty counts: {'entity_id': 0, 'business_name': 0, 'business_address': 168967, 'country': 0}
- Per-column whitespace-only counts: {'entity_id': 0, 'business_name': 0, 'business_address': 0, 'country': 0}

### `train_source3.tsv`
- Path: `D:\Documents\PARTH\AmazonML\AmazonMLchallenge\student_resource\dataset\train\train_source3.tsv`
- Header: `['entity_id', 'business_name', 'business_address', 'country']`
- Data rows: 5285603
- Blank rows: 0
- Missing expected columns: none
- entity_id: total=5285603, empty=0, unique=5285603, duplicate_values=0, duplicate_extra_rows=0, prefix_violations=0
- business_name length stats: {'count': 5285603, 'min': 2, 'max': 123, 'mean': 25.2}
- business_address length stats: {'count': 5285603, 'min': 0, 'max': 240, 'mean': 46.71}
- business_name: leading/trailing-whitespace=0, non_ascii_count=606737
- business_address: leading/trailing-whitespace=0, non_ascii_count=476588
- Duplicate business_name values: 258276 (extra rows: 633994) -- raw string duplicates only
- Duplicate business_address values: 382890 (extra rows: 476923) -- raw string duplicates only
- Duplicate (name, address) pairs: 18162 (extra rows: 18641) -- raw string duplicates only
- Distinct country values: 2
- Country missing/empty count: 0
- Top country values (up to 25): [{'value': 'US', 'count': 3170056}, {'value': 'India', 'count': 2115547}]
- Per-column empty counts: {'entity_id': 0, 'business_name': 0, 'business_address': 175916, 'country': 0}
- Per-column whitespace-only counts: {'entity_id': 0, 'business_name': 0, 'business_address': 0, 'country': 0}

### `test_source1.tsv`
- Path: `D:\Documents\PARTH\AmazonML\AmazonMLchallenge\student_resource\dataset\test\test_source1.tsv`
- Header: `['entity_id', 'business_name', 'business_address', 'country']`
- Data rows: 1732544
- Blank rows: 0
- Missing expected columns: none
- entity_id: total=1732544, empty=0, unique=1732544, duplicate_values=0, duplicate_extra_rows=0, prefix_violations=0
- business_name length stats: {'count': 1732544, 'min': 3, 'max': 92, 'mean': 23.84}
- business_address length stats: {'count': 1732544, 'min': 11, 'max': 268, 'mean': 57.21}
- business_name: leading/trailing-whitespace=0, non_ascii_count=40789
- business_address: leading/trailing-whitespace=0, non_ascii_count=73800
- Duplicate business_name values: 129955 (extra rows: 493677) -- raw string duplicates only
- Duplicate business_address values: 31396 (extra rows: 55061) -- raw string duplicates only
- Duplicate (name, address) pairs: 0 (extra rows: 0) -- raw string duplicates only
- Distinct country values: 3
- Country missing/empty count: 0
- Top country values (up to 25): [{'value': 'India', 'count': 809986}, {'value': 'US', 'count': 663106}, {'value': 'France', 'count': 259452}]
- Per-column empty counts: {'entity_id': 0, 'business_name': 0, 'business_address': 0, 'country': 0}
- Per-column whitespace-only counts: {'entity_id': 0, 'business_name': 0, 'business_address': 0, 'country': 0}

### `test_source2.tsv`
- Path: `D:\Documents\PARTH\AmazonML\AmazonMLchallenge\student_resource\dataset\test\test_source2.tsv`
- Header: `['entity_id', 'business_name', 'business_address', 'country']`
- Data rows: 4887273
- Blank rows: 0
- Missing expected columns: none
- entity_id: total=4887273, empty=0, unique=4887273, duplicate_values=0, duplicate_extra_rows=0, prefix_violations=0
- business_name length stats: {'count': 4887273, 'min': 2, 'max': 102, 'mean': 25.7}
- business_address length stats: {'count': 4887273, 'min': 0, 'max': 269, 'mean': 50.41}
- business_name: leading/trailing-whitespace=0, non_ascii_count=928158
- business_address: leading/trailing-whitespace=0, non_ascii_count=720665
- Duplicate business_name values: 223103 (extra rows: 576232) -- raw string duplicates only
- Duplicate business_address values: 444698 (extra rows: 533082) -- raw string duplicates only
- Duplicate (name, address) pairs: 21737 (extra rows: 22468) -- raw string duplicates only
- Distinct country values: 3
- Country missing/empty count: 0
- Top country values (up to 25): [{'value': 'India', 'count': 2312565}, {'value': 'US', 'count': 1871330}, {'value': 'France', 'count': 703378}]
- Per-column empty counts: {'entity_id': 0, 'business_name': 0, 'business_address': 129408, 'country': 0}
- Per-column whitespace-only counts: {'entity_id': 0, 'business_name': 0, 'business_address': 0, 'country': 0}

### `test_source3.tsv`
- Path: `D:\Documents\PARTH\AmazonML\AmazonMLchallenge\student_resource\dataset\test\test_source3.tsv`
- Header: `['entity_id', 'business_name', 'business_address', 'country']`
- Data rows: 5082316
- Blank rows: 0
- Missing expected columns: none
- entity_id: total=5082316, empty=0, unique=5082316, duplicate_values=0, duplicate_extra_rows=0, prefix_violations=0
- business_name length stats: {'count': 5082316, 'min': 2, 'max': 103, 'mean': 25.66}
- business_address length stats: {'count': 5082316, 'min': 0, 'max': 267, 'mean': 48.74}
- business_name: leading/trailing-whitespace=0, non_ascii_count=737515
- business_address: leading/trailing-whitespace=0, non_ascii_count=729222
- Duplicate business_name values: 238156 (extra rows: 560387) -- raw string duplicates only
- Duplicate business_address values: 404756 (extra rows: 489783) -- raw string duplicates only
- Duplicate (name, address) pairs: 15680 (extra rows: 16110) -- raw string duplicates only
- Distinct country values: 3
- Country missing/empty count: 0
- Top country values (up to 25): [{'value': 'India', 'count': 2405000}, {'value': 'US', 'count': 1945701}, {'value': 'France', 'count': 731615}]
- Per-column empty counts: {'entity_id': 0, 'business_name': 0, 'business_address': 136098, 'country': 0}
- Per-column whitespace-only counts: {'entity_id': 0, 'business_name': 0, 'business_address': 0, 'country': 0}

### `train_ground_truth.tsv`
- Path: `D:\Documents\PARTH\AmazonML\AmazonMLchallenge\student_resource\dataset\train\train_ground_truth.tsv`
- Header: `['source1_entity_id', 'matched_entity_ids']`
- id column used: source1_entity_id
- match column used: matched_entity_ids
- Data rows: 2206821
- Blank rows: 0
- source1 id: total=2206821, empty=0, duplicate_values=0, duplicate_extra_rows=0
- Singleton rows (empty match list): 123247
- Match-count distribution (num_matches -> num_rows): {'0': 123247, '1': 119157, '2': 375212, '3': 530841, '4': 484115, '5': 321957, '6': 164868, '7': 63968, '8': 18680, '9': 4205, '10': 534, '11': 37}
- Referential integrity: unknown_source1_id=0, unknown_matched_id=0, s1_id_used_as_matched=0

## Explicit non-actions taken
- No file was modified, renamed, moved, or deleted.
- No ML, feature engineering, blocking, matching, training, or prediction was performed.
- Exact-string duplicate counts are reported as raw counts only and are NOT interpreted as entity matches.
- No internet or external data/API lookup was performed.
- No column was assumed present; missing expected columns are reported, not worked around silently.
