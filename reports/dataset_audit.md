# Dataset Audit Report

- Generated (UTC): 2026-09-25T05:10:42.445457+00:00
- Search root: `D:\Documents\PARTH\AmazonML\AmazonMLchallenge`
- This report reflects only what was read from disk on the machine where the script was run. No dataset file was modified, renamed, moved, or deleted.

## Expected files not found
- None. All expected filenames were located.

## TSV files found that do not match an expected filename
- None.

## Per-file audit
### `D:\Documents\PARTH\AmazonML\AmazonMLchallenge\student_resource\dataset\test\test_source1.tsv`
- Role (by filename): **source**
- Size: 175022086 bytes
- Encoding used: utf-8
- Header field count: 4
- Header: `['entity_id', 'business_name', 'business_address', 'country']`
- Data row count (excludes header/blank/duplicate-header lines): 1732544
- Blank row count: 0
- Duplicate header row count: 0
- Malformed row count (field count != header field count): 0
- Missing expected columns: none
- Extra/unexpected columns vs. expected list: none

### `D:\Documents\PARTH\AmazonML\AmazonMLchallenge\student_resource\dataset\test\test_source2.tsv`
- Role (by filename): **source**
- Size: 509456422 bytes
- Encoding used: utf-8
- Header field count: 4
- Header: `['entity_id', 'business_name', 'business_address', 'country']`
- Data row count (excludes header/blank/duplicate-header lines): 4887273
- Blank row count: 0
- Duplicate header row count: 0
- Malformed row count (field count != header field count): 0
- Missing expected columns: none
- Extra/unexpected columns vs. expected list: none

### `D:\Documents\PARTH\AmazonML\AmazonMLchallenge\student_resource\dataset\test\test_source3.tsv`
- Role (by filename): **source**
- Size: 506002772 bytes
- Encoding used: utf-8
- Header field count: 4
- Header: `['entity_id', 'business_name', 'business_address', 'country']`
- Data row count (excludes header/blank/duplicate-header lines): 5082316
- Blank row count: 0
- Duplicate header row count: 0
- Malformed row count (field count != header field count): 0
- Missing expected columns: none
- Extra/unexpected columns vs. expected list: none

### `D:\Documents\PARTH\AmazonML\AmazonMLchallenge\student_resource\dataset\train\train_ground_truth.tsv`
- Role (by filename): **ground_truth**
- Size: 127015583 bytes
- Encoding used: utf-8
- Header field count: 2
- Header: `['source1_entity_id', 'matched_entity_ids']`
- Data row count (excludes header/blank/duplicate-header lines): 2206821
- Blank row count: 0
- Duplicate header row count: 0
- Malformed row count (field count != header field count): 0
- Other notes:
  - Ground-truth file: header reported as-is; not compared against source-file column list because its structure is expected to differ.

### `D:\Documents\PARTH\AmazonML\AmazonMLchallenge\student_resource\dataset\train\train_source1.tsv`
- Role (by filename): **source**
- Size: 210069713 bytes
- Encoding used: utf-8
- Header field count: 4
- Header: `['entity_id', 'business_name', 'business_address', 'country']`
- Data row count (excludes header/blank/duplicate-header lines): 2206821
- Blank row count: 0
- Duplicate header row count: 0
- Malformed row count (field count != header field count): 0
- Missing expected columns: none
- Extra/unexpected columns vs. expected list: none

### `D:\Documents\PARTH\AmazonML\AmazonMLchallenge\student_resource\dataset\train\train_source2.tsv`
- Role (by filename): **source**
- Size: 489301488 bytes
- Encoding used: utf-8
- Header field count: 4
- Header: `['entity_id', 'business_name', 'business_address', 'country']`
- Data row count (excludes header/blank/duplicate-header lines): 5034616
- Blank row count: 0
- Duplicate header row count: 0
- Malformed row count (field count != header field count): 0
- Missing expected columns: none
- Extra/unexpected columns vs. expected list: none

### `D:\Documents\PARTH\AmazonML\AmazonMLchallenge\student_resource\dataset\train\train_source3.tsv`
- Role (by filename): **source**
- Size: 503705637 bytes
- Encoding used: utf-8
- Header field count: 4
- Header: `['entity_id', 'business_name', 'business_address', 'country']`
- Data row count (excludes header/blank/duplicate-header lines): 5285603
- Blank row count: 0
- Duplicate header row count: 0
- Malformed row count (field count != header field count): 0
- Missing expected columns: none
- Extra/unexpected columns vs. expected list: none

## Explicit non-actions taken
- No file was modified, renamed, moved, or deleted.
- No malformed row was repaired, dropped, or reinterpreted.
- No missing value was inferred or filled in.
- No delimiter was substituted for tabs, even when tabs were missing.
- No ML, cleaning, feature engineering, blocking, or matching was performed.
- No internet or external data lookup was performed.
