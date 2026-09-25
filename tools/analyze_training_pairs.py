#!/usr/bin/env python3
"""
analyze_training_pairs.py

Read-only ANALYSIS tool for the Amazon ML Challenge 2026 "Business Entity
Resolution" training data. Produces evidence to inform later pipeline
design (blocking, features, model, thresholds). This script does NOT
train a model, does NOT perform final entity matching, and does NOT
generate test predictions.

This script has NOT been executed against the real dataset. It only runs
when copied into your project and executed on the machine where the
dataset lives. Nothing under reports/training_pair_analysis.* is real
until you run it there and open those files yourself.

SCALE STRATEGY (read this before running)
-------------------------------------------
train_source1.tsv, train_source2.tsv and train_source3.tsv can hold
millions of rows each. This script never builds a Source1 x Source2 x
Source3 cartesian product and never loads all three files into one
DataFrame. Instead it:

  1. Streams train_ground_truth.tsv once to learn, per Source-1 entity,
     which Source-2/Source-3 ids are true matches (by ID prefix: "S2-"
     matches go to Source 2, "S3-" to Source 3). While streaming it also
     draws a FIXED-SIZE, SEED-DETERMINISTIC reservoir sample of match
     PAIRS (default cap: --max-pairs-for-similarity) -- string-similarity
     work in sections 2-5 runs only on this sample, not on every pair,
     because pure-Python Levenshtein over millions of pairs is not
     practical. The report always states the sample size vs. the true
     total.
  2. Streams train_source1.tsv once into an in-memory dict keyed by
     entity_id (id -> name, address, country). With ~2-3M rows this is a
     genuine memory cost, but it is not a copy of the raw file and not a
     DataFrame -- it is the minimum needed to look up any Source-1 record
     by id without re-reading the file repeatedly.
  3. Streams train_source2.tsv and train_source3.tsv ONCE EACH. During
     that single pass each script:
       - keeps full records ONLY for ids referenced by the ground truth
         (a bounded set, not the whole file),
       - keeps normalized-name / normalized-address FREQUENCY COUNTS
         (integer counts only, not row lists) for every record in the
         file, used purely for duplicate/ambiguity diagnostics,
       - keeps a CAPPED inverted index (at most --cap-per-key example
         full records per normalized name / address / blocking key) used
         only to pull a handful of concrete examples and to support
         controlled negative-pair sampling,
       - keeps a CAPPED per-country sample (at most --country-sample-cap
         example records per country) used only for random same-country
         negative sampling.
     None of this stores the full file in memory; all of it is bounded
     by the number of DISTINCT keys or by the explicit caps, not by the
     row count.

  Even so, on a corpus of 5M+ rows the frequency counters can still use a
  meaningful amount of RAM (potentially multiple GB) because business
  names/addresses are often highly distinct. Pass --skip-corpus-frequency
  if this is impractical on your machine; the sections that depend on it
  will be explicitly marked "SKIPPED (--skip-corpus-frequency)" in the
  report rather than silently omitted.

WHAT THIS SCRIPT DOES NOT DO
-----------------------------
- Never modifies, renames, moves, or deletes any raw dataset file.
- Never creates a modified copy of the dataset.
- Performs no ML training, no final matching, no test predictions.
- Uses no external data, dictionaries, geocoders, or APIs -- all string
  normalization is purely algorithmic (Unicode NFKC + casefold + strip
  punctuation + collapse whitespace) and is spelled out in NORMALIZE_*.
- Does not silently repair malformed rows: any source/ground-truth row
  whose tab-field count does not match its header is skipped and counted,
  never reinterpreted.
- Does not claim negatives sampled here represent the full negative
  population, and does not declare a duplicate name/address to be an
  actual entity match.
- Does not choose a final blocking rule, model, or threshold. All
  thresholds used to bucket "strong/weak" similarity are labeled as
  analytical bins only.

USAGE (Windows PowerShell)
---------------------------
    cd D:\\Documents\\PARTH\\AmazonMLchallenge
    python tools\\analyze_training_pairs.py

Useful flags (all optional; defaults chosen to be practical on a laptop
-- widen them once you know your machine can handle it):
    --root PATH                      project root to search from (default:
                                      parent of this script's "tools" dir)
    --encoding NAME                  default: utf-8
    --seed N                         RNG seed for all sampling (default: 1337)
    --max-pairs-for-similarity N     reservoir size for true-match pairs
                                      used in sections 2-5 and 9 (default: 200000)
    --negative-sample-size N         number of negative pairs to attempt to
                                      construct per category in section 10
                                      (default: 20000)
    --cap-per-key N                  max example records kept per
                                      normalized-name/address/blocking key
                                      (default: 5)
    --country-sample-cap N           max example records kept per country
                                      for random negative sampling (default: 2000)
    --top-n-values N                 rows shown in top-N frequency tables
                                      (default: 25)
    --max-examples N                 max example ids/strings listed per
                                      finding (default: 50)
    --skip-corpus-frequency          disable full-corpus name/address
                                      frequency counting (sections 6/7 will
                                      be marked skipped) -- use this if RAM
                                      is limited
    --skip-tfidf                     disable the optional TF-IDF cosine
                                      check even if scikit-learn is installed
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

SCRIPT_VERSION = "1.0.0"

TRAIN_FILES = {
    "train_source1.tsv": "1",
    "train_source2.tsv": "2",
    "train_source3.tsv": "3",
}
GROUND_TRUTH_FILE = "train_ground_truth.tsv"
ALL_EXPECTED_FILES = list(TRAIN_FILES.keys()) + [GROUND_TRUTH_FILE]

EXPECTED_SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_ID_COL_CANDIDATES = ["source1_entity_id", "source1_id", "entity_id_source1"]
GT_MATCH_COL_CANDIDATES = ["matched_entity_ids", "matched_ids", "matches", "matched_entity_id"]

SKIP_DIR_NAMES = {
    ".git", ".hg", ".svn", "__pycache__", "node_modules",
    ".venv", "venv", "env", ".mypy_cache", ".pytest_cache",
    ".idea", ".vscode",
}

# Analytical similarity bins used ONLY to bucket observed pairs for
# reporting in section 5. These are NOT final model thresholds.
NAME_STRONG_THRESHOLD = 0.85
NAME_WEAK_THRESHOLD = 0.50
ADDR_STRONG_THRESHOLD = 0.80
ADDR_WEAK_THRESHOLD = 0.40


# ---------------------------------------------------------------------------
# Purely algorithmic normalization / similarity (no external data)
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)


def normalize_text(s: str) -> str:
    """Unicode NFKC normalize, casefold, strip punctuation, collapse
    whitespace. Purely algorithmic -- no external dictionaries or lookups."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.casefold()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def first_token(s_normalized: str) -> str:
    parts = s_normalized.split(" ")
    return parts[0] if parts and parts[0] != "" else ""


def levenshtein_distance(a: str, b: str) -> int:
    """Classic O(len(a)*len(b)) DP Levenshtein distance. Fine for the short
    strings involved (business names/addresses), not used on full corpus,
    only on the sampled pairs."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def levenshtein_similarity(a: str, b: str) -> float:
    if a == "" and b == "":
        return 1.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(a, b)
    return 1.0 - (dist / max_len)


def token_jaccard(a: str, b: str) -> Optional[float]:
    ta, tb = set(a.split(" ")) - {""}, set(b.split(" ")) - {""}
    if not ta and not tb:
        return None  # both empty after normalization -- not comparable
    union = ta | tb
    if not union:
        return None
    inter = ta & tb
    return len(inter) / len(union)


# ---------------------------------------------------------------------------
# Percentile helper (no numpy dependency required)
# ---------------------------------------------------------------------------

def percentile(sorted_values: list, pct: float) -> Optional[float]:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    k = (len(sorted_values) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_values[int(k)])
    d0 = sorted_values[int(f)] * (c - k)
    d1 = sorted_values[int(c)] * (k - f)
    return float(d0 + d1)


def distribution_summary(values: list) -> dict:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None,
                "p90": None, "p99": None, "max": None}
    values_sorted = sorted(values)
    n = len(values_sorted)
    return {
        "count": n,
        "min": values_sorted[0],
        "median": percentile(values_sorted, 50),
        "mean": round(sum(values_sorted) / n, 4),
        "p90": percentile(values_sorted, 90),
        "p99": percentile(values_sorted, 99),
        "max": values_sorted[-1],
    }


# ---------------------------------------------------------------------------
# Filesystem discovery
# ---------------------------------------------------------------------------

def _walk_skipping(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        yield dirpath, dirnames, filenames


def find_expected_files(root: Path) -> dict:
    found = {name: None for name in ALL_EXPECTED_FILES}
    for dirpath, _dirnames, filenames in _walk_skipping(root):
        for name in filenames:
            if name in found and found[name] is None:
                found[name] = Path(dirpath) / name
    return found


def col_index(header: list, name: str) -> Optional[int]:
    try:
        return header.index(name)
    except ValueError:
        return None


def find_gt_column(header: list, candidates: list) -> tuple:
    for c in candidates:
        idx = col_index(header, c)
        if idx is not None:
            return idx, c, None
    return None, None, (
        f"No column matching any of {candidates} found in header {header}."
    )


# ---------------------------------------------------------------------------
# Pass A: stream ground truth -> match structure + reservoir-sampled pairs
# ---------------------------------------------------------------------------

@dataclass
class GTPassResult:
    read_error: Optional[str] = None
    header: Optional[list] = None
    id_column_used: Optional[str] = None
    match_column_used: Optional[str] = None
    column_detection_note: Optional[str] = None

    n_rows: int = 0
    n_s1_entities_with_valid_row: int = 0
    singleton_ids: list = field(default_factory=list)       # capped list for reporting
    singleton_count: int = 0
    match_count_distribution: dict = field(default_factory=dict)

    s2_only_count: int = 0
    s3_only_count: int = 0
    both_count: int = 0

    unclassified_prefix_count: int = 0
    unclassified_prefix_examples: list = field(default_factory=list)

    referenced_s2_ids: set = field(default_factory=set, repr=False)
    referenced_s3_ids: set = field(default_factory=set, repr=False)

    # reservoir of (s1_id, matched_id, matched_source) tuples
    pair_reservoir: list = field(default_factory=list, repr=False)
    total_pairs_seen: int = 0


def pass_a_ground_truth(path: Path, encoding: str, max_pairs: int, rng: random.Random,
                          max_examples: int, max_singleton_examples: int) -> GTPassResult:
    res = GTPassResult()
    header = None
    header_len = None
    id_idx = match_idx = None
    match_dist = Counter()
    pairs_seen = 0

    try:
        with open(path, "r", encoding=encoding, newline="") as f:
            for raw_line in f:
                line = raw_line.rstrip("\r\n")
                if line == "":
                    continue
                fields = line.split("\t")
                if header is None:
                    header = fields
                    header_len = len(fields)
                    res.header = header
                    id_idx, id_name, id_note = find_gt_column(header, GT_ID_COL_CANDIDATES)
                    match_idx, match_name, match_note = find_gt_column(header, GT_MATCH_COL_CANDIDATES)
                    res.id_column_used = id_name
                    res.match_column_used = match_name
                    notes = [n for n in (id_note, match_note) if n]
                    res.column_detection_note = " | ".join(notes) if notes else None
                    continue
                if len(fields) != header_len:
                    continue
                if id_idx is None:
                    continue

                res.n_rows += 1
                s1_id = fields[id_idx]
                if s1_id == "":
                    continue
                res.n_s1_entities_with_valid_row += 1

                matched_ids = []
                if match_idx is not None:
                    raw_match = fields[match_idx]
                    matched_ids = [m.strip() for m in raw_match.split(",") if m.strip() != ""]

                n_matches = len(matched_ids)
                match_dist[n_matches] += 1

                if n_matches == 0:
                    res.singleton_count += 1
                    if len(res.singleton_ids) < max_singleton_examples:
                        res.singleton_ids.append(s1_id)
                    continue

                has_s2 = has_s3 = False
                for mid in matched_ids:
                    if mid.startswith("S2-"):
                        has_s2 = True
                        res.referenced_s2_ids.add(mid)
                        matched_source = "S2"
                    elif mid.startswith("S3-"):
                        has_s3 = True
                        res.referenced_s3_ids.add(mid)
                        matched_source = "S3"
                    else:
                        res.unclassified_prefix_count += 1
                        if len(res.unclassified_prefix_examples) < max_examples:
                            res.unclassified_prefix_examples.append(mid)
                        continue

                    pairs_seen += 1
                    # Reservoir sampling (Algorithm R), deterministic via rng.
                    if len(res.pair_reservoir) < max_pairs:
                        res.pair_reservoir.append((s1_id, mid, matched_source))
                    else:
                        j = rng.randint(0, pairs_seen - 1)
                        if j < max_pairs:
                            res.pair_reservoir[j] = (s1_id, mid, matched_source)

                if has_s2 and has_s3:
                    res.both_count += 1
                elif has_s2:
                    res.s2_only_count += 1
                elif has_s3:
                    res.s3_only_count += 1

    except UnicodeDecodeError as e:
        res.read_error = f"UnicodeDecodeError with encoding='{encoding}': {e}"
        return res
    except OSError as e:
        res.read_error = f"OS error reading file: {e}"
        return res

    res.match_count_distribution = {str(k): v for k, v in sorted(match_dist.items())}
    res.total_pairs_seen = pairs_seen
    return res


# ---------------------------------------------------------------------------
# Pass B: stream train_source1.tsv -> full id -> record dict
# ---------------------------------------------------------------------------

def pass_b_source1(path: Path, encoding: str) -> tuple:
    """Returns (records_dict, header, row_count, read_error)."""
    records = {}
    header = None
    header_len = None
    idx = {}
    row_count = 0
    try:
        with open(path, "r", encoding=encoding, newline="") as f:
            for raw_line in f:
                line = raw_line.rstrip("\r\n")
                if line == "":
                    continue
                fields = line.split("\t")
                if header is None:
                    header = fields
                    header_len = len(fields)
                    idx["entity_id"] = col_index(header, "entity_id")
                    idx["business_name"] = col_index(header, "business_name")
                    idx["business_address"] = col_index(header, "business_address")
                    idx["country"] = col_index(header, "country")
                    continue
                if len(fields) != header_len or idx.get("entity_id") is None:
                    continue
                eid = fields[idx["entity_id"]]
                if eid == "":
                    continue
                name = fields[idx["business_name"]] if idx.get("business_name") is not None else ""
                addr = fields[idx["business_address"]] if idx.get("business_address") is not None else ""
                country = fields[idx["country"]] if idx.get("country") is not None else ""
                records[eid] = (name, addr, country)
                row_count += 1
    except UnicodeDecodeError as e:
        return records, header, row_count, f"UnicodeDecodeError with encoding='{encoding}': {e}"
    except OSError as e:
        return records, header, row_count, f"OS error reading file: {e}"
    return records, header, row_count, None


# ---------------------------------------------------------------------------
# Pass C/D: stream train_source2/3.tsv -> bounded referenced-record dict,
# full-corpus frequency counters, capped inverted indexes, capped country
# samples. Single pass per file.
# ---------------------------------------------------------------------------

@dataclass
class SourceCorpusResult:
    header: Optional[list] = None
    row_count: int = 0
    read_error: Optional[str] = None
    referenced_records: dict = field(default_factory=dict, repr=False)  # id -> (name, addr, country)
    name_freq: Counter = field(default_factory=Counter, repr=False)
    address_freq: Counter = field(default_factory=Counter, repr=False)
    first_token_freq: Counter = field(default_factory=Counter, repr=False)          # blocking rule: first name token
    country_plus_token_freq: Counter = field(default_factory=Counter, repr=False)   # blocking rule: country + first token
    name_index_capped: dict = field(default_factory=lambda: defaultdict(list), repr=False)
    address_index_capped: dict = field(default_factory=lambda: defaultdict(list), repr=False)
    country_sample_capped: dict = field(default_factory=lambda: defaultdict(list), repr=False)
    total_records_considered: int = 0  # rows with non-empty entity_id, used as denominator


def pass_source_corpus(path: Path, encoding: str, referenced_ids: set,
                        cap_per_key: int, country_sample_cap: int,
                        skip_frequency: bool) -> SourceCorpusResult:
    res = SourceCorpusResult()
    header = None
    header_len = None
    idx = {}

    try:
        with open(path, "r", encoding=encoding, newline="") as f:
            for raw_line in f:
                line = raw_line.rstrip("\r\n")
                if line == "":
                    continue
                fields = line.split("\t")
                if header is None:
                    header = fields
                    header_len = len(fields)
                    res.header = header
                    idx["entity_id"] = col_index(header, "entity_id")
                    idx["business_name"] = col_index(header, "business_name")
                    idx["business_address"] = col_index(header, "business_address")
                    idx["country"] = col_index(header, "country")
                    continue
                if len(fields) != header_len or idx.get("entity_id") is None:
                    continue

                eid = fields[idx["entity_id"]]
                if eid == "":
                    continue

                name = fields[idx["business_name"]] if idx.get("business_name") is not None else ""
                addr = fields[idx["business_address"]] if idx.get("business_address") is not None else ""
                country = fields[idx["country"]] if idx.get("country") is not None else ""

                res.row_count += 1
                res.total_records_considered += 1
                record = (eid, name, addr, country)

                if eid in referenced_ids:
                    res.referenced_records[eid] = (name, addr, country)

                norm_name = normalize_text(name)
                norm_addr = normalize_text(addr)

                if not skip_frequency:
                    if norm_name != "":
                        res.name_freq[norm_name] += 1
                    if norm_addr != "":
                        res.address_freq[norm_addr] += 1

                    tok = first_token(norm_name)
                    if tok != "":
                        res.first_token_freq[tok] += 1
                        if country != "":
                            res.country_plus_token_freq[(country, tok)] += 1

                    if norm_name != "" and len(res.name_index_capped[norm_name]) < cap_per_key:
                        res.name_index_capped[norm_name].append(record)
                    if norm_addr != "" and len(res.address_index_capped[norm_addr]) < cap_per_key:
                        res.address_index_capped[norm_addr].append(record)

                if country != "" and len(res.country_sample_capped[country]) < country_sample_cap:
                    res.country_sample_capped[country].append(record)

    except UnicodeDecodeError as e:
        res.read_error = f"UnicodeDecodeError with encoding='{encoding}': {e}"
        return res
    except OSError as e:
        res.read_error = f"OS error reading file: {e}"
        return res

    return res


# ---------------------------------------------------------------------------
# Main analysis orchestration
# ---------------------------------------------------------------------------

def get_record(s1_records: dict, s2_corpus: SourceCorpusResult, s3_corpus: SourceCorpusResult,
                entity_id: str, source_hint: Optional[str] = None) -> Optional[tuple]:
    if source_hint == "S1" or entity_id.startswith("S1-"):
        return s1_records.get(entity_id)
    if source_hint == "S2" or entity_id.startswith("S2-"):
        return s2_corpus.referenced_records.get(entity_id)
    if source_hint == "S3" or entity_id.startswith("S3-"):
        return s3_corpus.referenced_records.get(entity_id)
    return None


def build_pair_metrics(s1_rec: tuple, m_rec: tuple) -> dict:
    s1_name, s1_addr, s1_country = s1_rec
    m_name, m_addr, m_country = m_rec

    n1, n2 = normalize_text(s1_name), normalize_text(m_name)
    a1, a2 = normalize_text(s1_addr), normalize_text(m_addr)

    name_exact_raw = (s1_name == m_name) and (s1_name != "")
    name_exact_norm = (n1 == n2) and (n1 != "")
    addr_exact_raw = (s1_addr == m_addr) and (s1_addr != "")
    addr_exact_norm = (a1 == a2) and (a1 != "")

    name_lev = levenshtein_similarity(n1, n2) if (n1 or n2) else None
    addr_lev = levenshtein_similarity(a1, a2) if (a1 or a2) else None
    name_jac = token_jaccard(n1, n2)
    addr_jac = token_jaccard(a1, a2)

    both_addr_present = (s1_addr.strip() != "") and (m_addr.strip() != "")
    either_addr_empty = (s1_addr.strip() == "") or (m_addr.strip() == "")

    country_equal = (s1_country == m_country) if (s1_country != "" and m_country != "") else None

    return {
        "name_exact_raw": name_exact_raw,
        "name_exact_norm": name_exact_norm,
        "addr_exact_raw": addr_exact_raw,
        "addr_exact_norm": addr_exact_norm,
        "name_levenshtein_sim": name_lev,
        "addr_levenshtein_sim": addr_lev,
        "name_jaccard": name_jac,
        "addr_jaccard": addr_jac,
        "either_addr_empty": either_addr_empty,
        "both_addr_present": both_addr_present,
        "country_equal": country_equal,
        "s1_country": s1_country,
        "m_country": m_country,
        "norm_s1_name": n1,
        "norm_m_name": n2,
        "norm_s1_addr": a1,
        "norm_m_addr": a2,
        "first_token_s1": first_token(n1),
        "first_token_m": first_token(n2),
    }


def classify_evidence_bucket(m: dict) -> str:
    name_sim = m["name_levenshtein_sim"] if m["name_levenshtein_sim"] is not None else 0.0
    addr_sim = m["addr_levenshtein_sim"] if m["addr_levenshtein_sim"] is not None else 0.0

    if m["name_exact_norm"]:
        name_bucket = "exact"
    elif name_sim >= NAME_STRONG_THRESHOLD:
        name_bucket = "strong"
    elif name_sim >= NAME_WEAK_THRESHOLD:
        name_bucket = "weak"
    else:
        name_bucket = "dissimilar"

    if m["either_addr_empty"]:
        addr_bucket = "empty"
    elif m["addr_exact_norm"]:
        addr_bucket = "exact"
    elif addr_sim >= ADDR_STRONG_THRESHOLD:
        addr_bucket = "strong"
    elif addr_sim >= ADDR_WEAK_THRESHOLD:
        addr_bucket = "weak"
    else:
        addr_bucket = "dissimilar"

    return f"name={name_bucket}/address={addr_bucket}"


def try_tfidf_cosine(pairs_text: list, skip: bool) -> tuple:
    """pairs_text: list of (text_a, text_b). Returns (list_of_cosine_or_None, note)."""
    if skip:
        return None, "TF-IDF cosine skipped (--skip-tfidf)."
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return None, "TF-IDF cosine skipped: scikit-learn is not installed in this environment."

    if not pairs_text:
        return None, "TF-IDF cosine skipped: no pairs available."

    corpus = []
    for a, b in pairs_text:
        corpus.append(a)
        corpus.append(b)
    try:
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(corpus)
    except ValueError as e:
        return None, f"TF-IDF cosine skipped: vectorizer error ({e}); likely all-empty input strings."

    sims = []
    for i in range(0, matrix.shape[0], 2):
        sim = cosine_similarity(matrix[i], matrix[i + 1])[0][0]
        sims.append(float(sim))
    return sims, f"TF-IDF cosine computed over {len(sims)} sampled pairs (character/word default TfidfVectorizer)."


def sample_negative_for_pair(rng: random.Random, s1_id: str, s1_rec: tuple, true_match_id: str,
                              m_corpus: SourceCorpusResult) -> dict:
    """Construct up to three categories of negative candidate for one
    positive pair, using only the capped indexes already built. Returns a
    dict with whichever categories could be constructed (missing keys mean
    "could not construct one for this positive pair", not a zero result)."""
    s1_name, s1_addr, s1_country = s1_rec
    out = {}

    # (a) random same-country non-match
    bucket = m_corpus.country_sample_capped.get(s1_country, [])
    candidates = [r for r in bucket if r[0] != true_match_id]
    if candidates:
        out["same_country_random"] = rng.choice(candidates)

    # (b) same normalized name, different entity id
    norm_name = normalize_text(s1_name)
    bucket = m_corpus.name_index_capped.get(norm_name, [])
    candidates = [r for r in bucket if r[0] != true_match_id]
    if candidates:
        out["same_name_diff_entity"] = rng.choice(candidates)

    # (c) same normalized address, different entity id
    norm_addr = normalize_text(s1_addr)
    bucket = m_corpus.address_index_capped.get(norm_addr, [])
    candidates = [r for r in bucket if r[0] != true_match_id]
    if candidates:
        out["same_address_diff_entity"] = rng.choice(candidates)

    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only analysis of TRAINING true-match pairs for the "
                     "Amazon ML Challenge 2026 Business Entity Resolution dataset."
    )
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--encoding", type=str, default="utf-8")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-pairs-for-similarity", type=int, default=200_000)
    parser.add_argument("--negative-sample-size", type=int, default=20_000)
    parser.add_argument("--cap-per-key", type=int, default=5)
    parser.add_argument("--country-sample-cap", type=int, default=2000)
    parser.add_argument("--top-n-values", type=int, default=25)
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--skip-corpus-frequency", action="store_true")
    parser.add_argument("--skip-tfidf", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    script_path = Path(__file__).resolve()
    if args.root:
        root = Path(args.root).resolve()
    elif script_path.parent.name.lower() == "tools":
        root = script_path.parent.parent
    else:
        root = script_path.parent
        print(f"[WARN] Script not inside a 'tools' folder; using '{root}' as root.", file=sys.stderr)

    if not root.exists():
        print(f"[ERROR] Search root does not exist: {root}", file=sys.stderr)
        return 2

    print(f"[INFO] Searching for training files under: {root}")
    expected = find_expected_files(root)
    for name, p in expected.items():
        print(f"[{'INFO' if p else 'WARN'}] {name}: {p if p else 'NOT FOUND'}")

    missing = [n for n, p in expected.items() if p is None]
    if missing:
        print(
            f"[ERROR] Missing required training file(s): {missing}. "
            f"Cannot proceed with pair analysis.",
            file=sys.stderr,
        )
        report = {
            "audit_metadata": {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "script": "tools/analyze_training_pairs.py",
                "script_version": SCRIPT_VERSION,
                "search_root": str(root.resolve()),
                "fatal_error": f"Missing required file(s): {missing}",
            }
        }
        _write_reports(root, report, f"# Training Pair Analysis Report\n\nFATAL: missing required file(s): {missing}\n")
        return 3

    print("[INFO] Pass A: streaming train_ground_truth.tsv ...")
    gt = pass_a_ground_truth(
        expected[GROUND_TRUTH_FILE], args.encoding, args.max_pairs_for_similarity, rng,
        args.max_examples, args.max_examples,
    )
    if gt.read_error:
        print(f"[ERROR] ground truth: {gt.read_error}", file=sys.stderr)
        report = {"audit_metadata": {"fatal_error": gt.read_error}}
        _write_reports(root, report, f"# Training Pair Analysis Report\n\nFATAL: {gt.read_error}\n")
        return 4
    print(
        f"[INFO]   -> rows={gt.n_rows} singleton={gt.singleton_count} "
        f"s2_only={gt.s2_only_count} s3_only={gt.s3_only_count} both={gt.both_count} "
        f"pairs_seen={gt.total_pairs_seen} pairs_sampled={len(gt.pair_reservoir)}"
    )

    print("[INFO] Pass B: streaming train_source1.tsv ...")
    s1_records, s1_header, s1_rowcount, s1_err = pass_b_source1(expected["train_source1.tsv"], args.encoding)
    if s1_err:
        print(f"[ERROR] train_source1.tsv: {s1_err}", file=sys.stderr)
    else:
        print(f"[INFO]   -> {s1_rowcount} Source-1 records indexed")

    print("[INFO] Pass C: streaming train_source2.tsv ...")
    s2_corpus = pass_source_corpus(
        expected["train_source2.tsv"], args.encoding, gt.referenced_s2_ids,
        args.cap_per_key, args.country_sample_cap, args.skip_corpus_frequency,
    )
    if s2_corpus.read_error:
        print(f"[ERROR] train_source2.tsv: {s2_corpus.read_error}", file=sys.stderr)
    else:
        print(f"[INFO]   -> {s2_corpus.row_count} rows scanned, {len(s2_corpus.referenced_records)} referenced records kept")

    print("[INFO] Pass D: streaming train_source3.tsv ...")
    s3_corpus = pass_source_corpus(
        expected["train_source3.tsv"], args.encoding, gt.referenced_s3_ids,
        args.cap_per_key, args.country_sample_cap, args.skip_corpus_frequency,
    )
    if s3_corpus.read_error:
        print(f"[ERROR] train_source3.tsv: {s3_corpus.read_error}", file=sys.stderr)
    else:
        print(f"[INFO]   -> {s3_corpus.row_count} rows scanned, {len(s3_corpus.referenced_records)} referenced records kept")

    # --- Section 2-5: true-match pair metrics over the sampled reservoir ---
    print(f"[INFO] Computing similarity metrics over {len(gt.pair_reservoir)} sampled true-match pairs ...")
    pair_metrics = []
    unresolvable_pairs = 0
    for s1_id, m_id, m_source in gt.pair_reservoir:
        s1_rec = s1_records.get(s1_id)
        m_corpus = s2_corpus if m_source == "S2" else s3_corpus
        m_rec = m_corpus.referenced_records.get(m_id)
        if s1_rec is None or m_rec is None:
            unresolvable_pairs += 1
            continue
        metrics = build_pair_metrics(s1_rec, m_rec)
        metrics["s1_id"] = s1_id
        metrics["matched_id"] = m_id
        metrics["matched_source"] = m_source
        metrics["evidence_bucket"] = classify_evidence_bucket(metrics)
        pair_metrics.append(metrics)

    name_lev_values = [m["name_levenshtein_sim"] for m in pair_metrics if m["name_levenshtein_sim"] is not None]
    addr_lev_values = [m["addr_levenshtein_sim"] for m in pair_metrics if m["addr_levenshtein_sim"] is not None]
    name_jac_values = [m["name_jaccard"] for m in pair_metrics if m["name_jaccard"] is not None]
    addr_jac_values = [m["addr_jaccard"] for m in pair_metrics if m["addr_jaccard"] is not None]

    name_exact_raw_rate = _rate(pair_metrics, "name_exact_raw")
    name_exact_norm_rate = _rate(pair_metrics, "name_exact_norm")
    addr_exact_raw_rate = _rate(pair_metrics, "addr_exact_raw")
    addr_exact_norm_rate = _rate(pair_metrics, "addr_exact_norm")
    either_addr_empty_rate = _rate(pair_metrics, "either_addr_empty")

    country_equal_known = [m["country_equal"] for m in pair_metrics if m["country_equal"] is not None]
    country_cross_count = sum(1 for v in country_equal_known if v is False)
    country_equal_rate = (sum(1 for v in country_equal_known if v) / len(country_equal_known)) if country_equal_known else None

    evidence_bucket_counts = Counter(m["evidence_bucket"] for m in pair_metrics)

    tfidf_note = None
    tfidf_summary = None
    if not args.skip_tfidf:
        tfidf_sample_size = min(2000, len(pair_metrics))
        sample_for_tfidf = pair_metrics[:tfidf_sample_size]
        pairs_text = [(m["norm_s1_name"], m["norm_m_name"]) for m in sample_for_tfidf]
        sims, note = try_tfidf_cosine(pairs_text, skip=False)
        tfidf_note = note
        if sims:
            tfidf_summary = distribution_summary(sims)
    else:
        tfidf_note = "TF-IDF cosine skipped (--skip-tfidf)."

    # --- Section 6: singleton diagnostics ---
    print("[INFO] Running singleton diagnostics ...")
    singleton_sample_size = min(len(gt.singleton_ids), args.negative_sample_size)
    singleton_sample = gt.singleton_ids[:singleton_sample_size]
    singleton_diag = {
        "singleton_total": gt.singleton_count,
        "singleton_examples_analyzed": singleton_sample_size,
        "note": (
            f"Diagnostics below run on {singleton_sample_size} of {gt.singleton_count} "
            f"singleton entities (capped example list from Pass A, not a fresh random "
            f"sample of the full singleton population)."
            if gt.singleton_count > singleton_sample_size else
            f"All {gt.singleton_count} singleton entities were available for this check "
            f"(subject to the --max-examples cap on Pass A's stored id list)."
        ),
        "has_exact_name_dup_in_s2_or_s3": 0,
        "has_exact_address_dup_in_s2_or_s3": 0,
    }
    if not args.skip_corpus_frequency:
        for sid in singleton_sample:
            rec = s1_records.get(sid)
            if not rec:
                continue
            name, addr, _country = rec
            nn, na = normalize_text(name), normalize_text(addr)
            name_dup = (s2_corpus.name_freq.get(nn, 0) > 0) or (s3_corpus.name_freq.get(nn, 0) > 0)
            addr_dup = (s2_corpus.address_freq.get(na, 0) > 0) or (s3_corpus.address_freq.get(na, 0) > 0)
            if name_dup and nn != "":
                singleton_diag["has_exact_name_dup_in_s2_or_s3"] += 1
            if addr_dup and na != "":
                singleton_diag["has_exact_address_dup_in_s2_or_s3"] += 1
    else:
        singleton_diag["note"] += " Frequency-based checks SKIPPED (--skip-corpus-frequency)."

    # --- Section 7: positive-match duplicate/ambiguity ---
    print("[INFO] Running positive-match ambiguity diagnostics ...")
    positive_ambiguity = {"note": None, "name_ambiguous_count": 0, "address_ambiguous_count": 0,
                           "name_address_ambiguous_count": 0, "pairs_checked": 0}
    if not args.skip_corpus_frequency:
        na_pair_counts = Counter()
        for m in pair_metrics:
            positive_ambiguity["pairs_checked"] += 1
            m_corpus = s2_corpus if m["matched_source"] == "S2" else s3_corpus
            if m["norm_m_name"] != "" and m_corpus.name_freq.get(m["norm_m_name"], 0) > 1:
                positive_ambiguity["name_ambiguous_count"] += 1
            if m["norm_m_addr"] != "" and m_corpus.address_freq.get(m["norm_m_addr"], 0) > 1:
                positive_ambiguity["address_ambiguous_count"] += 1
            na_pair_counts[(m["matched_source"], m["norm_m_name"], m["norm_m_addr"])] += 1
        positive_ambiguity["name_address_ambiguous_count"] = sum(1 for c in na_pair_counts.values() if c > 1)
        positive_ambiguity["note"] = (
            "Counts reflect exact normalized-string duplication within the matched source's "
            "corpus-wide frequency counters; this is NOT an entity-matching judgment."
        )
    else:
        positive_ambiguity["note"] = "Skipped (--skip-corpus-frequency)."

    # --- Section 9: blocking-relevant retrieval experiments ---
    print("[INFO] Running blocking-relevant retrieval experiments ...")
    blocking_rules_report = []
    if pair_metrics:
        # Rule 1: exact normalized name
        recall, cand_stats = _evaluate_blocking_rule(
            pair_metrics, s2_corpus, s3_corpus,
            key_fn=lambda m: m["norm_s1_name"],
            key_match_fn=lambda m: m["norm_s1_name"] == m["norm_m_name"] and m["norm_s1_name"] != "",
            freq_lookup=lambda src, key: src.name_freq.get(key, 0),
            skip_frequency=args.skip_corpus_frequency,
        )
        blocking_rules_report.append({"rule": "exact_normalized_business_name", "recall": recall,
                                       "candidate_set_size_stats": cand_stats})

        # Rule 2: first token of normalized name
        recall, cand_stats = _evaluate_blocking_rule(
            pair_metrics, s2_corpus, s3_corpus,
            key_fn=lambda m: m["first_token_s1"],
            key_match_fn=lambda m: m["first_token_s1"] == m["first_token_m"] and m["first_token_s1"] != "",
            freq_lookup=lambda src, key: src.first_token_freq.get(key, 0),
            skip_frequency=args.skip_corpus_frequency,
        )
        blocking_rules_report.append({"rule": "first_token_of_normalized_name", "recall": recall,
                                       "candidate_set_size_stats": cand_stats})

        # Rule 3: country + first token
        recall, cand_stats = _evaluate_blocking_rule(
            pair_metrics, s2_corpus, s3_corpus,
            key_fn=lambda m: (m["s1_country"], m["first_token_s1"]),
            key_match_fn=lambda m: (
                m["first_token_s1"] == m["first_token_m"] and m["first_token_s1"] != ""
                and m["s1_country"] == m["m_country"] and m["s1_country"] != ""
            ),
            freq_lookup=lambda src, key: src.country_plus_token_freq.get(key, 0),
            skip_frequency=args.skip_corpus_frequency,
        )
        blocking_rules_report.append({"rule": "country_plus_first_name_token", "recall": recall,
                                       "candidate_set_size_stats": cand_stats})

        # Rule 4: address token overlap (any shared token, Jaccard > 0)
        addr_overlap_matches = sum(
            1 for m in pair_metrics
            if m["addr_jaccard"] is not None and m["addr_jaccard"] > 0
        )
        addr_overlap_denominator = sum(1 for m in pair_metrics if m["addr_jaccard"] is not None)
        addr_overlap_recall = (addr_overlap_matches / addr_overlap_denominator) if addr_overlap_denominator else None
        blocking_rules_report.append({
            "rule": "any_shared_address_token",
            "recall": addr_overlap_recall,
            "candidate_set_size_stats": "not computed (would require a full address-token inverted "
                                          "index over the corpus; not built by this script -- see limitations)",
            "note": f"Recall computed over {addr_overlap_denominator} sampled pairs with at least one "
                    f"non-empty address on both sides.",
        })
    else:
        blocking_rules_report.append({"note": "No resolvable sampled pairs available; blocking analysis skipped."})

    # --- Section 10: negative sampling ---
    print("[INFO] Constructing controlled negative samples ...")
    negatives = {"same_country_random": [], "same_name_diff_entity": [], "same_address_diff_entity": []}
    neg_target = min(args.negative_sample_size, len(pair_metrics))
    sample_for_negatives = pair_metrics[:neg_target]
    for m in sample_for_negatives:
        s1_rec = s1_records.get(m["s1_id"])
        if not s1_rec:
            continue
        m_corpus = s2_corpus if m["matched_source"] == "S2" else s3_corpus
        neg = sample_negative_for_pair(rng, m["s1_id"], s1_rec, m["matched_id"], m_corpus)
        for key, rec in neg.items():
            if rec is None:
                continue
            neg_id, neg_name, neg_addr, neg_country = rec
            neg_metrics = build_pair_metrics(s1_rec, (neg_name, neg_addr, neg_country))
            neg_metrics["s1_id"] = m["s1_id"]
            neg_metrics["negative_id"] = neg_id
            negatives[key].append(neg_metrics)

    # Categories that can only be constructed from the capped name/address
    # inverted indexes built in pass_source_corpus(). Those indexes are only
    # populated when skip_corpus_frequency is False (see pass_source_corpus).
    # When the flag is set, these two categories are STRUCTURALLY forced to
    # zero regardless of the actual data -- that must be reported explicitly
    # so a zero here is never mistaken for a genuine "no duplicates" finding.
    CATEGORIES_REQUIRING_FREQUENCY_INDEX = {"same_name_diff_entity", "same_address_diff_entity"}

    negative_summaries = {}
    for cat, items in negatives.items():
        if cat in CATEGORIES_REQUIRING_FREQUENCY_INDEX and args.skip_corpus_frequency:
            negative_summaries[cat] = {
                "count_constructed": 0,
                "requested": neg_target,
                "skipped": True,
                "reason": (
                    "SKIPPED (--skip-corpus-frequency): this category depends on "
                    "name_index_capped/address_index_capped, which are only built when "
                    "--skip-corpus-frequency is NOT set. A count of 0 here is a structural "
                    "consequence of that flag, not evidence about the data. Re-run without "
                    "--skip-corpus-frequency to get a real reading for this category."
                ),
                "name_levenshtein_sim": None,
                "addr_levenshtein_sim": None,
                "name_exact_norm_rate": None,
                "addr_exact_norm_rate": None,
                "country_equal_rate": None,
            }
            continue
        negative_summaries[cat] = {
            "count_constructed": len(items),
            "requested": neg_target,
            "skipped": False,
            "name_levenshtein_sim": distribution_summary([i["name_levenshtein_sim"] for i in items if i["name_levenshtein_sim"] is not None]),
            "addr_levenshtein_sim": distribution_summary([i["addr_levenshtein_sim"] for i in items if i["addr_levenshtein_sim"] is not None]),
            "name_exact_norm_rate": _rate(items, "name_exact_norm"),
            "addr_exact_norm_rate": _rate(items, "addr_exact_norm"),
            "country_equal_rate": (
                sum(1 for i in items if i["country_equal"]) / len([i for i in items if i["country_equal"] is not None])
                if any(i["country_equal"] is not None for i in items) else None
            ),
        }

    positive_summary_for_compare = {
        "name_levenshtein_sim": distribution_summary(name_lev_values),
        "addr_levenshtein_sim": distribution_summary(addr_lev_values),
        "name_exact_norm_rate": name_exact_norm_rate,
        "addr_exact_norm_rate": addr_exact_norm_rate,
        "country_equal_rate": country_equal_rate,
    }

    # --- Section 8: region-specific ---
    print("[INFO] Running region-specific breakdown ...")
    training_countries_seen = set(m["s1_country"] for m in pair_metrics if m["s1_country"]) | \
                                set(m["m_country"] for m in pair_metrics if m["m_country"])
    region_report = {}
    for region_label, country_name in (("US", "US"), ("India", "India"), ("France", "France")):
        subset = [m for m in pair_metrics if m["s1_country"] == country_name or m["m_country"] == country_name]
        if country_name not in training_countries_seen and not subset:
            region_report[region_label] = {
                "present_in_sampled_training_pairs": False,
                "note": f"'{country_name}' did not appear (by exact string match) in the sampled "
                        f"training pairs' country fields. Not fabricating a breakdown for it. "
                        f"Note this checks exact string equality only -- alternate spellings/codes "
                        f"are not normalized against an external country list.",
            }
        else:
            region_report[region_label] = {
                "present_in_sampled_training_pairs": True,
                "sampled_pair_count": len(subset),
                "name_levenshtein_sim": distribution_summary([m["name_levenshtein_sim"] for m in subset if m["name_levenshtein_sim"] is not None]),
                "addr_levenshtein_sim": distribution_summary([m["addr_levenshtein_sim"] for m in subset if m["addr_levenshtein_sim"] is not None]),
                "name_exact_norm_rate": _rate(subset, "name_exact_norm"),
            }

    # --- Assemble report ---
    json_report = {
        "audit_metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "script": "tools/analyze_training_pairs.py",
            "script_version": SCRIPT_VERSION,
            "search_root": str(root.resolve()),
            "random_seed": args.seed,
            "sampling_used": True,
            "max_pairs_for_similarity": args.max_pairs_for_similarity,
            "negative_sample_size_requested": args.negative_sample_size,
            "cap_per_key": args.cap_per_key,
            "country_sample_cap": args.country_sample_cap,
            "skip_corpus_frequency": args.skip_corpus_frequency,
            "skip_tfidf": args.skip_tfidf,
            "note": (
                "This report reflects only what this script read from disk on the machine "
                "where it was executed. No dataset file was modified. True-match similarity "
                "statistics are computed on a fixed-size, seeded reservoir sample of match "
                "pairs, not the full population, when the population exceeds "
                "max_pairs_for_similarity -- see 'pair_sampling' below for exact counts."
            ),
        },
        "pair_sampling": {
            "total_true_match_pairs_in_ground_truth": gt.total_pairs_seen,
            "pairs_sampled_for_similarity_analysis": len(gt.pair_reservoir),
            "pairs_resolvable_to_records": len(pair_metrics),
            "pairs_unresolvable": unresolvable_pairs,
        },
        "section_1_ground_truth_structure": {
            "n_ground_truth_rows": gt.n_rows,
            "n_s1_entities_with_valid_row": gt.n_s1_entities_with_valid_row,
            "singleton_count": gt.singleton_count,
            "match_count_distribution": gt.match_count_distribution,
            "s2_only_count": gt.s2_only_count,
            "s3_only_count": gt.s3_only_count,
            "both_s2_and_s3_count": gt.both_count,
            "unclassified_matched_id_prefix_count": gt.unclassified_prefix_count,
            "unclassified_matched_id_prefix_examples": gt.unclassified_prefix_examples,
            "column_detection_note": gt.column_detection_note,
        },
        "section_2_3_4_true_match_similarity": {
            "name_exact_raw_rate": name_exact_raw_rate,
            "name_exact_normalized_rate": name_exact_norm_rate,
            "name_levenshtein_similarity": distribution_summary(name_lev_values),
            "name_jaccard": distribution_summary(name_jac_values),
            "name_tfidf_cosine": tfidf_summary,
            "name_tfidf_note": tfidf_note,
            "address_exact_raw_rate": addr_exact_raw_rate,
            "address_exact_normalized_rate": addr_exact_norm_rate,
            "address_levenshtein_similarity": distribution_summary(addr_lev_values),
            "address_jaccard": distribution_summary(addr_jac_values),
            "either_address_empty_rate": either_addr_empty_rate,
            "country_equal_rate_where_both_known": country_equal_rate,
            "country_cross_label_true_match_count": country_cross_count,
            "country_equality_known_pair_count": len(country_equal_known),
        },
        "section_5_evidence_buckets": dict(evidence_bucket_counts),
        "section_5_thresholds_used_analytical_only": {
            "name_strong_threshold": NAME_STRONG_THRESHOLD,
            "name_weak_threshold": NAME_WEAK_THRESHOLD,
            "address_strong_threshold": ADDR_STRONG_THRESHOLD,
            "address_weak_threshold": ADDR_WEAK_THRESHOLD,
            "note": "These are analytical bins chosen to summarize observed data for this "
                    "report only. They are NOT proposed as final model thresholds.",
        },
        "section_6_singleton_diagnostics": singleton_diag,
        "section_7_positive_match_ambiguity": positive_ambiguity,
        "section_8_region_specific": region_report,
        "section_9_blocking_experiments": blocking_rules_report,
        "section_10_positive_vs_negative": {
            "positive": positive_summary_for_compare,
            "negatives": negative_summaries,
            "note": (
                "Negative pairs are a CONSTRUCTED, DETERMINISTIC-SEED SAMPLE built only from "
                "capped indexes of the training corpora (at most cap_per_key example records "
                "per normalized name/address, at most country_sample_cap per country). They do "
                "NOT represent the full negative population and are not a random sample of all "
                "possible non-match pairs -- see limitations."
            ),
        },
    }

    md_report = _build_markdown(root, args, gt, s1_rowcount, s2_corpus, s3_corpus, json_report)

    _write_reports(root, json_report, md_report)
    print("[INFO] Analysis complete. No files were modified. No ML training, matching, or "
          "predictions were performed. Review the reports before designing the pipeline.")
    return 0


def _rate(items: list, key: str) -> Optional[float]:
    if not items:
        return None
    return sum(1 for i in items if i.get(key)) / len(items)


def _evaluate_blocking_rule(pair_metrics: list, s2_corpus: SourceCorpusResult, s3_corpus: SourceCorpusResult,
                             key_fn, key_match_fn, freq_lookup, skip_frequency: bool) -> tuple:
    denom = 0
    hits = 0
    candidate_sizes = []
    for m in pair_metrics:
        key = key_fn(m)
        if key in (None, "", ("", "")):
            continue
        denom += 1
        if key_match_fn(m):
            hits += 1
        if not skip_frequency:
            m_corpus = s2_corpus if m["matched_source"] == "S2" else s3_corpus
            size = freq_lookup(m_corpus, key_fn(m))
            candidate_sizes.append(size)
    recall = (hits / denom) if denom else None
    cand_stats = distribution_summary(candidate_sizes) if candidate_sizes else "not computed (--skip-corpus-frequency or no data)"
    return recall, cand_stats


def _write_reports(root: Path, json_report: dict, md_report: str) -> None:
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "training_pair_analysis.json"
    md_path = reports_dir / "training_pair_analysis.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2, default=str)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)
    print(f"[INFO] Wrote: {json_path}")
    print(f"[INFO] Wrote: {md_path}")


def _build_markdown(root: Path, args, gt: GTPassResult, s1_rowcount: int,
                     s2_corpus: SourceCorpusResult, s3_corpus: SourceCorpusResult,
                     j: dict) -> str:
    md = []
    md.append("# Training Pair Analysis Report")
    md.append("")
    md.append(f"- Generated (UTC): {j['audit_metadata']['generated_at_utc']}")
    md.append(f"- Search root: `{root.resolve()}`")
    md.append(f"- Script version: {SCRIPT_VERSION}, seed: {args.seed}")
    md.append(
        "- No raw dataset file was modified. No ML training, final matching, or test "
        "predictions were performed. This is an evidence-gathering report only."
    )
    md.append("")

    md.append("## Sampling disclosure")
    ps = j["pair_sampling"]
    md.append(
        f"- Total true-match pairs in ground truth: {ps['total_true_match_pairs_in_ground_truth']}"
    )
    md.append(f"- Pairs sampled for similarity analysis (seeded reservoir sample): {ps['pairs_sampled_for_similarity_analysis']}")
    md.append(f"- Sampled pairs resolvable to actual records: {ps['pairs_resolvable_to_records']}")
    md.append(f"- Sampled pairs NOT resolvable (record missing from referenced sets): {ps['pairs_unresolvable']}")
    if ps['total_true_match_pairs_in_ground_truth'] > ps['pairs_sampled_for_similarity_analysis']:
        md.append(
            "- **All similarity statistics below are estimates from a sample, not the full "
            "population.** Increase `--max-pairs-for-similarity` for a larger sample if your "
            "machine can handle the runtime."
        )
    md.append("")

    md.append("## A. Dataset facts actually observed")
    s1s = j["section_1_ground_truth_structure"]
    md.append(f"- train_source1.tsv rows indexed: {s1_rowcount}")
    md.append(f"- train_source2.tsv rows scanned: {s2_corpus.row_count}, referenced records kept: {len(s2_corpus.referenced_records)}")
    md.append(f"- train_source3.tsv rows scanned: {s3_corpus.row_count}, referenced records kept: {len(s3_corpus.referenced_records)}")
    md.append(f"- Ground-truth rows: {s1s['n_ground_truth_rows']}")
    md.append(f"- Singleton Source-1 entities: {s1s['singleton_count']}")
    md.append(f"- Match-count distribution (num_matches -> num_S1_entities): {s1s['match_count_distribution']}")
    md.append(f"- S2-only matches: {s1s['s2_only_count']}, S3-only: {s1s['s3_only_count']}, both: {s1s['both_s2_and_s3_count']}")
    if s1s["unclassified_matched_id_prefix_count"]:
        md.append(
            f"- **{s1s['unclassified_matched_id_prefix_count']} matched id(s) did not start with "
            f"'S2-' or 'S3-'** and were excluded from source-specific analysis. Examples: "
            f"{s1s['unclassified_matched_id_prefix_examples']}"
        )
    if s1s["column_detection_note"]:
        md.append(f"- Ground-truth column detection note: {s1s['column_detection_note']}")
    md.append("")

    md.append("## B. True-match characteristics")
    s234 = j["section_2_3_4_true_match_similarity"]
    md.append(f"- Exact raw name equality rate: {s234['name_exact_raw_rate']}")
    md.append(f"- Exact normalized name equality rate: {s234['name_exact_normalized_rate']}")
    md.append(f"- Name Levenshtein similarity distribution: {s234['name_levenshtein_similarity']}")
    md.append(f"- Name Jaccard (token overlap) distribution: {s234['name_jaccard']}")
    if s234["name_tfidf_cosine"]:
        md.append(f"- Name TF-IDF cosine distribution (sampled subset): {s234['name_tfidf_cosine']}")
    md.append(f"- TF-IDF note: {s234['name_tfidf_note']}")
    md.append(f"- Exact raw address equality rate: {s234['address_exact_raw_rate']}")
    md.append(f"- Exact normalized address equality rate: {s234['address_exact_normalized_rate']}")
    md.append(f"- Address Levenshtein similarity distribution: {s234['address_levenshtein_similarity']}")
    md.append(f"- Address Jaccard distribution: {s234['address_jaccard']}")
    md.append(f"- Rate of true matches with at least one empty address: {s234['either_address_empty_rate']}")
    md.append(
        f"- Country equality rate among true matches where both countries are known: "
        f"{s234['country_equal_rate_where_both_known']} "
        f"(n={s234['country_equality_known_pair_count']})"
    )
    md.append(
        f"- True matches with DIFFERENT country labels on each side: "
        f"{s234['country_cross_label_true_match_count']} -- country equality alone should not "
        f"be assumed to guarantee identity, and this dataset shows why."
    )
    md.append("")
    md.append("### Evidence buckets (name/address strength combinations, analytical bins only)")
    for bucket, count in j["section_5_evidence_buckets"].items():
        md.append(f"- `{bucket}`: {count}")
    th = j["section_5_thresholds_used_analytical_only"]
    md.append(
        f"- Thresholds used for binning (NOT final model thresholds): name strong >= "
        f"{th['name_strong_threshold']}, name weak >= {th['name_weak_threshold']}; "
        f"address strong >= {th['address_strong_threshold']}, address weak >= {th['address_weak_threshold']}."
    )
    md.append("")

    md.append("## C. Singleton characteristics")
    sd = j["section_6_singleton_diagnostics"]
    md.append(f"- Total singleton Source-1 entities: {sd['singleton_total']}")
    md.append(f"- Singleton examples analyzed for duplicate-name/address diagnostics: {sd['singleton_examples_analyzed']}")
    md.append(f"- {sd['note']}")
    md.append(f"- Of those, entities whose name has an exact-normalized duplicate somewhere in S2/S3: {sd['has_exact_name_dup_in_s2_or_s3']}")
    md.append(f"- Of those, entities whose address has an exact-normalized duplicate somewhere in S2/S3: {sd['has_exact_address_dup_in_s2_or_s3']}")
    md.append(
        "- **These are diagnostics only.** A name/address duplicate does NOT mean the singleton "
        "actually has a hidden match -- correctly predicting a singleton as unmatched is rewarded "
        "under macro F0.5, and this section exists only to flag where that could be *harder*."
    )
    md.append("")

    md.append("## D. Positive-vs-negative similarity observations")
    pa = j["section_7_positive_match_ambiguity"]
    md.append(f"- {pa['note']}")
    md.append(f"- True-match pairs checked: {pa['pairs_checked']}")
    md.append(f"- ... where the matched record's name has other exact-normalized duplicates in its own source: {pa['name_ambiguous_count']}")
    md.append(f"- ... where the matched record's address has other exact-normalized duplicates in its own source: {pa['address_ambiguous_count']}")
    md.append(f"- ... where the matched record's exact (name,address) pair recurs more than once in its own source: {pa['name_address_ambiguous_count']}")
    md.append("")
    pn = j["section_10_positive_vs_negative"]
    md.append(f"- {pn['note']}")
    md.append(f"- Positive (true match) summary: {pn['positive']}")
    for cat, summ in pn["negatives"].items():
        md.append(f"- Negative category `{cat}`: {summ}")
    md.append("")

    md.append("## E. Blocking-relevant retrieval experiments")
    for rule in j["section_9_blocking_experiments"]:
        md.append(f"- {rule}")
    md.append(
        "- Recall here means: fraction of SAMPLED true-match pairs where the rule's key computed "
        "on the Source-1 record equals the key computed on its true matched record. It is an "
        "upper-bound estimate of what a block-then-compare strategy using that key COULD retrieve, "
        "not a validated end-to-end blocking recall (it assumes a block is actually searched for "
        "that key across the whole source file, which this script does not simulate exhaustively)."
    )
    md.append("")

    md.append("## F. Region/country observations")
    for region, info in j["section_8_region_specific"].items():
        md.append(f"- **{region}**: {info}")
    md.append("")

    md.append("## G. Ambiguities and limitations")
    md.append(f"- Similarity statistics are based on a sample of {ps['pairs_sampled_for_similarity_analysis']} of "
               f"{ps['total_true_match_pairs_in_ground_truth']} total true-match pairs (seed={args.seed}).")
    md.append(
        "- Negative examples are a constructed, capped, deterministic sample -- not a random "
        "sample of the true negative population, and not proof of what a model would see across "
        "all non-matches."
    )
    md.append(
        "- Blocking-rule 'recall' estimates assume a rule's key matches exactly between the two "
        "sides; actual blocking implementation details (index construction, multi-key blocking, "
        "sorted-neighborhood, etc.) are not simulated here."
    )
    if args.skip_corpus_frequency:
        md.append("- Corpus-wide frequency/ambiguity sections (C, D partially) were SKIPPED because --skip-corpus-frequency was set.")
        md.append(
            "- Section D's `same_name_diff_entity` and `same_address_diff_entity` negative "
            "categories are ALSO structurally forced to 0 by --skip-corpus-frequency (see the "
            "`skipped`/`reason` fields on those entries) -- do not read those zeros as a finding "
            "about the data. `same_country_random` is unaffected by this flag."
        )
    else:
        md.append(
            "- If `same_name_diff_entity` and/or `same_address_diff_entity` in section D show a "
            "genuine (non-skipped) count of 0 while section D's positive-match ambiguity counts "
            "(section 7 / section D above) are clearly nonzero, treat that as worth re-checking "
            "(e.g. a very small `--cap-per-key`, or an unusually small `--negative-sample-size` "
            "relative to the data) rather than assuming it reflects the true absence of "
            "same-name/same-address non-matches in the corpus."
        )
    md.append(
        "- Country-field comparisons use exact string equality only; no external country-name "
        "normalization or geocoding was used, so distinct strings for the same country (e.g. "
        "different codes or spellings) are NOT reconciled and would show as 'not equal'."
    )
    md.append(
        "- Address token overlap ('any_shared_address_token' blocking rule) reports recall only; "
        "candidate-set-size statistics for that rule were not computed because doing so would "
        "require a full address-token inverted index over the entire corpus, which this script "
        "does not build (see script docstring for the memory rationale)."
    )
    md.append("")

    md.append("## H. Recommended questions for the next modelling stage")
    md.append("- Given the observed name/address similarity distributions above, which combination "
               "of thresholds on Levenshtein/Jaccard maximizes candidate recall at an acceptable "
               "candidate-set size for full-scale blocking (not just the sampled estimate here)?")
    md.append("- How should the pipeline handle the empty-address rate observed among true matches "
               "(see section B) -- does address similarity need a distinct 'missing' feature state "
               "rather than being scored as dissimilar?")
    md.append("- Given the observed rate of true matches with cross-country labels (section B), "
               "should country equality be used as a hard blocking filter or only as a soft feature?")
    md.append("- Given the singleton diagnostics in section C, how should the model or thresholds be "
               "tuned to avoid false positives on singletons with duplicate names/addresses elsewhere "
               "in S2/S3, given the macro-F0.5 reward for correctly predicting 'no match'?")
    md.append("- Should the final blocking strategy combine multiple keys (e.g. country + name token "
               "AND address token overlap) given that no single rule here reached both high recall "
               "and a small candidate set in this sampled analysis?")
    md.append("- Does the ambiguity observed in section D (duplicate names/addresses among true "
               "matches themselves) suggest that name+address alone will be insufficient and "
               "additional fields/features are needed?")
    md.append("")

    return "\n".join(md)


if __name__ == "__main__":
    sys.exit(main())