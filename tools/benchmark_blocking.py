#!/usr/bin/env python3
"""
benchmark_blocking.py

Read-only BENCHMARK tool for blocking / candidate-generation strategies on
the Amazon ML Challenge 2026 "Business Entity Resolution" TRAINING data
only. This is NOT the final blocker -- it is an empirical benchmark used to
choose/design one, run on a deterministic SAMPLE of Source-1 entities, not
the full population.

This script has NOT been executed against the real dataset. It only runs
when copied into your project (into the same tools/ directory as
analyze_training_pairs.py) and executed on the machine where the dataset
lives. Nothing under reports/blocking_benchmark.* is real until you run it
there and open those files yourself.

WHAT THIS SCRIPT DOES NOT DO
-----------------------------
- Never modifies, renames, moves, or deletes any raw dataset file.
- Never modifies any existing report or the challenge-provided utilities.
- Performs no ML training, no final matching, no test predictions.
- Uses no external data, dictionaries, geocoders, or APIs. All blocking
  keys and similarity metrics are purely algorithmic and derived only from
  the four train/* files listed below.
- Does not choose a final blocking rule or architecture. Section 11 of the
  report proposes candidate directions based only on the measured numbers
  in this run; it is not a final decision.
- Does not call any candidate set a "match" -- candidates are pre-matching-
  model records only.

NORMALIZATION REUSE (read this before changing anything)
----------------------------------------------------------
This script does NOT reimplement text normalization, similarity, or
percentile logic. It imports the following directly from
analyze_training_pairs.py, which must sit in the same tools/ directory:
    normalize_text, first_token, levenshtein_similarity, token_jaccard,
    percentile, col_index, find_gt_column, GT_ID_COL_CANDIDATES,
    GT_MATCH_COL_CANDIDATES, SKIP_DIR_NAMES
All of these were judged fully suitable for reuse as-is and are imported,
not copied, so the two scripts can never silently drift apart on what
"normalized" means. The one exception is distribution_summary(): the
existing helper in analyze_training_pairs.py reports count/min/median/
mean/p90/p99/max, but this benchmark also needs p95 (required by the
candidate-count spec below). Rather than modify the existing function (or
the file it lives in), this script adds a small local wrapper,
full_distribution_summary(), that reuses the SAME imported percentile()
routine and only adds the extra p95 field. No existing file is changed.

SAMPLING / SCALE STRATEGY
----------------------------
The full training corpus (train_source2 + train_source3) holds ~10M+ rows
and the target machine has 8 GB RAM. This script never loads Source 2 or
Source 3 into a DataFrame and never builds an inverted index over the
whole corpus. Instead:

  1. train_ground_truth.tsv is streamed ONCE. A fixed-size, seed-
     deterministic reservoir sample (Algorithm R) of --sample-size
     (default 20000) Source-1 entity rows is drawn while streaming --
     this also gives each sampled entity's ground-truth matched ids for
     free, in the same pass.
  2. train_source1.tsv is streamed ONCE. Only rows whose entity_id is in
     the sampled set are kept (bounded to --sample-size records).
  3. train_source2.tsv and train_source3.tsv are each streamed ONCE for a
     "frequency pass": this collects (a) full records for the bounded set
     of ids actually referenced as ground-truth matches by the sampled
     entities (typically a few times --sample-size, not millions), and
     (b) corpus-wide frequency counts, but ONLY for the specific blocking
     keys and address tokens that the 20000 sampled entities actually
     produce ("keys/tokens of interest") -- not for every distinct key in
     the whole corpus. This bounds Counter memory to a small multiple of
     --sample-size, not to corpus cardinality.
  4. Recall (whether a true match is retrieved by a rule) is computed
     directly from the bounded referenced-records dict from step 3 -- it
     needs no further file scan.
  5. Only candidate-SET-SIZE statistics for the token-set rules (E, the H
     rare-token variants, and any union involving them) require a second,
     bounded streaming pass over train_source2.tsv/train_source3.tsv (a
     "candidate pass"), because a token-set rule can match a single
     candidate record through more than one shared token and a naive
     sum-of-token-frequencies would double count. This pass uses small
     reverse indexes built ONLY from the 20000 sampled entities' own keys/
     tokens (bounded), never a corpus-wide index. Pass --skip-token-set-
     candidates to disable this second scan entirely (recall for those
     rules is still reported; their candidate-count stats are marked
     skipped).

Net I/O per run: train_ground_truth.tsv x1, train_source1.tsv x1,
train_source2.tsv x2, train_source3.tsv x2 (x1 each if
--skip-token-set-candidates is passed). All in sequential streaming mode.

USAGE (Windows PowerShell)
---------------------------
    cd D:\\Documents\\PARTH\\AmazonMLchallenge
    python tools\\benchmark_blocking.py

Useful flags (all optional):
    --root PATH                    project root to search from (default:
                                    parent of this script's "tools" dir)
    --encoding NAME                 default: utf-8
    --seed N                        RNG seed for the entity sample (default: 1337)
    --sample-size N                 number of Source-1 entities to sample (default: 20000)
    --rare-token-cutoffs "a,b,c"    absolute corpus document-frequency cutoffs
                                     tested for rule H (default: "50,200,1000")
    --union-h-cutoff N               which of --rare-token-cutoffs to use for the
                                     C+H union in rule I (default: 200; must be a
                                     value present in --rare-token-cutoffs)
    --negative-sample-cap N          max non-match candidate pairs to keep for the
                                     negative-similarity comparison (default: 5000)
    --skip-token-set-candidates      skip the second (candidate-count) scan for
                                     rules E / H_* / I_* -- recall for those rules
                                     is unaffected; only their candidate-count and
                                     reduction-ratio stats become unavailable
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Reuse existing normalization / similarity / percentile logic verbatim.
# See "NORMALIZATION REUSE" in the module docstring above.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from analyze_training_pairs import (
        normalize_text,
        first_token,
        levenshtein_similarity,
        token_jaccard,
        percentile,
        col_index,
        find_gt_column,
        GT_ID_COL_CANDIDATES,
        GT_MATCH_COL_CANDIDATES,
        SKIP_DIR_NAMES,
    )
except ImportError as e:
    print(
        "[FATAL] Could not import from analyze_training_pairs.py. "
        "This script must be placed in the SAME tools/ directory as "
        "analyze_training_pairs.py (it reuses that file's normalization "
        f"logic rather than reimplementing it). Original error: {e}",
        file=sys.stderr,
    )
    sys.exit(2)

SCRIPT_VERSION = "1.0.0"

TRAIN_FILE_NAMES = [
    "train_source1.tsv",
    "train_source2.tsv",
    "train_source3.tsv",
    "train_ground_truth.tsv",
]

EXPECTED_SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]

RULE_ORDER = ["A", "B", "C", "D", "E", "F", "G"]  # H_* and I_* appended dynamically


# ---------------------------------------------------------------------------
# Small local extension of the imported percentile() to add p95 (see docstring)
# ---------------------------------------------------------------------------

def full_distribution_summary(values: list) -> dict:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None,
                "p90": None, "p95": None, "p99": None, "max": None}
    values_sorted = sorted(values)
    n = len(values_sorted)
    return {
        "count": n,
        "min": values_sorted[0],
        "median": percentile(values_sorted, 50),
        "mean": round(sum(values_sorted) / n, 4),
        "p90": percentile(values_sorted, 90),
        "p95": percentile(values_sorted, 95),
        "p99": percentile(values_sorted, 99),
        "max": values_sorted[-1],
    }


# ---------------------------------------------------------------------------
# Filesystem discovery (train files only)
# ---------------------------------------------------------------------------

def find_train_files(root: Path) -> dict:
    found = {name: None for name in TRAIN_FILE_NAMES}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        for name in filenames:
            if name in found and found[name] is None:
                found[name] = Path(dirpath) / name
    return found


# ---------------------------------------------------------------------------
# Generic source-file streaming (entity_id, business_name, business_address,
# country) -- shared by train_source1/2/3.tsv across every pass.
# ---------------------------------------------------------------------------

def iter_source_rows(path: Path, encoding: str):
    """Yields (eid, name, addr, country) for each valid data row. Skips rows
    whose tab-field count does not match the header, and rows with an empty
    entity_id, without silently reinterpreting them."""
    with open(path, "r", encoding=encoding, newline="") as f:
        header = None
        header_len = None
        idx = {}
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
            yield eid, name, addr, country


# ---------------------------------------------------------------------------
# Pass 1: reservoir-sample Source-1 entities from train_ground_truth.tsv
# ---------------------------------------------------------------------------

def sample_ground_truth_entities(path: Path, encoding: str, sample_size: int, rng) -> tuple:
    """Returns (reservoir, total_gt_rows, unclassified_matched_id_count,
    column_detection_note). reservoir is a list of (s1_id, matched_ids)
    tuples, matched_ids containing only S2-/S3- prefixed ids."""
    reservoir = []
    total_rows = 0
    unclassified = 0
    seen_count = 0
    header = None
    header_len = None
    id_idx = match_idx = None
    note = None
    with open(path, "r", encoding=encoding, newline="") as f:
        for raw_line in f:
            line = raw_line.rstrip("\r\n")
            if line == "":
                continue
            fields = line.split("\t")
            if header is None:
                header = fields
                header_len = len(fields)
                id_idx, _id_name, id_note = find_gt_column(header, GT_ID_COL_CANDIDATES)
                match_idx, _match_name, match_note = find_gt_column(header, GT_MATCH_COL_CANDIDATES)
                notes = [n for n in (id_note, match_note) if n]
                note = " | ".join(notes) if notes else None
                continue
            if len(fields) != header_len or id_idx is None:
                continue
            total_rows += 1
            s1_id = fields[id_idx]
            if s1_id == "":
                continue
            matched_ids = []
            if match_idx is not None:
                raw_match = fields[match_idx]
                matched_ids = [m.strip() for m in raw_match.split(",") if m.strip() != ""]
            valid_matched = []
            for mid in matched_ids:
                if mid.startswith("S2-") or mid.startswith("S3-"):
                    valid_matched.append(mid)
                else:
                    unclassified += 1
            seen_count += 1
            entry = (s1_id, valid_matched)
            if len(reservoir) < sample_size:
                reservoir.append(entry)
            else:
                j = rng.randint(0, seen_count - 1)
                if j < sample_size:
                    reservoir[j] = entry
    return reservoir, total_rows, unclassified, note


# ---------------------------------------------------------------------------
# Pass 2: load sampled Source-1 records + derive blocking keys
# ---------------------------------------------------------------------------

@dataclass
class EntityKeys:
    eid: str
    norm_name: str
    norm_addr: str
    country: str
    tokens_addr: frozenset
    key_A: Optional[str]
    key_B: Optional[str]
    key_C: Optional[tuple]
    key_D: Optional[str]
    key_F: Optional[str]
    key_G: Optional[tuple]
    name_empty: bool
    addr_empty: bool
    lt2_name_tokens: bool


def compute_entity_keys(eid: str, name: str, addr: str, country: str) -> EntityKeys:
    norm_name = normalize_text(name)
    norm_addr = normalize_text(addr)
    tokens_name = [t for t in norm_name.split(" ") if t != ""]
    tokens_addr = frozenset(t for t in norm_addr.split(" ") if t != "")
    tok = tokens_name[0] if tokens_name else ""
    first_two = " ".join(tokens_name[:2]) if len(tokens_name) >= 2 else None
    name_empty = norm_name == ""
    addr_empty = norm_addr == ""
    key_A = norm_name if not name_empty else None
    key_B = tok if tok != "" else None
    key_C = (country, tok) if country != "" and tok != "" else None
    key_D = norm_addr if not addr_empty else None
    key_F = first_two
    key_G = (country, first_two) if country != "" and first_two is not None else None
    return EntityKeys(
        eid=eid, norm_name=norm_name, norm_addr=norm_addr, country=country,
        tokens_addr=tokens_addr, key_A=key_A, key_B=key_B, key_C=key_C,
        key_D=key_D, key_F=key_F, key_G=key_G, name_empty=name_empty,
        addr_empty=addr_empty, lt2_name_tokens=(first_two is None),
    )


def load_sampled_source1(path: Path, encoding: str, sample_ids: set) -> dict:
    records = {}
    for eid, name, addr, country in iter_source_rows(path, encoding):
        if eid in sample_ids:
            records[eid] = compute_entity_keys(eid, name, addr, country)
    return records


# ---------------------------------------------------------------------------
# Pass 3: frequency pass over train_source2.tsv / train_source3.tsv
# (bounded to keys/tokens of interest) + referenced-record capture
# ---------------------------------------------------------------------------

def run_frequency_pass(path: Path, encoding: str, keys_of_interest: dict,
                        tokens_of_interest: frozenset, referenced_ids: set,
                        freq: dict, addr_token_doc_freq: Counter,
                        referenced_records: dict) -> int:
    row_count = 0
    for eid, name, addr, country in iter_source_rows(path, encoding):
        row_count += 1
        if eid in referenced_ids:
            referenced_records[eid] = (name, addr, country)
        norm_name = normalize_text(name)
        norm_addr = normalize_text(addr)
        tokens_name = [t for t in norm_name.split(" ") if t != ""]
        tok = tokens_name[0] if tokens_name else ""
        first_two = " ".join(tokens_name[:2]) if len(tokens_name) >= 2 else None
        key_A = norm_name if norm_name != "" else None
        key_B = tok if tok != "" else None
        key_C = (country, tok) if country != "" and tok != "" else None
        key_D = norm_addr if norm_addr != "" else None
        key_F = first_two
        key_G = (country, first_two) if country != "" and first_two is not None else None
        if key_A is not None and key_A in keys_of_interest["A"]:
            freq["A"][key_A] += 1
        if key_B is not None and key_B in keys_of_interest["B"]:
            freq["B"][key_B] += 1
        if key_C is not None and key_C in keys_of_interest["C"]:
            freq["C"][key_C] += 1
        if key_D is not None and key_D in keys_of_interest["D"]:
            freq["D"][key_D] += 1
        if key_F is not None and key_F in keys_of_interest["F"]:
            freq["F"][key_F] += 1
        if key_G is not None and key_G in keys_of_interest["G"]:
            freq["G"][key_G] += 1
        if norm_addr != "":
            row_tokens = set(t for t in norm_addr.split(" ") if t != "")
            interesting = row_tokens & tokens_of_interest
            for tk in interesting:
                addr_token_doc_freq[tk] += 1
    return row_count


# ---------------------------------------------------------------------------
# Pass 4 (optional): candidate-set-size pass for token-set rules (E, H_*)
# and unions (I_AE, I_CE, I_CH). Uses reverse indexes built ONLY from the
# sampled entities' own keys/tokens.
# ---------------------------------------------------------------------------

def run_candidate_pass(path: Path, encoding: str,
                        reverse_index_A: dict, reverse_index_C: dict,
                        reverse_index_E_token: dict, reverse_index_H: dict,
                        cutoffs: list, union_h_cutoff: int,
                        tokens_of_interest: frozenset,
                        candidate_count: dict,
                        s1_records: dict, gt_matches: dict,
                        negative_examples: list, negative_cap: int) -> int:
    row_count = 0
    for eid, name, addr, country in iter_source_rows(path, encoding):
        row_count += 1
        norm_name = normalize_text(name)
        tok = first_token(norm_name)
        key_A_row = norm_name if norm_name != "" else None
        key_C_row = (country, tok) if country != "" and tok != "" else None
        hit_A = set(reverse_index_A.get(key_A_row, [])) if key_A_row is not None else set()
        hit_C = set(reverse_index_C.get(key_C_row, [])) if key_C_row is not None else set()

        norm_addr = normalize_text(addr)
        hit_E = set()
        hit_H = {c: set() for c in cutoffs}
        if norm_addr != "":
            row_tokens = set(t for t in norm_addr.split(" ") if t != "")
            interesting = row_tokens & tokens_of_interest
            for tk in interesting:
                if tk in reverse_index_E_token:
                    hit_E |= set(reverse_index_E_token[tk])
                for c in cutoffs:
                    idx_c = reverse_index_H[c]
                    if tk in idx_c:
                        hit_H[c] |= set(idx_c[tk])

        for s1id in hit_E:
            candidate_count["E"][s1id] += 1
        for c in cutoffs:
            key = f"H_{c}"
            for s1id in hit_H[c]:
                candidate_count[key][s1id] += 1

        union_AE = hit_A | hit_E
        for s1id in union_AE:
            candidate_count["I_AE"][s1id] += 1
        union_CE = hit_C | hit_E
        for s1id in union_CE:
            candidate_count["I_CE"][s1id] += 1
        union_CH = hit_C | hit_H.get(union_h_cutoff, set())
        for s1id in union_CH:
            candidate_count["I_CH"][s1id] += 1

        # Deterministic (scan-order) negative-pair sample, drawn from rule
        # E's candidates: a record that shares an address token with a
        # sampled S1 entity but is NOT one of that entity's true matches.
        if len(negative_examples) < negative_cap:
            for s1id in hit_E:
                if eid == s1id:
                    continue
                if eid in gt_matches.get(s1id, ()):
                    continue
                s1_ek = s1_records[s1id]
                sim_name = levenshtein_similarity(s1_ek.norm_name, norm_name)
                sim_addr_j = token_jaccard(s1_ek.norm_addr, norm_addr)
                negative_examples.append({"name_levenshtein": sim_name, "addr_jaccard": sim_addr_j})
                if len(negative_examples) >= negative_cap:
                    break
    return row_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=None,
                   help="Project root to search from (default: parent of this script's tools/ dir)")
    p.add_argument("--encoding", type=str, default="utf-8")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--sample-size", type=int, default=20000)
    p.add_argument("--rare-token-cutoffs", type=str, default="50,200,1000",
                   help="Comma-separated absolute corpus document-frequency cutoffs for rule H")
    p.add_argument("--union-h-cutoff", type=int, default=200,
                   help="Which --rare-token-cutoffs value to use for the C+H union (must be one of them)")
    p.add_argument("--negative-sample-cap", type=int, default=5000)
    p.add_argument("--skip-token-set-candidates", action="store_true",
                   help="Skip the second (candidate-count) scan for rules E/H_*/I_*; recall for "
                        "those rules is still computed and reported.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    import random as _random
    args = parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    root = args.root if args.root is not None else script_dir.parent

    cutoffs = sorted(set(int(x.strip()) for x in args.rare_token_cutoffs.split(",") if x.strip() != ""))
    if not cutoffs:
        print("[FATAL] --rare-token-cutoffs produced an empty list.", file=sys.stderr)
        return 2
    if args.union_h_cutoff not in cutoffs:
        print(f"[FATAL] --union-h-cutoff {args.union_h_cutoff} is not one of --rare-token-cutoffs {cutoffs}.",
              file=sys.stderr)
        return 2

    files = find_train_files(root)
    missing = [name for name, p in files.items() if p is None]
    if missing:
        print(f"[FATAL] Could not locate required train file(s) under {root}: {missing}", file=sys.stderr)
        return 2

    generated_at = datetime.now(timezone.utc).isoformat()
    rng = _random.Random(args.seed)

    print(f"[INFO] benchmark_blocking.py v{SCRIPT_VERSION} starting. seed={args.seed} "
          f"sample_size={args.sample_size} cutoffs={cutoffs} union_h_cutoff={args.union_h_cutoff}")

    # --- Pass 1: reservoir-sample S1 entities from ground truth ---
    print("[INFO] Pass 1: sampling Source-1 entities from train_ground_truth.tsv ...")
    reservoir, total_gt_rows, unclassified_prefix_count, gt_note = sample_ground_truth_entities(
        files["train_ground_truth.tsv"], args.encoding, args.sample_size, rng)
    gt_matches = {s1_id: tuple(matched) for s1_id, matched in reservoir}
    sample_ids = set(gt_matches.keys())
    actual_sample_size = len(sample_ids)
    singleton_ids = {s1_id for s1_id, matched in gt_matches.items() if len(matched) == 0}
    non_singleton_ids = sample_ids - singleton_ids
    all_true_match_ids = set()
    for matched in gt_matches.values():
        all_true_match_ids.update(matched)
    total_true_match_pairs_in_sample = sum(len(m) for m in gt_matches.values())
    print(f"[INFO]   sampled {actual_sample_size} Source-1 entities "
          f"({len(singleton_ids)} singletons, {len(non_singleton_ids)} with >=1 true match, "
          f"{total_true_match_pairs_in_sample} true-match pairs, "
          f"{unclassified_prefix_count} unclassified matched-id prefixes skipped)")

    # --- Pass 2: load sampled S1 records + derive keys ---
    print("[INFO] Pass 2: loading sampled Source-1 records from train_source1.tsv ...")
    s1_records = load_sampled_source1(files["train_source1.tsv"], args.encoding, sample_ids)
    missing_s1 = sample_ids - set(s1_records.keys())
    if missing_s1:
        print(f"[WARN]   {len(missing_s1)} sampled Source-1 ids from ground truth were NOT found "
              f"in train_source1.tsv (reported in output, not silently dropped).")

    # keys/tokens of interest, restricted to the sample
    keys_of_interest = {r: set() for r in ["A", "B", "C", "D", "F", "G"]}
    tokens_of_interest = set()
    reverse_index_A = defaultdict(list)
    reverse_index_C = defaultdict(list)
    reverse_index_E_token = defaultdict(list)
    excluded_empty_key = {r: 0 for r in ["A", "B", "C", "D", "F", "G"]}
    excluded_empty_addr_for_tokens = 0
    for eid, ek in s1_records.items():
        if ek.key_A is not None:
            keys_of_interest["A"].add(ek.key_A)
            reverse_index_A[ek.key_A].append(eid)
        else:
            excluded_empty_key["A"] += 1
        if ek.key_B is not None:
            keys_of_interest["B"].add(ek.key_B)
        else:
            excluded_empty_key["B"] += 1
        if ek.key_C is not None:
            keys_of_interest["C"].add(ek.key_C)
            reverse_index_C[ek.key_C].append(eid)
        else:
            excluded_empty_key["C"] += 1
        if ek.key_D is not None:
            keys_of_interest["D"].add(ek.key_D)
        else:
            excluded_empty_key["D"] += 1
        if ek.key_F is not None:
            keys_of_interest["F"].add(ek.key_F)
        else:
            excluded_empty_key["F"] += 1
        if ek.key_G is not None:
            keys_of_interest["G"].add(ek.key_G)
        else:
            excluded_empty_key["G"] += 1
        if ek.addr_empty:
            excluded_empty_addr_for_tokens += 1
        else:
            tokens_of_interest.update(ek.tokens_addr)
            for tk in ek.tokens_addr:
                reverse_index_E_token[tk].append(eid)
    tokens_of_interest = frozenset(tokens_of_interest)

    # --- Pass 3: frequency pass over source2 + source3 ---
    print("[INFO] Pass 3: frequency pass over train_source2.tsv ...")
    freq = {r: Counter() for r in ["A", "B", "C", "D", "F", "G"]}
    addr_token_doc_freq = Counter()
    referenced_records = {}
    s2_rows = run_frequency_pass(files["train_source2.tsv"], args.encoding, keys_of_interest,
                                  tokens_of_interest, all_true_match_ids, freq,
                                  addr_token_doc_freq, referenced_records)
    print("[INFO] Pass 3: frequency pass over train_source3.tsv ...")
    s3_rows = run_frequency_pass(files["train_source3.tsv"], args.encoding, keys_of_interest,
                                  tokens_of_interest, all_true_match_ids, freq,
                                  addr_token_doc_freq, referenced_records)
    total_corpus_rows = s2_rows + s3_rows
    unresolved_matched_ids = all_true_match_ids - set(referenced_records.keys())
    print(f"[INFO]   scanned {s2_rows} + {s3_rows} = {total_corpus_rows} corpus rows; "
          f"{len(referenced_records)}/{len(all_true_match_ids)} true-match ids resolved to a record "
          f"({len(unresolved_matched_ids)} unresolved)")

    # candidate_count[rule][s1_id] -> int, populated for ALL rules up front
    all_rule_keys = list(RULE_ORDER) + [f"H_{c}" for c in cutoffs] + ["I_AE", "I_CE", "I_CH"]
    candidate_count = {r: defaultdict(int) for r in all_rule_keys}
    for eid, ek in s1_records.items():
        if ek.key_A is not None:
            candidate_count["A"][eid] = freq["A"].get(ek.key_A, 0)
        if ek.key_B is not None:
            candidate_count["B"][eid] = freq["B"].get(ek.key_B, 0)
        if ek.key_C is not None:
            candidate_count["C"][eid] = freq["C"].get(ek.key_C, 0)
        if ek.key_D is not None:
            candidate_count["D"][eid] = freq["D"].get(ek.key_D, 0)
        if ek.key_F is not None:
            candidate_count["F"][eid] = freq["F"].get(ek.key_F, 0)
        if ek.key_G is not None:
            candidate_count["G"][eid] = freq["G"].get(ek.key_G, 0)

    # --- Build H reverse indexes now that corpus token doc-freq is known ---
    rare_tokens_per_entity = {c: {} for c in cutoffs}
    reverse_index_H = {c: defaultdict(list) for c in cutoffs}
    for eid, ek in s1_records.items():
        if ek.addr_empty:
            continue
        for c in cutoffs:
            rare = frozenset(t for t in ek.tokens_addr if addr_token_doc_freq.get(t, 0) <= c)
            rare_tokens_per_entity[c][eid] = rare
            for t in rare:
                reverse_index_H[c][t].append(eid)

    # --- Recall computation (exact, from referenced_records -- no extra scan) ---
    print("[INFO] Computing recall from resolved ground-truth matches ...")
    retrieved = {r: defaultdict(set) for r in all_rule_keys}
    positive_name_sims = []
    positive_addr_jaccards = []
    resolved_pairs = 0
    for s1_id, matched in gt_matches.items():
        if s1_id not in s1_records:
            continue
        s1_ek = s1_records[s1_id]
        for mid in matched:
            if mid not in referenced_records:
                continue
            resolved_pairs += 1
            m_name, m_addr, m_country = referenced_records[mid]
            m_ek = compute_entity_keys(mid, m_name, m_addr, m_country)

            positive_name_sims.append(levenshtein_similarity(s1_ek.norm_name, m_ek.norm_name))
            aj = token_jaccard(s1_ek.norm_addr, m_ek.norm_addr)
            if aj is not None:
                positive_addr_jaccards.append(aj)

            hit_A = s1_ek.key_A is not None and m_ek.key_A is not None and s1_ek.key_A == m_ek.key_A
            hit_B = s1_ek.key_B is not None and m_ek.key_B is not None and s1_ek.key_B == m_ek.key_B
            hit_C = s1_ek.key_C is not None and m_ek.key_C is not None and s1_ek.key_C == m_ek.key_C
            hit_D = s1_ek.key_D is not None and m_ek.key_D is not None and s1_ek.key_D == m_ek.key_D
            hit_F = s1_ek.key_F is not None and m_ek.key_F is not None and s1_ek.key_F == m_ek.key_F
            hit_G = s1_ek.key_G is not None and m_ek.key_G is not None and s1_ek.key_G == m_ek.key_G
            hit_E = (not s1_ek.addr_empty) and (not m_ek.addr_empty) and bool(s1_ek.tokens_addr & m_ek.tokens_addr)
            hit_H = {}
            for c in cutoffs:
                rare = rare_tokens_per_entity[c].get(s1_id, frozenset())
                hit_H[c] = (not s1_ek.addr_empty) and (not m_ek.addr_empty) and bool(rare & m_ek.tokens_addr)

            if hit_A:
                retrieved["A"][s1_id].add(mid)
            if hit_B:
                retrieved["B"][s1_id].add(mid)
            if hit_C:
                retrieved["C"][s1_id].add(mid)
            if hit_D:
                retrieved["D"][s1_id].add(mid)
            if hit_E:
                retrieved["E"][s1_id].add(mid)
            if hit_F:
                retrieved["F"][s1_id].add(mid)
            if hit_G:
                retrieved["G"][s1_id].add(mid)
            for c in cutoffs:
                if hit_H[c]:
                    retrieved[f"H_{c}"][s1_id].add(mid)
            if hit_A or hit_E:
                retrieved["I_AE"][s1_id].add(mid)
            if hit_C or hit_E:
                retrieved["I_CE"][s1_id].add(mid)
            if hit_C or hit_H.get(args.union_h_cutoff, False):
                retrieved["I_CH"][s1_id].add(mid)

    # --- Pass 4 (optional): candidate-set-size stats for E / H_* / I_* ---
    negative_examples = []
    pass4_run = False
    if not args.skip_token_set_candidates:
        pass4_run = True
        print("[INFO] Pass 4: candidate-count scan (rules E/H_*/I_*) over train_source2.tsv ...")
        run_candidate_pass(files["train_source2.tsv"], args.encoding, reverse_index_A, reverse_index_C,
                            reverse_index_E_token, reverse_index_H, cutoffs, args.union_h_cutoff,
                            tokens_of_interest, candidate_count, s1_records, gt_matches,
                            negative_examples, args.negative_sample_cap)
        print("[INFO] Pass 4: candidate-count scan (rules E/H_*/I_*) over train_source3.tsv ...")
        run_candidate_pass(files["train_source3.tsv"], args.encoding, reverse_index_A, reverse_index_C,
                            reverse_index_E_token, reverse_index_H, cutoffs, args.union_h_cutoff,
                            tokens_of_interest, candidate_count, s1_records, gt_matches,
                            negative_examples, args.negative_sample_cap)
    else:
        print("[INFO] Pass 4 skipped (--skip-token-set-candidates). Recall for E/H_*/I_* still computed.")

    # --- Assemble per-rule stats ---
    rule_labels = {
        "A": "A: exact normalized business_name",
        "B": "B: first token of normalized business_name",
        "C": "C: country + first normalized name token",
        "D": "D: exact normalized business_address",
        "E": "E: any shared normalized address token",
        "F": "F: first two normalized name tokens (where available)",
        "G": "G: country + first two normalized name tokens (where available)",
        "I_AE": "I: union of candidates from rule A and rule E",
        "I_CE": "I: union of candidates from rule C and rule E",
        "I_CH": f"I: union of candidates from rule C and rule H (cutoff={args.union_h_cutoff})",
    }
    for c in cutoffs:
        rule_labels[f"H_{c}"] = f"H: rare shared address token (corpus doc-freq <= {c} among tokens of interest)"

    candidate_stats_rules = set(RULE_ORDER) if args.skip_token_set_candidates else set(all_rule_keys)
    # A/B/C/D/F/G candidate counts always come from Pass 3 (never need Pass 4)
    for r in ["A", "B", "C", "D", "F", "G"]:
        candidate_stats_rules.add(r)

    rules_report = {}
    for r in all_rule_keys:
        # Applicability: for A/B/C/D/F/G use key-not-None; for E/H use addr-not-empty;
        # for I_* (unions) every sampled entity is applicable (name-based component alone can apply).
        if r in ("A", "B", "C", "D", "F", "G"):
            applicable_eids = [eid for eid, ek in s1_records.items()
                               if getattr(ek, f"key_{r}") is not None]
        elif r == "E" or r.startswith("H_"):
            applicable_eids = [eid for eid, ek in s1_records.items() if not ek.addr_empty]
        else:  # unions
            applicable_eids = list(s1_records.keys())

        counts = [candidate_count[r].get(eid, 0) for eid in applicable_eids]
        has_candidate_stats = (r in candidate_stats_rules) if r in ("A", "B", "C", "D", "F", "G") else \
                               (pass4_run if (r == "E" or r.startswith("H_") or r in ("I_AE", "I_CE", "I_CH")) else False)

        cand_stats = full_distribution_summary(counts) if (has_candidate_stats and counts) else \
            ("not computed (--skip-token-set-candidates)" if not has_candidate_stats else
             full_distribution_summary([]))

        total_candidate_pairs = sum(counts) if has_candidate_stats else None
        mean_candidates = (sum(counts) / len(counts)) if (has_candidate_stats and counts) else None
        reduction_ratio = (1.0 - (mean_candidates / total_corpus_rows)) if (
            has_candidate_stats and mean_candidates is not None and total_corpus_rows > 0) else None

        # recall (pair-level and entity-level), over non-singleton sampled entities with resolved matches
        pair_hits = 0
        pair_total = 0
        entities_all_retrieved = 0
        entities_some_missed = 0
        entities_considered = 0
        for s1_id in non_singleton_ids:
            if s1_id not in s1_records:
                continue
            matched = gt_matches[s1_id]
            resolved_matched = [m for m in matched if m in referenced_records]
            if not resolved_matched:
                continue
            entities_considered += 1
            found = retrieved[r].get(s1_id, set())
            pair_hits += len(found)
            pair_total += len(resolved_matched)
            if found >= set(resolved_matched):
                entities_all_retrieved += 1
            else:
                entities_some_missed += 1
        pair_level_recall = (pair_hits / pair_total) if pair_total else None
        entity_level_all_retrieved_rate = (entities_all_retrieved / entities_considered) if entities_considered else None

        # singleton candidate behavior
        singleton_applicable = [eid for eid in applicable_eids if eid in singleton_ids]
        singleton_counts = [candidate_count[r].get(eid, 0) for eid in singleton_applicable]
        singleton_zero = sum(1 for v in singleton_counts if v == 0)
        singleton_nonzero = sum(1 for v in singleton_counts if v > 0)
        singleton_stats = {
            "n_singleton_applicable": len(singleton_applicable),
            "pct_zero_candidates": (singleton_zero / len(singleton_counts)) if singleton_counts else None,
            "pct_nonzero_candidates": (singleton_nonzero / len(singleton_counts)) if singleton_counts else None,
            "candidate_count_distribution": full_distribution_summary(singleton_counts) if (
                has_candidate_stats and singleton_counts) else "not computed (--skip-token-set-candidates)"
                if not has_candidate_stats else full_distribution_summary([]),
        }

        rules_report[r] = {
            "label": rule_labels[r],
            "n_applicable_sampled_entities": len(applicable_eids),
            "n_excluded_not_applicable": actual_sample_size - len(applicable_eids),
            "candidate_count_stats": cand_stats,
            "total_candidate_pairs_generated": total_candidate_pairs,
            "reduction_ratio_vs_brute_force": reduction_ratio,
            "recall": {
                "pair_level_recall": pair_level_recall,
                "pair_hits": pair_hits,
                "pair_total_resolved": pair_total,
                "entities_considered_non_singleton_resolved": entities_considered,
                "entities_all_true_matches_retrieved": entities_all_retrieved,
                "entities_at_least_one_match_missed": entities_some_missed,
                "entity_level_all_retrieved_rate": entity_level_all_retrieved_rate,
            },
            "singleton_candidate_behavior": singleton_stats,
        }

    # --- Ambiguity section (derived from rules A and D candidate counts) ---
    def ambiguity_for(rule_key):
        applicable = [eid for eid, ek in s1_records.items() if getattr(ek, f"key_{rule_key}") is not None]
        counts = [candidate_count[rule_key].get(eid, 0) for eid in applicable]
        ambiguous = [c for c in counts if c > 1]
        return {
            "n_applicable_sampled_entities": len(applicable),
            "n_ambiguous_entities_gt1_corpus_record_same_key": len(ambiguous),
            "pct_ambiguous": (len(ambiguous) / len(applicable)) if applicable else None,
            "ambiguous_candidate_set_size_distribution": full_distribution_summary(ambiguous) if ambiguous else full_distribution_summary([]),
        }

    ambiguity_report = {
        "exact_normalized_name (rule A)": ambiguity_for("A"),
        "exact_normalized_address (rule D)": ambiguity_for("D"),
        "note": "Ambiguity here means: for this sampled Source-1 entity's exact-normalized key, "
                "more than one Source-2/Source-3 record in the FULL training corpus shares that same "
                "key. This is the same statistic as the rule A / rule D candidate-count stats above, "
                "re-framed as an ambiguity question. It says nothing about whether those extra records "
                "are correct matches for a DIFFERENT Source-1 entity -- only that the key alone cannot "
                "disambiguate them.",
    }

    # --- Negative-pair observations ---
    neg_name_sims = [e["name_levenshtein"] for e in negative_examples]
    neg_addr_jaccards = [e["addr_jaccard"] for e in negative_examples if e["addr_jaccard"] is not None]
    negative_report = {
        "performed": pass4_run and len(negative_examples) > 0,
        "source_of_negatives": "Deterministic, scan-order sample of rule-E candidates (share >=1 address "
                                "token with the sampled Source-1 entity) that are NOT in that entity's "
                                "ground-truth matched-id list. Not a random sample of all possible "
                                "non-matches, and not claimed to represent the full negative population.",
        "count_constructed": len(negative_examples),
        "requested_cap": args.negative_sample_cap,
        "name_levenshtein_similarity": full_distribution_summary(neg_name_sims) if neg_name_sims else full_distribution_summary([]),
        "addr_jaccard": full_distribution_summary(neg_addr_jaccards) if neg_addr_jaccards else full_distribution_summary([]),
    }
    positive_report = {
        "count": resolved_pairs,
        "name_levenshtein_similarity": full_distribution_summary(positive_name_sims) if positive_name_sims else full_distribution_summary([]),
        "addr_jaccard": full_distribution_summary(positive_addr_jaccards) if positive_addr_jaccards else full_distribution_summary([]),
        "note": "Computed directly over every resolved ground-truth true-match pair in this benchmark's "
                "sample (not a further sub-sample) -- see 'resolved_pairs' vs 'total_true_match_pairs_in_sample'.",
    }

    # --- Assemble full JSON report ---
    json_report = {
        "benchmark_metadata": {
            "generated_at_utc": generated_at,
            "script": "tools/benchmark_blocking.py",
            "script_version": SCRIPT_VERSION,
            "search_root": str(root),
            "random_seed": args.seed,
            "requested_sample_size": args.sample_size,
            "actual_sample_size": actual_sample_size,
            "rare_token_cutoffs": cutoffs,
            "union_h_cutoff": args.union_h_cutoff,
            "negative_sample_cap": args.negative_sample_cap,
            "skip_token_set_candidates": args.skip_token_set_candidates,
            "note": "ALL statistics in this report are BENCHMARK RESULTS on a deterministic sample of "
                    f"{actual_sample_size} Source-1 training entities (seed={args.seed}), not full-"
                    "population results. This script has not trained any model, generated any "
                    "prediction, or modified any dataset file.",
            "column_detection_note_ground_truth": gt_note,
        },
        "sample_composition": {
            "total_ground_truth_rows_seen": total_gt_rows,
            "unclassified_matched_id_prefix_count": unclassified_prefix_count,
            "sampled_entities": actual_sample_size,
            "sampled_singletons": len(singleton_ids),
            "sampled_non_singletons": len(non_singleton_ids),
            "total_true_match_pairs_in_sample": total_true_match_pairs_in_sample,
            "s1_ids_missing_from_train_source1": len(missing_s1),
            "true_match_ids_resolved_to_a_record": len(referenced_records),
            "true_match_ids_unresolved": len(unresolved_matched_ids),
            "resolved_pairs_used_for_recall": resolved_pairs,
        },
        "corpus_scan_totals": {
            "train_source2_rows_scanned": s2_rows,
            "train_source3_rows_scanned": s3_rows,
            "total_corpus_rows_scanned": total_corpus_rows,
            "note": "These are exact row counts from this run's own streaming scan, not assumed from "
                    "any earlier report.",
        },
        "excluded_not_applicable_by_rule": excluded_empty_key,
        "excluded_empty_address_for_token_rules": excluded_empty_addr_for_tokens,
        "blocking_rules": rules_report,
        "ambiguity": ambiguity_report,
        "positive_pair_similarity": positive_report,
        "negative_pair_observations": negative_report,
        "limitations": [
            "All figures are computed on a single deterministic sample of "
            f"{actual_sample_size} Source-1 training entities (seed={args.seed}), not the full "
            "2.2M-entity training population, and not the test population at all.",
            "Candidate-count / reduction-ratio statistics for rules E, H_*, and I_* are only "
            "available when --skip-token-set-candidates is NOT passed (this run: "
            f"{'computed' if pass4_run else 'skipped'}).",
            "Rule H's 'rare' cutoffs are absolute document-frequency thresholds on the training "
            "corpus only, counted among tokens that appear in the SAMPLED entities' own addresses "
            "('tokens of interest') -- not a corpus-wide token frequency table.",
            "Recall is computed by direct key/token comparison against the actual resolved "
            "ground-truth matched records (exact, not sampled) -- but only for matched ids that "
            "were found during the corpus scan; unresolved matched ids are excluded from recall "
            "denominators and reported separately above.",
            "Negative-pair statistics come from a deterministic, scan-order sample of rule-E "
            "candidates that are not true matches -- not a random sample of the full negative "
            "population, and not claimed to be representative of harder negative categories "
            "(e.g. same-name-different-entity) that rule E itself would not surface.",
            "Reduction ratio uses this run's own measured train_source2+train_source3 row count "
            "as the brute-force denominator, not the test-set size.",
            "Country-field comparisons (rules C, G, and their unions) use exact string equality "
            "only, consistent with the rest of this project's normalization logic; no country-name "
            "normalization or geocoding is used.",
        ],
        "recommended_candidate_generation_architecture": (
            "NOT decided by this script. Section 11 of the Markdown report is intentionally left as "
            "a set of observations to weigh (recall vs. candidate-set size vs. reduction ratio per "
            "rule, from the tables above) rather than a chosen architecture -- that choice should be "
            "made after reviewing this run's actual numbers, not assumed in advance."
        ),
    }

    md_report = build_markdown(root, args, json_report, cutoffs)

    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "blocking_benchmark.json"
    md_path = reports_dir / "blocking_benchmark.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2, default=str)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)
    print(f"[INFO] Wrote: {json_path}")
    print(f"[INFO] Wrote: {md_path}")
    return 0


def _fmt(v):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def build_markdown(root: Path, args, j: dict, cutoffs: list) -> str:
    md = j["benchmark_metadata"]
    sc = j["sample_composition"]
    cs = j["corpus_scan_totals"]
    lines = []
    lines.append("# Blocking / Candidate-Generation Benchmark")
    lines.append("")
    lines.append(f"- Generated (UTC): {md['generated_at_utc']}")
    lines.append(f"- Search root: `{md['search_root']}`")
    lines.append(f"- Script version: {md['script_version']}, seed: {md['random_seed']}")
    lines.append(f"- **This is a BENCHMARK on a deterministic sample, not a full-population result.**")
    lines.append(f"- No raw dataset file was modified. No ML training, blocking implementation, or "
                 f"test predictions were performed. This is an empirical benchmark used to inform "
                 f"blocker design.")
    lines.append("")
    lines.append("## 1-2. Benchmark configuration and sample size")
    lines.append(f"- Requested sample size: {md['requested_sample_size']}; actual: {md['actual_sample_size']}")
    lines.append(f"- Random seed: {md['random_seed']} (Algorithm-R reservoir sampling over "
                 f"train_ground_truth.tsv, deterministic given the same file content)")
    lines.append(f"- Rare-token cutoffs tested (rule H): {cutoffs}")
    lines.append(f"- Union rule I_CH uses cutoff: {md['union_h_cutoff']}")
    lines.append(f"- Negative-pair sample cap: {md['negative_sample_cap']}")
    lines.append(f"- Token-set candidate-count pass (rules E/H_*/I_*): "
                 f"{'RUN' if not md['skip_token_set_candidates'] else 'SKIPPED (--skip-token-set-candidates)'}")
    lines.append(f"- Sampled entities: {sc['sampled_entities']} "
                 f"({sc['sampled_singletons']} singletons, {sc['sampled_non_singletons']} with >=1 true match)")
    lines.append(f"- Total true-match pairs in sample: {sc['total_true_match_pairs_in_sample']} "
                 f"({sc['resolved_pairs_used_for_recall']} resolved to an actual corpus record and used for recall)")
    if sc['true_match_ids_unresolved']:
        lines.append(f"- **{sc['true_match_ids_unresolved']} true-match ids could not be resolved to a "
                     f"record during the corpus scan** -- excluded from recall denominators, not silently counted as hits or misses.")
    if sc['s1_ids_missing_from_train_source1']:
        lines.append(f"- **{sc['s1_ids_missing_from_train_source1']} sampled Source-1 ids were not found "
                     f"in train_source1.tsv.**")
    lines.append(f"- Corpus rows scanned this run: train_source2={cs['train_source2_rows_scanned']}, "
                 f"train_source3={cs['train_source3_rows_scanned']}, total={cs['total_corpus_rows_scanned']}")
    lines.append("")

    lines.append("## 3. Blocking-rule definitions")
    for r, info in j["blocking_rules"].items():
        lines.append(f"- **{r}**: {info['label']}")
    lines.append("")
    lines.append("- Empty-address handling: entities with an empty normalized business_address are "
                 "EXCLUDED from rules D, E, H_*, and from the address-dependent side of any union "
                 "involving them (they never contribute an address-based candidate). They still "
                 "participate normally in name-only rules (A, B, C, F, G) and in a union's name-based "
                 "component.")
    lines.append(f"- Empty-key exclusions observed this run (rule -> count of sampled entities excluded "
                 f"because the rule's key was empty/not applicable): {j['excluded_not_applicable_by_rule']}")
    lines.append(f"- Sampled entities excluded from ALL address-token rules due to empty address: "
                 f"{j['excluded_empty_address_for_token_rules']}")
    lines.append("")

    lines.append("## 4-7. Candidate-count, recall, reduction-ratio, and singleton statistics per rule")
    for r, info in j["blocking_rules"].items():
        lines.append(f"### {r} -- {info['label']}")
        lines.append(f"- Applicable sampled entities: {info['n_applicable_sampled_entities']} "
                     f"(excluded as not-applicable: {info['n_excluded_not_applicable']})")
        lines.append(f"- Candidate-count distribution: `{info['candidate_count_stats']}`")
        lines.append(f"- Total candidate pairs generated (this rule, this sample): "
                     f"{_fmt(info['total_candidate_pairs_generated'])}")
        lines.append(f"- Approximate reduction ratio vs. brute force "
                     f"(1 - mean_candidates / total_corpus_rows_scanned): "
                     f"{_fmt(info['reduction_ratio_vs_brute_force'])}")
        rec = info["recall"]
        lines.append(f"- Pair-level recall: {_fmt(rec['pair_level_recall'])} "
                     f"({rec['pair_hits']}/{rec['pair_total_resolved']} resolved true-match pairs retrieved)")
        lines.append(f"- Entity-level: {rec['entities_all_true_matches_retrieved']} of "
                     f"{rec['entities_considered_non_singleton_resolved']} considered non-singleton "
                     f"entities had ALL true matches retrieved "
                     f"({_fmt(rec['entity_level_all_retrieved_rate'])}); "
                     f"{rec['entities_at_least_one_match_missed']} had at least one match missed")
        sb = info["singleton_candidate_behavior"]
        lines.append(f"- Singleton behavior: {sb['n_singleton_applicable']} applicable sampled singletons; "
                     f"pct with zero candidates: {_fmt(sb['pct_zero_candidates'])}; "
                     f"pct with nonzero candidates: {_fmt(sb['pct_nonzero_candidates'])} "
                     f"(a nonzero candidate set for a singleton is NOT a false match -- it is only a "
                     f"pre-matching-model candidate)")
        lines.append(f"- Singleton candidate-count distribution: `{sb['candidate_count_distribution']}`")
        lines.append("")

    lines.append("## 8. Ambiguity observations")
    amb = j["ambiguity"]
    for k in ["exact_normalized_name (rule A)", "exact_normalized_address (rule D)"]:
        a = amb[k]
        lines.append(f"- **{k}**: {a['n_ambiguous_entities_gt1_corpus_record_same_key']} of "
                     f"{a['n_applicable_sampled_entities']} applicable sampled entities "
                     f"({_fmt(a['pct_ambiguous'])}) have MORE THAN ONE corpus record sharing their "
                     f"exact key; candidate-set-size distribution among those ambiguous entities: "
                     f"`{a['ambiguous_candidate_set_size_distribution']}`")
    lines.append(f"- {amb['note']}")
    lines.append("")

    lines.append("## 9. Negative-pair observations")
    neg = j["negative_pair_observations"]
    if neg["performed"]:
        lines.append(f"- {neg['source_of_negatives']}")
        lines.append(f"- Constructed: {neg['count_constructed']} (requested cap: {neg['requested_cap']})")
        lines.append(f"- Name Levenshtein similarity distribution (negatives): `{neg['name_levenshtein_similarity']}`")
        lines.append(f"- Address token Jaccard distribution (negatives): `{neg['addr_jaccard']}`")
        pos = j["positive_pair_similarity"]
        lines.append(f"- For comparison, TRUE-match pairs in this sample ({pos['count']} resolved pairs): "
                     f"name Levenshtein `{pos['name_levenshtein_similarity']}`, "
                     f"address Jaccard `{pos['addr_jaccard']}`")
    else:
        lines.append("- Not performed this run (requires the token-set candidate pass; "
                     "--skip-token-set-candidates was set, or no rule-E candidates were found).")
    lines.append("")

    lines.append("## 10. Limitations")
    for lim in j["limitations"]:
        lines.append(f"- {lim}")
    lines.append("")

    lines.append("## 11. Recommended candidate-generation architecture")
    lines.append(f"- {j['recommended_candidate_generation_architecture']}")
    lines.append("- When reviewing the tables above, weigh recall (section 4-7) together with "
                 "candidate-set size and reduction ratio for the same rule -- a rule is not reported "
                 "as \"best\" here merely for having the highest recall; a very high-recall rule with "
                 "an unmanageable candidate-set size and low reduction ratio is not automatically "
                 "preferable to a slightly-lower-recall rule with a much smaller, more tractable "
                 "candidate set. No such trade-off has been resolved by this script.")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())