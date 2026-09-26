#!/usr/bin/env python3
"""
v1_features.py

Feature engineering module for Amazon ML Challenge V1 Entity Resolution.
Provides:
  1. Full 15-Feature Extractor (for Matcher training and final prediction).
  2. Fast Heuristic Candidate Scorer (for candidate bounding per noisy record).
"""

from __future__ import annotations

import math
from typing import Dict, List, Set, Tuple

from common import (
    digit_jaccard_sim,
    first_token,
    jaccard_sim,
    levenshtein_sim,
    normalize_text,
    token_sort_sim,
    tokenize_address,
)

FEATURE_NAMES = [
    "name_exact_raw",
    "name_exact_norm",
    "name_levenshtein_sim",
    "name_jaccard_sim",
    "name_token_sort_sim",
    "name_first_token_match",
    "name_length_diff_abs",
    "name_length_ratio",
    "address_exact_norm",
    "address_levenshtein_sim",
    "address_jaccard_sim",
    "address_overlap_ratio",
    "address_is_missing",
    "address_digit_jaccard",
    "country_match",
]


def extract_v1_features(s1_row: Dict[str, str], s23_row: Dict[str, str]) -> List[float]:
    """
    Computes the exact 15-feature vector for a pair of entity records.

    Features (15):
      1. name_exact_raw
      2. name_exact_norm
      3. name_levenshtein_sim
      4. name_jaccard_sim
      5. name_token_sort_sim
      6. name_first_token_match
      7. name_length_diff_abs
      8. name_length_ratio
      9. address_exact_norm
      10. address_levenshtein_sim
      11. address_jaccard_sim
      12. address_overlap_ratio
      13. address_is_missing
      14. address_digit_jaccard
      15. country_match
    """
    raw_s1_name = s1_row.get("business_name", "").strip()
    raw_s23_name = s23_row.get("business_name", "").strip()

    raw_s1_addr = s1_row.get("business_address", "").strip()
    raw_s23_addr = s23_row.get("business_address", "").strip()

    country_s1 = s1_row.get("country", "").strip()
    country_s23 = s23_row.get("country", "").strip()

    norm_s1_name = normalize_text(raw_s1_name)
    norm_s23_name = normalize_text(raw_s23_name)

    norm_s1_addr = normalize_text(raw_s1_addr)
    norm_s23_addr = normalize_text(raw_s23_addr)

    # 1. name_exact_raw
    f_name_exact_raw = 1.0 if (raw_s1_name and raw_s1_name == raw_s23_name) else 0.0

    # 2. name_exact_norm
    f_name_exact_norm = 1.0 if (norm_s1_name and norm_s1_name == norm_s23_name) else 0.0

    # 3. name_levenshtein_sim
    f_name_levenshtein = levenshtein_sim(norm_s1_name, norm_s23_name)

    # 4. name_jaccard_sim
    tokens1_name = tokenize_address(norm_s1_name)
    tokens23_name = tokenize_address(norm_s23_name)
    f_name_jaccard = jaccard_sim(tokens1_name, tokens23_name)

    # 5. name_token_sort_sim
    f_name_token_sort = token_sort_sim(norm_s1_name, norm_s23_name)

    # 6. name_first_token_match
    ft1 = first_token(norm_s1_name)
    ft23 = first_token(norm_s23_name)
    f_name_first_token = 1.0 if (ft1 and ft1 == ft23) else 0.0

    # 7. name_length_diff_abs
    f_name_len_diff = float(abs(len(norm_s1_name) - len(norm_s23_name)))

    # 8. name_length_ratio
    max_name_len = max(len(norm_s1_name), len(norm_s23_name))
    min_name_len = min(len(norm_s1_name), len(norm_s23_name))
    f_name_len_ratio = (min_name_len / max_name_len) if max_name_len > 0 else 0.0

    # 9. address_exact_norm
    f_addr_exact_norm = 1.0 if (norm_s1_addr and norm_s1_addr == norm_s23_addr) else 0.0

    # 10. address_levenshtein_sim
    f_addr_levenshtein = levenshtein_sim(norm_s1_addr, norm_s23_addr)

    # 11. address_jaccard_sim
    tokens1_addr = tokenize_address(norm_s1_addr)
    tokens23_addr = tokenize_address(norm_s23_addr)
    f_addr_jaccard = jaccard_sim(tokens1_addr, tokens23_addr)

    # 12. address_overlap_ratio
    min_tokens = min(len(tokens1_addr), len(tokens23_addr))
    intersection = len(tokens1_addr & tokens23_addr)
    f_addr_overlap_ratio = (intersection / min_tokens) if min_tokens > 0 else 0.0

    # 13. address_is_missing
    f_addr_missing = 1.0 if (not norm_s1_addr or not norm_s23_addr) else 0.0

    # 14. address_digit_jaccard
    f_addr_digit_jaccard = digit_jaccard_sim(norm_s1_addr, norm_s23_addr)

    # 15. country_match
    f_country_match = 1.0 if (country_s1 and country_s1 == country_s23) else 0.0

    return [
        f_name_exact_raw,
        f_name_exact_norm,
        f_name_levenshtein,
        f_name_jaccard,
        f_name_token_sort,
        f_name_first_token,
        f_name_len_diff,
        f_name_len_ratio,
        f_addr_exact_norm,
        f_addr_levenshtein,
        f_addr_jaccard,
        f_addr_overlap_ratio,
        f_addr_missing,
        f_addr_digit_jaccard,
        f_country_match,
    ]


def fast_heuristic_candidate_score(
    s1_norm_name: str,
    s1_norm_addr: str,
    s1_country: str,
    s23_norm_name: str,
    s23_norm_addr: str,
    s23_country: str,
    is_exact_name_hit: bool,
) -> float:
    """
    Fast O(1) heuristic scorer used during streaming candidate generation
    to rank broad blocker hits (G, E_rare@200) and select top-K candidates.
    """
    if is_exact_name_hit:
        return 10.0  # Exact name hits score maximum and are always retained

    score = 0.0

    # Country equality
    if s1_country and s1_country == s23_country:
        score += 1.0
    elif s1_country and s23_country and s1_country != s23_country:
        score -= 2.0  # Country conflict penalty

    # First token match
    ft1 = first_token(s1_norm_name)
    ft2 = first_token(s23_norm_name)
    if ft1 and ft1 == ft2:
        score += 1.5

    # Token-level Name Overlap
    t_n1 = tokenize_address(s1_norm_name)
    t_n2 = tokenize_address(s23_norm_name)
    if t_n1 and t_n2:
        name_jaccard = jaccard_sim(t_n1, t_n2)
        score += 3.0 * name_jaccard

    # Token-level Address Overlap
    t_a1 = tokenize_address(s1_norm_addr)
    t_a2 = tokenize_address(s23_norm_addr)
    if t_a1 and t_a2:
        addr_jaccard = jaccard_sim(t_a1, t_a2)
        score += 2.0 * addr_jaccard

    return score