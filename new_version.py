import json
import sys
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List

@dataclass
class NormalizedRecord:
    category: str                  # STATE_MARKS / COMMON_LAW / WEB_COMMON_LAW / BUSINESS_NAME / WEB_DOMAIN
    source_side: str               # "json1" or "json2"
    vendor_name: Optional[str] = None

    record_id: Optional[str] = None
    registration_number: Optional[str] = None
    primary_sic: Optional[str] = None

    owner_raw: str = ""
    owner_norm: str = ""

    trademark_raw: str = ""
    trademark_norm: str = ""

    goods_raw: str = ""
    goods_norm: str = ""

    class_list: Tuple[int, ...] = ()

    skip_from_gate2: bool = False  # for WEB_DOMAIN rule_filter skip behavior

    raw_payload: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CategoryComparisonResult:
    category: str
    passed: int
    total: int
    failures: List[str]
    report_lines: List[str]
    skipped: int = 0
    total_extracted: int = 0

# Unified category mapping for internal comparison
CATEGORY_MAP = {
    "STATE_LAW": "STATE_MARKS",
    "COMMON_LAW": "COMMON_LAW",
    "WEB_COMMON_LAW": "WEB_COMMON_LAW",
    "USPTO": "USPTO_MARKS",
    "FEDERAL_USPTO": "USPTO_MARKS",
    "BUSINESS_NAME": "BUSINESS_NAME",
    "DOMAIN_NAME": "WEB_DOMAIN",
}

def normalize_owner(name):
    """
    Normalizes company names to handle case, punctuation, and common suffixes.
    Example: 'Alberto-Culver USA, Inc.' -> 'ALBERTO CULVER USA'
    """
    if not name:
        return ""
    # Convert to uppercase
    name = str(name).upper()
    
    # Requirement 3: Normalize acronym punctuation (U.S. -> US)
    # Match patterns like U.S. or A.B.C. and remove the dots
    # We look for single letters followed by a dot, repeated.
    name = re.sub(r'\b([A-Z])\.(?=[A-Z]\.|\b)', r'\1', name)

    # Remove apostrophes to match (e.g., "Zatarain's" -> "ZATARAINS")
    name = name.replace("'", "")
    # Remove other common punctuation and replace with space
    name = re.sub(r'[.,\-&]', ' ', name)
    # Remove common legal suffixes and extra words
    suffixes = [r'\bINC\b', r'\bLTD\b', r'\bCO\b', r'\bCORP\b', r'\bCORPORATED\b', r'\bINCORPORATED\b', r'\bUSA\b', r'\bLLC\b']
    for suffix in suffixes:
        name = re.sub(suffix, '', name)
    # Compress multiple spaces and strip
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def normalize_goods_services(text):
    """
    Normalizes goods and services text: uppercase, strip, collapse spaces,
    and collapses repeated consecutive phrases.
    """
    if not text:
        return ""
    
    # Basic normalization
    text = re.sub(r'\s+', ' ', str(text).upper()).strip()
    
    # Requirement 1: Collapse repeated consecutive phrases
    # Example: "ABC ABC" -> "ABC"
    # We use a regex that matches a phrase followed by itself one or more times.
    # To keep it safe and avoid accidental over-matching, we look for space-separated identical blocks.
    # This is a bit tricky with regex for long phrases. A better way is a loop or specific split.
    
    words = text.split()
    if not words:
        return ""
        
    def collapse_repeats(word_list):
        n = len(word_list)
        # Try finding repeated sequences of length 1 to n/2
        for length in range(1, n // 2 + 1):
            for i in range(n - 2 * length + 1):
                phrase1 = word_list[i : i + length]
                phrase2 = word_list[i + length : i + 2 * length]
                if phrase1 == phrase2:
                    # Found a repeat. Collapse it and recurse.
                    return collapse_repeats(word_list[: i + length] + word_list[i + 2 * length :])
        return word_list

    collapsed_words = collapse_repeats(words)
    return " ".join(collapsed_words)

def normalize_mark_text(text):
    """
    Normalizes trademark text: uppercase, strip, punctuation normalization.
    """
    if not text:
        return ""
    # Uppercase
    text = str(text).upper()
    # Remove punctuation similar to owner but perhaps less aggressive if needed
    # For now, let's keep it consistent with owner normalization but specific to trademarks
    text = re.sub(r'[.,\-&]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def clean_registration_number(s):
    """
    Strips all non-digit characters for robust matching.
    """
    return "".join(re.findall(r'\d+', str(s or "")))

def compare_state_marks(json1_records: List[NormalizedRecord], 
                        json2_records: List[NormalizedRecord]) -> CategoryComparisonResult:
    """
    Comparator for STATE_MARKS. Anchor: registration_number.
    Modernized Approach (Option B): Normalized comparison + class only if both exist.
    """
    # Group JSON 1 by registration number
    json1_map = {}
    for r in json1_records:
        if r.registration_number:
            if r.registration_number not in json1_map: json1_map[r.registration_number] = []
            json1_map[r.registration_number].append(r)
    
    passed = 0
    failures = []
    report_lines = ["CATEGORY: STATE_MARKS GATE 2 REPORT", "="*40]

    for rec2 in json2_records:
        reg_no = rec2.registration_number
        rec_id = rec2.record_id
        
        candidates = json1_map.get(reg_no, [])
        rec1, is_ambiguous = _find_best_candidate(rec2, candidates, "STATE_MARKS")
        
        if is_ambiguous:
            failures.append(f"{rec_id} (Reg {reg_no}): Multiple equally valid JSON1 candidates found. Unable to uniquely identify matching record.")
            report_lines.append(f"FAIL: {rec_id} - Ambiguous candidates")
            continue

        if not rec1:
            failures.append(f"{rec_id}: No matching registration number '{reg_no}' found in JSON 1.")
            report_lines.append(f"FAIL: {rec_id} - No matching registration number found")
            continue
            
        match_errors = []
        field_results = []
        
        f_owner = _get_field_match_info(rec1.owner_raw, rec1.owner_norm, rec2.owner_raw, rec2.owner_norm, "Owner")
        field_results.append(f_owner)
        
        f_goods = _get_field_match_info(rec1.goods_raw, rec1.goods_norm, rec2.goods_raw, rec2.goods_norm, "Goods")
        field_results.append(f_goods)
        
        f_tm = _get_field_match_info(rec1.trademark_raw, rec1.trademark_norm, rec2.trademark_raw, rec2.trademark_norm, "Trademark")
        field_results.append(f_tm)
        
        # modernized: compare class only if both are non-empty
        if rec1.class_list and rec2.class_list and rec1.class_list != rec2.class_list:
            match_errors.append(f"Class: {rec1.class_list} != {rec2.class_list}")
            
        for fr in field_results:
            if fr["status"] == "FAIL":
                match_errors.append(fr["error"])
        
        if not match_errors:
            passed += 1
            is_normalized = any(fr["status"] == "NORMALIZED" for fr in field_results)
            status = "PASS_NORMALIZED" if is_normalized else "PASS_EXACT"
            report_lines.append(f"{status}: {rec_id} (Reg: {reg_no})")
            if is_normalized:
                for fr in field_results:
                    if fr["status"] == "NORMALIZED":
                        report_lines.append(f"  - {fr['info']}")
        else:
            failures.append(f"{rec_id} (Reg {reg_no}): " + " | ".join(match_errors))
            report_lines.append(f"FAIL/PARTIAL: {rec_id}")
            for err in match_errors: report_lines.append(f"  - {err}")

    return CategoryComparisonResult("STATE_MARKS", passed, len(json2_records), failures, report_lines)


def _score_candidate(rec2: NormalizedRecord, cand: NormalizedRecord, category: str) -> float:
    """
    Heuristic scoring for candidate selection.
    Weights: Trademark (+10), Owner (+8), Goods (+5), Class (+2), SIC (+1).
    """
    score = 0.0
    # Exact Trademark Match
    if rec2.trademark_norm == cand.trademark_norm:
        score += 10.0
    # Exact Owner Match
    if rec2.owner_norm == cand.owner_norm:
        score += 8.0
    # Exact Goods Match
    if rec2.goods_norm == cand.goods_norm:
        score += 5.0
    # Class match (only if both non-empty)
    if rec2.class_list and cand.class_list and rec2.class_list == cand.class_list:
        score += 2.0
    # Primary SIC (Business Name only)
    if category == "BUSINESS_NAME":
        if rec2.primary_sic and cand.primary_sic and rec2.primary_sic == cand.primary_sic:
            score += 1.0
    return score

def _find_best_candidate(rec2: NormalizedRecord, candidates: List[NormalizedRecord], category: str) -> Tuple[Optional[NormalizedRecord], bool]:
    """
    Evaluates every candidate, selects the unique best one.
    Returns (best_candidate, is_ambiguous).
    """
    if not candidates:
        return None, False
    
    # Deduplicate pool by object identity (if multiple tiers combined)
    seen = set()
    unique_candidates = []
    for c in candidates:
        if id(c) not in seen:
            seen.add(id(c))
            unique_candidates.append(c)

    if len(unique_candidates) == 1:
        return unique_candidates[0], False
        
    best_cand = None
    max_score = -1.0
    scores = []
    
    for cand in unique_candidates:
        s = _score_candidate(rec2, cand, category)
        scores.append((s, cand))
        if s > max_score:
            max_score = s
            best_cand = cand
            
    # Check for ambiguity: multiple candidates with the same max_score
    top_candidates = [c for s, c in scores if s == max_score]
    if len(top_candidates) > 1:
        return None, True
        
    return best_cand, False

def _get_field_match_info(raw1: str, norm1: str, raw2: str, norm2: str, field_name: str) -> Dict[str, Any]:
    """
    Analyzes matching depth: EXACT, NORMALIZED, PASS_SUBSTRING, PASS_SEMANTIC_NEAR or FAIL.
    """
    s1 = str(raw1 or "").strip()
    s2 = str(raw2 or "").strip()
    if s1 == s2:
        return {"status": "EXACT", "error": None, "info": None}
    
    n1 = str(norm1 or "").strip()
    n2 = str(norm2 or "").strip()
    if n1 == n2:
        return {
            "status": "NORMALIZED",
            "error": None,
            "info": f"{field_name}: [NORMALIZED MATCH] J1:'{s1}' | J2:'{s2}' (Norm: '{n1}')"
        }
    
    # Requirement 2: PASS_SUBSTRING
    if n1 and n2:
        if n1 in n2 or n2 in n1:
            return {
                "status": "PASS_SUBSTRING",
                "error": None,
                "info": f"{field_name}: [SUBSTRING MATCH] J1:'{s1}' | J2:'{s2}'"
            }

    # Requirement 2: PASS_SEMANTIC_NEAR (Conservative matching for common semantic expansions)
    # Wording like "mix" vs "mix for food preparation"
    if field_name == "Goods" and n1 and n2:
        # Check if the core meaning is preserved (very conservative)
        # Using a simple check: if 70% of words in the shorter string are in the longer string
        words1 = set(n1.split())
        words2 = set(n2.split())
        if words1 and words2:
            intersection = words1.intersection(words2)
            smaller_set = words1 if len(words1) < len(words2) else words2
            if len(intersection) / len(smaller_set) >= 0.75:
                # Also ensure the first word matches (common anchor)
                if n1.split()[0] == n2.split()[0]:
                    return {
                        "status": "PASS_SEMANTIC_NEAR",
                        "error": None,
                        "info": f"{field_name}: [SEMANTIC NEAR] J1:'{s1}' | J2:'{s2}'"
                    }

    return {
        "status": "FAIL",
        "error": f"{field_name}: [J1: '{s1}'] != [J2: '{s2}']",
        "info": None
    }


def compare_common_law(json1_records: List[NormalizedRecord], 
                       json2_records: List[NormalizedRecord]) -> CategoryComparisonResult:
    """
    Comparator for COMMON_LAW. 
    Implements tiered search: Owner+TM -> TM -> Owner -> Goods+TM.
    """
    # Build tiered maps for JSON 1
    maps = {
        "owner_tm": {},
        "tm": {},
        "owner": {},
        "goods_tm": {}
    }
    for r in json1_records:
        if r.owner_norm and r.trademark_norm:
            key = f"{r.owner_norm}|{r.trademark_norm}"
            if key not in maps["owner_tm"]: maps["owner_tm"][key] = []
            maps["owner_tm"][key].append(r)
        if r.trademark_norm:
            if r.trademark_norm not in maps["tm"]: maps["tm"][r.trademark_norm] = []
            maps["tm"][r.trademark_norm].append(r)
        if r.owner_norm:
            if r.owner_norm not in maps["owner"]: maps["owner"][r.owner_norm] = []
            maps["owner"][r.owner_norm].append(r)
        if r.goods_norm and r.trademark_norm:
            key = f"{r.goods_norm}|{r.trademark_norm}"
            if key not in maps["goods_tm"]: maps["goods_tm"][key] = []
            maps["goods_tm"][key].append(r)
            
    passed = 0
    failures = []
    report_lines = ["CATEGORY: COMMON_LAW GATE 2 REPORT", "="*40]

    for rec2 in json2_records:
        rec_id = rec2.record_id
        
        # Tiered lookup
        candidates = []
        if rec2.owner_norm and rec2.trademark_norm:
            candidates = maps["owner_tm"].get(f"{rec2.owner_norm}|{rec2.trademark_norm}", [])
        if not candidates and rec2.trademark_norm:
            candidates = maps["tm"].get(rec2.trademark_norm, [])
        if not candidates and rec2.owner_norm:
            candidates = maps["owner"].get(rec2.owner_norm, [])
        if not candidates and rec2.goods_norm and rec2.trademark_norm:
            candidates = maps["goods_tm"].get(f"{rec2.goods_norm}|{rec2.trademark_norm}", [])
            
        rec1, is_ambiguous = _find_best_candidate(rec2, candidates, "COMMON_LAW")
        
        if is_ambiguous:
            failures.append(f"{rec_id}: Multiple equally valid JSON1 candidates found. Unable to uniquely identify matching record.")
            report_lines.append(f"FAIL: {rec_id} - Ambiguous candidates")
            continue

        if rec1:
            match_errors = []
            field_results = []
            
            f_owner = _get_field_match_info(rec1.owner_raw, rec1.owner_norm, rec2.owner_raw, rec2.owner_norm, "Owner")
            field_results.append(f_owner)
            
            f_goods = _get_field_match_info(rec1.goods_raw, rec1.goods_norm, rec2.goods_raw, rec2.goods_norm, "Goods")
            field_results.append(f_goods)
            
            f_tm = _get_field_match_info(rec1.trademark_raw, rec1.trademark_norm, rec2.trademark_raw, rec2.trademark_norm, "Trademark")
            field_results.append(f_tm)

            if rec1.class_list and rec2.class_list and rec1.class_list != rec2.class_list:
                match_errors.append(f"Class: {rec1.class_list} != {rec2.class_list}")
                
            for fr in field_results:
                if fr["status"] == "FAIL":
                    match_errors.append(fr["error"])
            
            if not match_errors:
                passed += 1
                is_normalized = any(fr["status"] == "NORMALIZED" for fr in field_results)
                status = "PASS_NORMALIZED" if is_normalized else "PASS_EXACT"
                report_lines.append(f"{status}: {rec_id} (Owner: {rec2.owner_raw})")
                if is_normalized:
                    for fr in field_results:
                        if fr["status"] == "NORMALIZED":
                            report_lines.append(f"  - {fr['info']}")
            else:
                failures.append(f"{rec_id} (Owner: {rec2.owner_raw}): " + " | ".join(match_errors))
                report_lines.append(f"FAIL/PARTIAL: {rec_id}")
                for err in match_errors: report_lines.append(f"  - {err}")
        else:
            failures.append(f"{rec_id}: No matching JSON1 record found for '{rec2.trademark_raw}' / '{rec2.owner_raw}'")
            report_lines.append(f"FAIL: {rec_id} - No matching record found")

    return CategoryComparisonResult("COMMON_LAW", passed, len(json2_records), failures, report_lines)

def compare_web_common_law(json1_records: List[NormalizedRecord], 
                           json2_records: List[NormalizedRecord]) -> CategoryComparisonResult:
    """
    Comparator for WEB_COMMON_LAW.
    Tiered search: TM+Owner -> TM -> Owner.
    Requirement 4 & 5: Improved owner and goods logic.
    """
    json1_tm_owner_map = {}
    json1_tm_map = {}
    json1_owner_map = {}
    
    for r in json1_records:
        # Multi-key for Mark+Owner
        key = f"{r.trademark_norm or ''}|{r.owner_norm or ''}"
        if key not in json1_tm_owner_map: json1_tm_owner_map[key] = []
        json1_tm_owner_map[key].append(r)
        
        if r.trademark_norm:
            if r.trademark_norm not in json1_tm_map: json1_tm_map[r.trademark_norm] = []
            json1_tm_map[r.trademark_norm].append(r)
        if r.owner_norm:
            if r.owner_norm not in json1_owner_map: json1_owner_map[r.owner_norm] = []
            json1_owner_map[r.owner_norm].append(r)
            
    passed = 0
    failures = []
    report_lines = ["CATEGORY: WEB_COMMON_LAW GATE 2 REPORT", "="*40]

    # Requirement 4: Unreliable owner list
    UNRELIABLE_OWNERS = {"SHOP", "BAY", "PINTEREST", "POSTMATES", "MARKETPLACE", "SELLER", "HANDLE", "UNKNOWN"}

    for rec2 in json2_records:
        rec_id = rec2.record_id
        
        # Tiered lookup
        candidates = []
        key = f"{rec2.trademark_norm or ''}|{rec2.owner_norm or ''}"
        if key in json1_tm_owner_map:
            candidates = json1_tm_owner_map[key]
        if not candidates:
            if rec2.trademark_norm and rec2.trademark_norm in json1_tm_map:
                candidates.extend(json1_tm_map[rec2.trademark_norm])
            if rec2.owner_norm and rec2.owner_norm in json1_owner_map:
                candidates.extend(json1_owner_map[rec2.owner_norm])
        
        rec1, is_ambiguous = _find_best_candidate(rec2, candidates, "WEB_COMMON_LAW")
        
        if is_ambiguous:
            failures.append(f"{rec_id}: Multiple equally valid JSON1 candidates found. Unable to uniquely identify matching record.")
            report_lines.append(f"FAIL: {rec_id} - Ambiguous candidates")
            continue

        if rec1:
            match_errors = []
            field_results = []
            warning_only = False
            
            # Requirement 5: Missing Goods Handling
            if not rec2.goods_raw and rec1.goods_raw:
                field_results.append({"status": "SKIPPED_MISSING_JSON2", "error": None, "info": "Goods: [SKIPPED] Missing in JSON 2"})
            elif rec2.goods_raw:
                f_goods = _get_field_match_info(rec1.goods_raw, rec1.goods_norm, rec2.goods_raw, rec2.goods_norm, "Goods")
                field_results.append(f_goods)
            
            # Requirement 4: Owner Unreliable Check
            f_owner = _get_field_match_info(rec1.owner_raw, rec1.owner_norm, rec2.owner_raw, rec2.owner_norm, "Owner")
            if f_owner["status"] == "FAIL":
                # Check if JSON 2 owner is in unreliable list or is very short/generic
                if rec2.owner_norm in UNRELIABLE_OWNERS or not rec2.owner_norm or len(rec2.owner_norm) < 3:
                    f_owner["status"] = "OWNER_WARNING"
                    f_owner["info"] = f"Owner: [WARNING] Unreliable source '{rec2.owner_raw}'"
                
            field_results.append(f_owner)
            
            f_tm = _get_field_match_info(rec1.trademark_raw, rec1.trademark_norm, rec2.trademark_raw, rec2.trademark_norm, "Trademark")
            field_results.append(f_tm)

            if rec1.class_list and rec2.class_list and rec1.class_list != rec2.class_list:
                match_errors.append(f"Class: {rec1.class_list} != {rec2.class_list}")
                
            for fr in field_results:
                if fr["status"] == "FAIL":
                    match_errors.append(fr["error"])
                
            if not match_errors:
                passed += 1
                # Determine top status
                status_priority = ["PASS_EXACT", "PASS_NORMALIZED", "PASS_SUBSTRING", "PASS_SEMANTIC_NEAR", "SKIPPED_MISSING_JSON2", "OWNER_WARNING"]
                current_top = "PASS_EXACT"
                for fr in field_results:
                    if fr["status"] in status_priority:
                        if status_priority.index(fr["status"]) > status_priority.index(current_top):
                            current_top = fr["status"]
                
                report_lines.append(f"{current_top}: {rec_id} (Mark: {rec2.trademark_raw})")
                for fr in field_results:
                    if fr.get("info"):
                        report_lines.append(f"  - {fr['info']}")
            else:
                failures.append(f"{rec_id}: " + " | ".join(match_errors))
                report_lines.append(f"FAIL/PARTIAL: {rec_id}")
                for err in match_errors: report_lines.append(f"  - {err}")
        else:
            reason = f"No match found for Mark: '{rec2.trademark_raw}' and Owner: '{rec2.owner_raw}'"
            failures.append(f"{rec_id}: {reason}")
            report_lines.append(f"FAIL: {rec_id} - {reason}")

    return CategoryComparisonResult("WEB_COMMON_LAW", passed, len(json2_records), failures, report_lines)


def compare_business_name(json1_records: List[NormalizedRecord], 
                          json2_records: List[NormalizedRecord]) -> CategoryComparisonResult:
    """
    Comparator for BUSINESS_NAME. 
    Tiered search: TM+Owner -> TM -> Owner -> SIC.
    """
    maps = {
        "tm_owner": {},
        "tm": {},
        "owner": {},
        "sic": {}
    }
    for r in json1_records:
        if r.trademark_norm and r.owner_norm:
            key = f"{r.trademark_norm}|{r.owner_norm}"
            if key not in maps["tm_owner"]: maps["tm_owner"][key] = []
            maps["tm_owner"][key].append(r)
        if r.trademark_norm:
            if r.trademark_norm not in maps["tm"]: maps["tm"][r.trademark_norm] = []
            maps["tm"][r.trademark_norm].append(r)
        if r.owner_norm:
            if r.owner_norm not in maps["owner"]: maps["owner"][r.owner_norm] = []
            maps["owner"][r.owner_norm].append(r)
        if r.primary_sic:
            if r.primary_sic not in maps["sic"]: maps["sic"][r.primary_sic] = []
            maps["sic"][r.primary_sic].append(r)
    
    passed = 0
    total = len(json2_records)
    failures = []
    report_lines = ["CATEGORY: BUSINESS_NAME GATE 2 REPORT", "="*40]

    for rec2 in json2_records:
        rec_id = rec2.record_id
        
        # Tiered lookup
        candidates = []
        if rec2.trademark_norm and rec2.owner_norm:
            candidates = maps["tm_owner"].get(f"{rec2.trademark_norm}|{rec2.owner_norm}", [])
        if not candidates and rec2.trademark_norm:
            candidates = maps["tm"].get(rec2.trademark_norm, [])
        if not candidates and rec2.owner_norm:
            candidates = maps["owner"].get(rec2.owner_norm, [])
        if not candidates and rec2.primary_sic:
            candidates = maps["sic"].get(rec2.primary_sic, [])
            
        rec1, is_ambiguous = _find_best_candidate(rec2, candidates, "BUSINESS_NAME")
        
        if is_ambiguous:
            failures.append(f"{rec_id}: Multiple equally valid JSON1 candidates found. Unable to uniquely identify matching record.")
            report_lines.append(f"FAIL: {rec_id} - Ambiguous candidates")
            continue

        if rec1:
            match_errors = []
            field_results = []
            
            f_owner = _get_field_match_info(rec1.owner_raw, rec1.owner_norm, rec2.owner_raw, rec2.owner_norm, "Owner")
            field_results.append(f_owner)
            
            f_goods = _get_field_match_info(rec1.goods_raw, rec1.goods_norm, rec2.goods_raw, rec2.goods_norm, "Goods")
            field_results.append(f_goods)
            
            f_tm = _get_field_match_info(rec1.trademark_raw, rec1.trademark_norm, rec2.trademark_raw, rec2.trademark_norm, "Trademark")
            field_results.append(f_tm)

            if rec1.class_list and rec2.class_list and rec1.class_list != rec2.class_list:
                match_errors.append(f"Class: {rec1.class_list} != {rec2.class_list}")
                
            for fr in field_results:
                if fr["status"] == "FAIL":
                    match_errors.append(fr["error"])
                
            if not match_errors:
                passed += 1
                is_normalized = any(fr["status"] == "NORMALIZED" for fr in field_results)
                status = "PASS_NORMALIZED" if is_normalized else "PASS_EXACT"
                report_lines.append(f"{status}: {rec_id} (Mark: {rec2.trademark_raw})")
                if is_normalized:
                    for fr in field_results:
                        if fr["status"] == "NORMALIZED":
                            report_lines.append(f"  - {fr['info']}")
            else:
                failures.append(f"{rec_id}: " + " | ".join(match_errors))
                report_lines.append(f"FAIL/PARTIAL: {rec_id}")
                for err in match_errors: report_lines.append(f"  - {err}")
        else:
            failures.append(f"{rec_id}: No matching JSON1 record found for '{rec2.trademark_raw}' / SIC: '{rec2.primary_sic}'")
            report_lines.append(f"FAIL: {rec_id} - No matching record found")

    return CategoryComparisonResult("BUSINESS_NAME", passed, total, failures, report_lines)

def compare_web_domain(json1_records: List[NormalizedRecord], 
                       json2_records: List[NormalizedRecord]) -> CategoryComparisonResult:
    """
    Comparator for WEB_DOMAIN. Implements a 5-tier candidate selection architecture.
    Tier 1: TM + Owner
    Tier 2: TM + Goods
    Tier 3: TM only
    Tier 4: Owner only
    Tier 5: Goods only
    """
    # 1. Build Tiers of Maps for JSON 1
    # TM + Owner
    m1_tm_owner_raw = {}
    m1_tm_owner_norm = {}
    # TM + Goods
    m2_tm_goods_raw = {}
    m2_tm_goods_norm = {}
    # TM only
    m3_tm_raw = {}
    m3_tm_norm = {}
    # Owner only
    m4_owner_raw = {}
    m4_owner_norm = {}
    # Goods only
    m5_goods_raw = {}
    m5_goods_norm = {}

    def add_to_map(m, key, record):
        if not key: return
        if key not in m: m[key] = []
        m[key].append(record)

    for r in json1_records:
        # Tier 1
        add_to_map(m1_tm_owner_raw, f"{r.trademark_raw}|{r.owner_raw}", r)
        add_to_map(m1_tm_owner_norm, f"{r.trademark_norm}|{r.owner_norm}", r)
        # Tier 2
        add_to_map(m2_tm_goods_raw, f"{r.trademark_raw}|{r.goods_raw}", r)
        add_to_map(m2_tm_goods_norm, f"{r.trademark_norm}|{r.goods_norm}", r)
        # Tier 3
        add_to_map(m3_tm_raw, r.trademark_raw, r)
        add_to_map(m3_tm_norm, r.trademark_norm, r)
        # Tier 4
        add_to_map(m4_owner_raw, r.owner_raw, r)
        add_to_map(m4_owner_norm, r.owner_norm, r)
        # Tier 5
        add_to_map(m5_goods_raw, r.goods_raw, r)
        add_to_map(m5_goods_norm, r.goods_norm, r)

    passed = 0
    total_effective = 0
    total_skipped = 0
    total_extracted = len(json2_records)
    failures = []
    report_lines = []

    for rec2 in json2_records:
        if rec2.skip_from_gate2:
            total_skipped += 1
            continue
            
        total_effective += 1
        rec_id = rec2.record_id
        
        candidates = []
        
        # Tier 1: TM + Owner
        t1_raw_key = f"{rec2.trademark_raw}|{rec2.owner_raw}"
        t1_norm_key = f"{rec2.trademark_norm}|{rec2.owner_norm}"
        candidates = m1_tm_owner_raw.get(t1_raw_key) or m1_tm_owner_norm.get(t1_norm_key)
        
        # Tier 2: TM + Goods
        if not candidates:
            t2_raw_key = f"{rec2.trademark_raw}|{rec2.goods_raw}"
            t2_norm_key = f"{rec2.trademark_norm}|{rec2.goods_norm}"
            candidates = m2_tm_goods_raw.get(t2_raw_key) or m2_tm_goods_norm.get(t2_norm_key)
            
        # Tier 3: TM Only
        if not candidates:
            candidates = m3_tm_raw.get(rec2.trademark_raw) or m3_tm_norm.get(rec2.trademark_norm)
            
        # Tier 4: Owner Only
        if not candidates:
            candidates = m4_owner_raw.get(rec2.owner_raw) or m4_owner_norm.get(rec2.owner_norm)
            
        # Tier 5: Goods Only
        if not candidates:
            candidates = m5_goods_raw.get(rec2.goods_raw) or m5_goods_norm.get(rec2.goods_norm)

        rec1, is_ambiguous = _find_best_candidate(rec2, candidates or [], "WEB_DOMAIN")
        
        if is_ambiguous:
            failures.append(f"{rec_id} (TM: {rec2.trademark_raw}): Multiple equally valid JSON1 candidates found. Unable to uniquely identify matching record.")
            report_lines.append(f"FAIL: {rec_id} - Ambiguous candidates")
            continue

        if rec1:
            match_errors = []
            field_results = []
            
            f_owner = _get_field_match_info(rec1.owner_raw, rec1.owner_norm, rec2.owner_raw, rec2.owner_norm, "Owner")
            field_results.append(f_owner)
            
            f_goods = _get_field_match_info(rec1.goods_raw, rec1.goods_norm, rec2.goods_raw, rec2.goods_norm, "Goods")
            field_results.append(f_goods)
            
            f_tm = _get_field_match_info(rec1.trademark_raw, rec1.trademark_norm, rec2.trademark_raw, rec2.trademark_norm, "Trademark")
            field_results.append(f_tm)

            if rec1.class_list and rec2.class_list and rec1.class_list != rec2.class_list:
                match_errors.append(f"Class: {rec1.class_list} != {rec2.class_list}")
                
            for fr in field_results:
                if fr["status"] == "FAIL":
                    match_errors.append(fr["error"])
                
            if not match_errors:
                passed += 1
                # Status priority logic similar to WEB_COMMON_LAW
                status_priority = ["PASS_EXACT", "PASS_NORMALIZED", "PASS_SUBSTRING", "PASS_SEMANTIC_NEAR", "SKIPPED_MISSING_JSON2", "OWNER_WARNING"]
                current_top = "PASS_EXACT"
                for fr in field_results:
                    if fr["status"] in status_priority:
                        if status_priority.index(fr["status"]) > status_priority.index(current_top):
                            current_top = fr["status"]
                
                report_lines.append(f"{current_top}: {rec_id} (TM: {rec2.trademark_raw})")
                for fr in field_results:
                    if fr.get("info"):
                        report_lines.append(f"  - {fr['info']}")
            else:
                failures.append(f"{rec_id} (TM: {rec2.trademark_raw}): " + " | ".join(match_errors))
                report_lines.append(f"FAIL/PARTIAL: {rec_id}")
                for err in match_errors: report_lines.append(f"  - {err}")
        else:
            failures.append(f"{rec_id}: No domain match found for '{rec2.trademark_raw}' / '{rec2.owner_raw}'")
            report_lines.append(f"FAIL: {rec_id} - No match found for record")

    # Final summary assembly
    final_summary = [
        "CATEGORY: WEB_DOMAIN GATE 2 REPORT",
        "="*40,
        f"Total extracted records : {total_extracted}",
        f"Skipped (rule_filter)   : {total_skipped}",
        f"Effective compared      : {total_effective}",
        f"Passed                  : {passed}",
        f"Failed                  : {len(failures)}",
        "="*40
    ]
    full_report_lines = final_summary + report_lines

    return CategoryComparisonResult("WEB_DOMAIN", passed, total_effective, failures, full_report_lines,
                                    skipped=total_skipped, total_extracted=total_extracted)



def normalize_class_list(val):
    """
    Normalizes class list representations into a sorted tuple of integers.
    Handles list of ints: [29, 30]
    Handles string JSON array: '["001", "029"]'
    """
    if val is None:
        return ()
    
    if isinstance(val, str):
        try:
            # Try parsing as JSON first
            parsed = json.loads(val)
            if isinstance(parsed, list):
                val = parsed
            else:
                # If not list, split by comma if applicable
                val = [v.strip() for v in val.split(",") if v.strip()]
        except json.JSONDecodeError:
            # Fallback to simple split
            val = [v.strip() for v in val.split(",") if v.strip()]
            
    if isinstance(val, list):
        result = []
        for item in val:
            try:
                # Remove leading zeros and convert to int
                result.append(int(str(item).lstrip('0') or '0'))
            except (ValueError, TypeError):
                continue
        return tuple(sorted(list(set(result))))
        
    return ()


def extract_json1_records(data1) -> List[NormalizedRecord]:
    """
    Parses JSON 1 into canonical NormalizedRecord objects.
    """
    if not isinstance(data1, list):
        raise ValueError("Critical Error: JSON1 payload must be a list of records.")

    records = []
    for block in data1:
        if not isinstance(block, dict):
            continue # Or raise an error here as well
            
        source_type = block.get("source_type")
        if not source_type:
            continue

    for block in data1:
        source_type = block.get("source_type")
        category = CATEGORY_MAP.get(source_type)
        if not category:
            continue

        rec = NormalizedRecord(
            category=category,
            source_side="json1",
            owner_raw=str(block.get("owner", "")),
            owner_norm=normalize_owner(block.get("owner", "")),
            trademark_raw=str(block.get("trademark", "")),
            trademark_norm=normalize_mark_text(block.get("trademark", "")),
            goods_raw=str(block.get("goods_services", "")),
            goods_norm=normalize_goods_services(block.get("goods_services", "")),
            class_list=normalize_class_list(block.get("class_list")),
            registration_number=clean_registration_number(block.get("registration_number")),
            primary_sic=str(block.get("primary_sic", "")).strip().upper() if block.get("primary_sic") else None,
            raw_payload=block
        )
        records.append(rec)
    return records

def group_records_by_category(records: List[NormalizedRecord]) -> Dict[str, List[NormalizedRecord]]:
    """
    Groups canonical records into category buckets.
    """
    grouped = {}
    for rec in records:
        if rec.category not in grouped:
            grouped[rec.category] = []
        grouped[rec.category].append(rec)
    return grouped

def _extract_json2_state_marks(data2: Dict[str, Any]) -> List[NormalizedRecord]:
    """
    Extracts state mark records from 'state_summary_data'.
    Handles multiple JSON variants (CompuMark, Clarivate).
    """
    records = []
    items = data2.get("state_summary_data")
    if items and isinstance(items, list):
        for item in items:
            # Anchor detection: must have one of these to be counted
            r_id = item.get("us_identifier") or item.get("ST") or item.get("serialnum")
            if not r_id:
                continue
                
            # Perform clean mapping
            clean_payload = {k: v for k, v in item.items() if k not in ["Image_Base64", "state_image_path"]}
            
            records.append(NormalizedRecord(
                category="STATE_MARKS",
                source_side="json2",
                record_id=str(r_id),
                registration_number=clean_registration_number(item.get("registration_no")),
                owner_raw=str(item.get("owner_name", "")),
                owner_norm=normalize_owner(item.get("owner_name", "")),
                goods_raw=str(item.get("goods_services_description", "")),
                goods_norm=normalize_goods_services(item.get("goods_services_description", "")),
                trademark_raw=str(item.get("mark_text", "")),
                trademark_norm=normalize_mark_text(item.get("mark_text", "")),
                class_list=normalize_class_list(item.get("intl_class")),
                raw_payload=clean_payload
            ))
    return records

def _extract_json2_common_law(data2: Dict[str, Any]) -> List[NormalizedRecord]:
    records = []
    items = data2.get("records")
    if items and isinstance(items, list):
        return_type = str(data2.get("return_type", "")).lower()
        section = data2.get("section", "")
        vendor = data2.get("vendor_name", "")

        # Identify if this section is WEB or standard CL
        is_web = ("web_results" in section) or (vendor == "CompuMark" and "Web_start_page" in data2)
        
        for item in items:
            # Pattern 4 check: skip if it's actually a domain record sitting in 'records'
            if "dnn" in item or "bsn" in item:
                continue
                
            cat = "WEB_COMMON_LAW" if is_web else "COMMON_LAW"
            r_id = item.get("Doc No.") or item.get("COL") or item.get("web") or item.get("record_number") or "Unknown"
            
            records.append(NormalizedRecord(
                category=cat,
                source_side="json2",
                vendor_name=vendor,
                record_id=str(r_id),
                owner_raw=str(item.get("owner_name", "")),
                owner_norm=normalize_owner(item.get("owner_name", "")),
                goods_raw=str(item.get("goods_services", "")),
                goods_norm=normalize_goods_services(item.get("goods_services", "")),
                trademark_raw=str(item.get("mark_text", "")),
                trademark_norm=normalize_mark_text(item.get("mark_text", "")),
                class_list=normalize_class_list(item.get("nice_class")),
                raw_payload=item
            ))
    return records

def _extract_json2_business_names(data2: Dict[str, Any]) -> List[NormalizedRecord]:
    records = []
    items = data2.get("business_records")
    if items and isinstance(items, list):
        for item in items:
            records.append(NormalizedRecord(
                category="BUSINESS_NAME",
                source_side="json2",
                record_id=str(item.get("bsn", "Unknown")),
                primary_sic=str(item.get("primary_sic", "")).strip().upper(),
                owner_raw=str(item.get("owner_name", "")),
                owner_norm=normalize_owner(item.get("owner_name", "")),
                goods_raw=str(item.get("Goods/Services", "")),
                goods_norm=normalize_goods_services(item.get("Goods/Services", "")),
                trademark_raw=str(item.get("cited_mark", "")),
                trademark_norm=normalize_mark_text(item.get("cited_mark", "")),
                class_list=normalize_class_list(item.get("final_nice_class")),
                raw_payload=item
            ))
    return records

def _extract_json2_clarivate_web_common_law(data2: Dict[str, Any]) -> List[NormalizedRecord]:
    """
    Extends Pattern 3 support for Clarivate 'web_common_law_overview_data' variant.
    Maps to WEB_COMMON_LAW category.
    """
    records = []
    items = data2.get("web_common_law_overview_data", [])
    if not isinstance(items, list): return records
    
    for item in items:
        records.append(NormalizedRecord(
            category="WEB_COMMON_LAW",
            source_side="json2",
            record_id=str(item.get("Record Nr.", "Unknown")),
            primary_sic="",
            owner_raw="",
            owner_norm="",
            goods_raw="",
            goods_norm="",
            trademark_raw=str(item.get("Web Page Title", "")),
            trademark_norm=normalize_mark_text(item.get("Web Page Title", "")),
            class_list=[],
            raw_payload={k: v for k, v in item.items() if "image" not in k.lower()}
        ))
    return records

    return records

def _extract_json2_clarivate_common_law(data2: Dict[str, Any]) -> List[NormalizedRecord]:
    """
    Extends Pattern 2 support for Clarivate 'common_law_database_overview_data' variant.
    Maps to COMMON_LAW category.
    """
    records = []
    outer = data2.get("common_law_database_overview_data", {})
    if not isinstance(outer, dict): return records
    
    # Process both pools
    pool_identical = outer.get("Identical Names", [])
    pool_similar = outer.get("Similar Names", [])
    
    all_items = []
    if isinstance(pool_identical, list): all_items.extend(pool_identical)
    if isinstance(pool_similar, list): all_items.extend(pool_similar)
    
    for item in all_items:
        r_id = item.get("Nr.") or item.get("COL") or "Unknown"
        
        records.append(NormalizedRecord(
            category="COMMON_LAW",
            source_side="json2",
            record_id=str(r_id),
            owner_raw=str(item.get("owner_name", "")),
            owner_norm=normalize_owner(item.get("owner_name", "")),
            goods_raw=str(item.get("goods_services", "")),
            goods_norm=normalize_goods_services(item.get("goods_services", "")),
            trademark_raw=str(item.get("mark_text", "")),
            trademark_norm=normalize_mark_text(item.get("mark_text", "")),
            class_list=normalize_class_list(item.get("nice_class")),
            raw_payload=item
        ))
    return records

def _extract_json2_clarivate_business_names(data2: Dict[str, Any]) -> List[NormalizedRecord]:
    """
    Extends Pattern 4 support for Clarivate 'business_names_overview_data' variant.
    """
    records = []
    outer = data2.get("business_names_overview_data", {})
    if not isinstance(outer, dict): return records
    
    # Process both pools
    pool_identical = outer.get("Identical Names", [])
    pool_similar = outer.get("Similar Names", [])
    
    all_items = []
    if isinstance(pool_identical, list): all_items.extend(pool_identical)
    if isinstance(pool_similar, list): all_items.extend(pool_similar)
    
    for item in all_items:
        records.append(NormalizedRecord(
            category="BUSINESS_NAME",
            source_side="json2",
            record_id=str(item.get("BUS") or item.get("Nr.", "Unknown")),
            primary_sic=str(item.get("SIC Code", "")).strip().upper(),
            owner_raw=str(item.get("owner_name", "")),
            owner_norm=normalize_owner(item.get("owner_name", "")),
            goods_raw=str(item.get("Goods/Services", "")),
            goods_norm=normalize_goods_services(item.get("Goods/Services", "")),
            trademark_raw=str(item.get("mark_text", "")),
            trademark_norm=normalize_mark_text(item.get("mark_text", "")),
            class_list=normalize_class_list(item.get("final_nice_class")),
            raw_payload=item
        ))
    return records

def _extract_json2_business_names_variant(data2: Dict[str, Any]) -> List[NormalizedRecord]:
    """
    Extends Pattern 4 support for 'business_names_data' variant.
    """
    records = []
    items = data2.get("business_names_data")
    if items and isinstance(items, list):
        for item in items:
            records.append(NormalizedRecord(
                category="BUSINESS_NAME",
                source_side="json2",
                record_id=str(item.get("BUS", "Unknown")),
                primary_sic=str(item.get("SIC Code", "")).strip().upper(),
                owner_raw=str(item.get("owner_name", "")),
                owner_norm=normalize_owner(item.get("owner_name", "")),
                goods_raw=str(item.get("Goods/Services", "")),
                goods_norm=normalize_goods_services(item.get("Goods/Services", "")),
                trademark_raw=str(item.get("mark_text", "")),
                trademark_norm=normalize_mark_text(item.get("mark_text", "")),
                class_list=normalize_class_list(item.get("final_nice_class")),
                raw_payload=item
            ))
    return records

def _extract_json2_web_domains(data2: Dict[str, Any]) -> List[NormalizedRecord]:
    records = []
    items = data2.get("records")
    # Domains only exist in Pattern 4 which also has business_records
    if items and isinstance(items, list) and "business_records" in data2:
        for item in items:
            if "dnn" in item:
                is_rule_filter = item.get("goods_services_source") == "rule_filter"
                records.append(NormalizedRecord(
                    category="WEB_DOMAIN",
                    source_side="json2",
                    record_id=str(item.get("dnn", "Unknown")),
                    owner_raw=str(item.get("owner_nameD", "")),
                    owner_norm=normalize_owner(item.get("owner_nameD", "")),
                    trademark_raw=str(item.get("mark_text", "")),
                    trademark_norm=normalize_mark_text(item.get("mark_text", "")),
                    goods_raw=str(item.get("goods_services", "")),
                    goods_norm=normalize_goods_services(item.get("goods_services", "")),
                    class_list=normalize_class_list(item.get("nice_class")),
                    skip_from_gate2=is_rule_filter,
                    raw_payload=item
                ))
    return records

def _extract_json2_web_domains_variant(data2: Dict[str, Any]) -> List[NormalizedRecord]:
    """
    Extends Pattern 5 support for 'domain_names_data' variant.
    """
    records = []
    items = data2.get("domain_names_data")
    if items and isinstance(items, list):
        for item in items:
            records.append(NormalizedRecord(
                category="WEB_DOMAIN",
                source_side="json2",
                record_id=str(item.get("DN", "Unknown")),
                owner_raw=str(item.get("owner_nameD", "")),
                owner_norm=normalize_owner(item.get("owner_nameD", "")),
                trademark_raw=str(item.get("mark_text", "")),
                trademark_norm=normalize_mark_text(item.get("mark_text", "")),
                goods_raw=str(item.get("goods_services", "")),
                goods_norm=normalize_goods_services(item.get("goods_services", "")),
                class_list=normalize_class_list(item.get("nice_class")),
                raw_payload=item
            ))
    return records

def _extract_json2_clarivate_web_domains(data2: Dict[str, Any]) -> List[NormalizedRecord]:
    """
    Extends Pattern 5 support for Clarivate 'Identical Names' and 'Similar Names' variant.
    """
    records = []
    
    # Process both pools
    pool_identical = data2.get("Identical Names", [])
    pool_similar = data2.get("Similar Names", [])
    
    # Combined list for iteration
    all_items = []
    if isinstance(pool_identical, list): all_items.extend(pool_identical)
    if isinstance(pool_similar, list): all_items.extend(pool_similar)
    
    for item in all_items:
        records.append(NormalizedRecord(
            category="WEB_DOMAIN",
            source_side="json2",
            record_id=str(item.get("Nr.", "Unknown")),
            owner_raw=str(item.get("owner_nameD", "")),
            owner_norm=normalize_owner(item.get("owner_nameD", "")),
            trademark_raw=str(item.get("mark_text", "")),
            trademark_norm=normalize_mark_text(item.get("mark_text", "")),
            goods_raw=str(item.get("goods_services", "")),
            goods_norm=normalize_goods_services(item.get("goods_services", "")),
            class_list=normalize_class_list(item.get("nice_class")),
            raw_payload=item
        ))
    return records

def _extract_json2_federal_citations(data2: Dict[str, Any]) -> List[NormalizedRecord]:
    """
    Extends Pattern-6 support for 'corsearch' federal citations.
    Only counts records where DocID starts with 'UF-'.
    Maps to USPTO_MARKS category.
    """
    records = []
    results = data2.get("results", {})
    corsearch = results.get("corsearch", {})
    items = corsearch.get("Federal_Citations", [])
    
    if not isinstance(items, list): return records
    
    for item in items:
        doc_id = str(item.get("DocID", ""))
        # Core Requirement: Only count UF- records for Gate-1
        if doc_id.startswith("UF-"):
            records.append(NormalizedRecord(
                category="USPTO_MARKS",
                source_side="json2",
                record_id=doc_id,
                registration_number=clean_registration_number(item.get("Registration_Number")),
                owner_raw=str(item.get("Owner", "")),
                owner_norm=normalize_owner(item.get("Owner", "")),
                goods_raw=str(item.get("Goods_Services", "")),
                goods_norm=normalize_goods_services(item.get("Goods_Services", "")),
                trademark_raw=str(item.get("Trademark", "")),
                trademark_norm=normalize_mark_text(item.get("Trademark", "")),
                class_list=normalize_class_list(item.get("Class")),
                raw_payload=item
            ))
    return records

def extract_json2_records(data2) -> List[NormalizedRecord]:
    """
    Parses JSON 2 into canonical NormalizedRecord objects using helper extractors.
    """
    records = []
    
    # Validation: if keys exist, they must be lists
    for key in ["state_summary_data", "records", "business_records", "business_names_data", "domain_names_data", "Identical Names", "Similar Names"]:
        val = data2.get(key)
        if val is not None and not isinstance(val, list):
            raise ValueError(f"Invalid JSON 2: '{key}' must be a list if present.")

    records.extend(_extract_json2_state_marks(data2))
    records.extend(_extract_json2_common_law(data2))
    records.extend(_extract_json2_business_names(data2))
    records.extend(_extract_json2_business_names_variant(data2))
    records.extend(_extract_json2_web_domains(data2))
    records.extend(_extract_json2_web_domains_variant(data2))
    records.extend(_extract_json2_clarivate_web_domains(data2))
    records.extend(_extract_json2_clarivate_business_names(data2))
    records.extend(_extract_json2_clarivate_web_common_law(data2))
    records.extend(_extract_json2_clarivate_common_law(data2))
    records.extend(_extract_json2_federal_citations(data2))
    
    return records

def compare_category_counts(json1_by_cat: Dict[str, List[NormalizedRecord]], 
                            json2_by_cat: Dict[str, List[NormalizedRecord]]) -> Tuple[bool, str]:
    """
    Compares record counts per category.
    Returns (match_ok, failure_reason)
    """
    # Categories to check are those present in JSON 2
    # This preserves existing behavior where we only check what the current pattern reports
    relevant_categories = list(json2_by_cat.keys())
    
    failures = []
    print("\n--- GATE 1: CATEGORY COUNT VERIFICATION ---")
    for cat in relevant_categories:
        c1 = len(json1_by_cat.get(cat, []))
        c2 = len(json2_by_cat.get(cat, []))
        print(f"{cat:15}: JSON 1 = {c1:3}, JSON 2 = {c2:3}")
        if c1 != c2:
            failures.append(f"{cat}: JSON 1 count = {c1}, JSON 2 count = {c2}")
            
    if not failures:
        return True, ""
    else:
        return False, "; ".join(failures)
def run_gate2(json1_by_cat: Dict[str, List[NormalizedRecord]], 
              json2_by_cat: Dict[str, List[NormalizedRecord]]) -> List[CategoryComparisonResult]:
    """
    Orchestrates Gate 2 comparison for all categories present in JSON 2.
    """
    results = []
    
    # State Marks
    if "STATE_MARKS" in json2_by_cat:
        results.append(compare_state_marks(json1_by_cat.get("STATE_MARKS", []), json2_by_cat["STATE_MARKS"]))
        
    # Common Law
    if "COMMON_LAW" in json2_by_cat:
        results.append(compare_common_law(json1_by_cat.get("COMMON_LAW", []), json2_by_cat["COMMON_LAW"]))
        
    # Web Common Law
    if "WEB_COMMON_LAW" in json2_by_cat:
        results.append(compare_web_common_law(json1_by_cat.get("WEB_COMMON_LAW", []), json2_by_cat["WEB_COMMON_LAW"]))
        
    # Business Name
    if "BUSINESS_NAME" in json2_by_cat:
        results.append(compare_business_name(json1_by_cat.get("BUSINESS_NAME", []), json2_by_cat["BUSINESS_NAME"]))
        
    # Web Domain
    if "WEB_DOMAIN" in json2_by_cat:
        results.append(compare_web_domain(json1_by_cat.get("WEB_DOMAIN", []), json2_by_cat["WEB_DOMAIN"]))
        
    return results

def process_single_json2(json2_path: str, json1_by_cat: Dict[str, List[NormalizedRecord]]) -> Tuple[str, List[str], bool]:
    """
    Processes a single JSON2 file against pre-loaded JSON1 data.
    Returns (filename, report_lines, is_pass).
    """
    fname = os.path.basename(json2_path)
    print(f"\n{'='*60}")
    print(f"PROCESSING: {fname}")
    print(f"{'='*60}")
    
    try:
        with open(json2_path, "r", encoding="utf-8") as f:
            data2 = json.load(f)
        
        json2_records = extract_json2_records(data2)
        if not json2_records:
            msg = f"ERROR: No valid records found in {fname}"
            print(msg)
            return fname, [msg], False
            
        json2_by_cat = group_records_by_category(json2_records)
        
        # Gate 1
        match_ok, g1_reason = compare_category_counts(json1_by_cat, json2_by_cat)
        g1_status = "PASS" if match_ok else "FAIL"
        
        report_lines = [
            f"========================================",
            f"FILE: {fname}",
            f"========================================",
            f"Gate-1 Status: {g1_status}",
        ]
        if not match_ok:
            report_lines.append(f"Gate-1 Failure Reason: {g1_reason}")
            print(f"Gate-1 FAILED: {g1_reason}")
            # Even if Gate 1 fails, we return False but keep the report
            return fname, report_lines, False
            
        print("Gate-1 PASSED.")
        
        # Gate 2
        print("Starting Gate-2 Deep Field Comparison...")
        results = run_gate2(json1_by_cat, json2_by_cat)
        
        is_g2_pass = True
        g2_report_parts = []
        for res in results:
            # Console output
            print(f"\n  Category: {res.category}")
            if res.category == "WEB_DOMAIN" and res.total_extracted > 0:
                print(f"    Total extracted: {res.total_extracted} | Skipped: {res.skipped} | Effective: {res.total}")
                print(f"    Passed: {res.passed} | Failed: {len(res.failures)}")
            else:
                print(f"    Result: {res.passed}/{res.total} records passed.")
            
            if res.failures:
                print(f"    Failures/Mismatches:")
                for fail in res.failures:
                    print(f"      - {fail}")
            
            if res.passed != res.total:
                is_g2_pass = False
            
            # Report assembly
            g2_report_parts.extend(res.report_lines)
            g2_report_parts.append("")
            
        g2_status = "PASS" if is_g2_pass else "FAIL"
        report_lines.append(f"Gate-2 Status: {g2_status}")
        report_lines.append("-" * 40)
        report_lines.extend(g2_report_parts)
        
        return fname, report_lines, is_g2_pass
        
    except Exception as e:
        err_msg = f"CRITICAL ERROR processing {fname}: {e}"
        print(err_msg)
        return fname, [f"FILE: {fname}", err_msg], False

def main():
    if len(sys.argv) < 3:
        print("Usage: python comparition_v1.py <path_to_json1> <path_to_json2_file1> [path_to_json2_file2] ...")
        sys.exit(1)

    path1 = sys.argv[1]
    json2_paths = sys.argv[2:]

    try:
        with open(path1, "r", encoding="utf-8") as f: 
            data1 = json.load(f)
    except Exception as e:
        print(f"Error reading JSON1 file: {e}")
        sys.exit(1)

    # 1. Extraction and Grouping for JSON1 (Constant)
    try:
        json1_records = extract_json1_records(data1)
        json1_by_cat = group_records_by_category(json1_records)
    except Exception as e:
        print(f"JSON1 Extraction Error: {e}")
        sys.exit(1)

    all_file_reports = []
    final_summary_stats = []

    # 2. Process each JSON2 file
    for p2 in json2_paths:
        fname, report_lines, is_pass = process_single_json2(p2, json1_by_cat)
        all_file_reports.extend(report_lines)
        all_file_reports.append("\n" + "="*60 + "\n") # Big separator between files
        
        status_str = "PASS" if is_pass else "FAIL"
        final_summary_stats.append((fname, status_str))

    # 3. Generate Final Summary
    summary_block = [
        "==============================",
        "FINAL SUMMARY",
        "=============================="
    ]
    all_passed = True
    for fname, status in final_summary_stats:
        summary_block.append(f"{fname:30} : {status}")
        if status == "FAIL":
            all_passed = False
    
    # 4. Write Unified Report
    combined_report = ["GATE 2 UNIFIED COMPARISON REPORT", "="*40, ""]
    combined_report.extend(all_file_reports)
    combined_report.extend(summary_block)

    with open("comparison_report.txt", "w", encoding="utf-8") as rf:
        rf.write("\n".join(combined_report))
    
    print(f"\nUnified comparison report written to comparison_report.txt")
    
    print("\n" + "\n".join(summary_block))

    if all_passed:
        print("\nOVERALL STATUS: PASS")
        sys.exit(0)
    else:
        print("\nOVERALL STATUS: FAIL (One or more patterns failed)")
        sys.exit(1)

if __name__ == "__main__":
    main()
