"""
regex_parser.py
---------------
Reusable regex extraction engine for medical blood report parameters.

Architecture:
    raw text → parameter lookup → regex match → float value

Design decisions:
    - One universal extract_parameter() function, not one function per parameter
    - PARAMETER_ALIASES maps every known name/abbreviation to a canonical key
    - MULTILINE_PARAMETERS marks parameters whose value may be on the next line
    - _build_pattern() constructs the regex dynamically from aliases + context
    - No pandas, no ML, no UI — pure text → value pipeline

Changelog:
    v1.1 — Fixed FE false positive (word boundary + unicode alias)
           Fixed CRP None result (multiline-aware pattern)
           Fixed Gvozdje unicode matching (dot instead of \\W)
           Added MULTILINE_PARAMETERS architecture
"""

import re
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. ALIAS MAP
#    Maps every known name or abbreviation → canonical parameter key.
#    This is the only place you add new parameters or lab-specific names.
#
#    ALIAS WRITING RULES:
#    - Plain strings   → will be re.escape()'d automatically (safe for .-() etc.)
#    - Regex fragments → must contain a backslash; escaping is skipped for these
#    - For short/ambiguous aliases like "Fe" → add to WORD_BOUNDARY_ALIASES below
# ─────────────────────────────────────────────────────────────────────────────

PARAMETER_ALIASES: dict[str, list[str]] = {
    "WBC": [
        "WBC",
        "Leukociti",
        "eK-Leukociti",
        "Leukocytes",
        "White Blood Cell",
        "White Blood Cells",
    ],
    "RBC": [
        "RBC",
        "Eritrociti",
        "eK-Eritrociti",
        "Erythrocytes",
        "Red Blood Cell",
        "Red Blood Cells",
    ],
    "HGB": [
        "HGB",
        "HB",
        "Hemoglobin",
        "eK-Hemoglobin",
        "Haemoglobin",
        "Hgb",
    ],
    "HCT": [
        "HCT",
        "HTC",
        "Htc",
        "Hematokrit",
        "eK-Hematokrit",
        "Haematocrit",
        "Hematocrit",
    ],
    "MCV": [
        "MCV",
        "eK-MCV",
    ],
    "MCH": [
        "MCH",
        "eK-MCH",
    ],
    "MCHC": [
        "MCHC",
        "eK-MCHC",
    ],
    "PLT": [
        "PLT",
        "Trombociti",
        "eK-Trombociti",
        "Platelets",
        "Thrombocytes",
    ],
    "LYM": [
        "LYM",
        "LYM%",
        "eK-LYM",
        "ek-LYM",
        "Limfociti",
        "Lymphocytes",
        "LYMPH",
    ],
    "GRA": [
        "GRA",
        "GRA%",
        "ek-GRA",
        "Granulociti",
        "Granulocytes",
        "GRAN",
        "Neutrofili",
        "NEUT",
        # NOTE: "NEU" removed — too short, risks collision with longer medical words
    ],
    "MID": [
        "MID",
        "MID%",
        "ek-MID",
    ],
    "CRP": [
        "s-C-reaktivni protein",
        "C-reaktivni protein",
        "C-reactive protein",
        "CRP",
    ],
    "GLU": [
        "s- Glukoza",
        "s-Glukoza",
        "Glukoza",
        "Glucose",
        "GLU",
    ],
    "FER": [
        "s- Feritin",
        "s-Feritin",
        "Feritin",
        "Ferritin",
        "FER",
    ],
    "FE": [
        # FIX v1.1: Added explicit unicode variants of Gvozdje
        "s- Gvo\u017e\u0111e",     # s- Gvožđe  (actual PDF text)
        r"s\-\ Gvo.{1,4}e",        # regex: handles encoding variants
        "s-Gvo\u017e\u0111e",
        r"s\-Gvo.{1,4}e",
        "Gvo\u017e\u0111e",         # Gvožđe
        r"Gvo.{1,4}e",             # catches all unicode/ascii variants
        "Gvozdje",                  # ascii fallback (no diacritics)
        "Iron",
        "Fe",                       # SHORT — wrapped in \b by WORD_BOUNDARY_ALIASES
    ],
    "DIMER": [
        "cP- D-dimer",
        "cP-D-dimer",
        "D-dimer",
        "D dimer",
        "DIMER",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. WORD BOUNDARY ALIASES
#    Short aliases that must be wrapped in \b...\b in the regex pattern.
#
#    Why is \b necessary for "Fe"?
#
#    Without \b:  pattern "Fe" matches anywhere those two characters appear:
#        "Feritin"  → 'Fe' at position 0  → FALSE POSITIVE → 123.0
#        "FEU/mL"   → 'FE' at position 0  → FALSE POSITIVE → 0.5
#
#    With \b:  \bFe\b only matches when "Fe" is surrounded by non-word chars:
#        "s- Fe 15.0"   → matches ✓  → 15.0  (correct)
#        "Feritin 123"  → no match ✓ → None  (correct)
#        "FEU/mL"       → no match ✓ → None  (correct)
#
#    \b is a zero-width assertion (it consumes no characters).
#    It matches at the boundary between \w [a-zA-Z0-9_] and \W.
# ─────────────────────────────────────────────────────────────────────────────

WORD_BOUNDARY_ALIASES: set[str] = {
    "Fe",
    "HB",    # "HB" is inside "HGB" and "Hemoglobin" — boundary prevents collision
    "FER",   # "FER" matches inside "Ferozin" (iron test method name in some PDFs)
             # \bFER\b rejects "Ferozin" because 'i' after 'r' is a word char (no boundary)
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. MULTILINE PARAMETERS
#    Parameters whose label and numeric value may appear on different lines.
#
#    Why does this happen in PDF reports?
#    PDF layout engines sometimes wrap long parameter names across lines, or
#    place value columns on a separate line from the label column.
#
#    Real example from this project (pdfplumber output):
#        "s-C-reaktivni protein \n"    ← label line
#        "(CRP) 3.80 mg/L < 5.00\n"   ← value line
#
#    Standard pattern [^\d\n]{0,25} FAILS here:
#        [^\d\n] = match any char that is NOT a digit AND NOT a newline
#        → stops at \n → never reaches 3.80 → returns None
#
#    Multiline pattern [\s\S]{0,80}? WORKS:
#        \s  = whitespace (space, tab, \n, \r)
#        \S  = non-whitespace
#        [\s\S] = ANY character (including newlines)
#        {0,80}? = lazy quantifier — stops at first match, max 80 chars
#
#    Why lazy (?) instead of greedy?
#    Lazy stops as soon as the rest of the pattern matches.
#    Greedy would consume all 80 chars then backtrack — risks skipping past
#    the correct value and landing on the next parameter's number.
#
#    Why require decimal (\d+[.,]\d+) for multiline parameters?
#    Because [\s\S] crosses lines, it may pass through text containing
#    isolated digits (e.g., "(CRP)" contains no digit, but "109/L" does).
#    Requiring a decimal point (e.g., 3.80 not just 3) eliminates integer
#    artifacts from unit strings. Patient values almost always have decimals.
#
#    Why NOT use re.DOTALL globally?
#    DOTALL makes "." match newlines everywhere — all patterns become
#    multiline, dramatically increasing false positive risk for short
#    aliases like "Fe", "HB", "MID" which could jump to unrelated lines.
# ─────────────────────────────────────────────────────────────────────────────

MULTILINE_PARAMETERS: set[str] = {
    "CRP",    # "s-C-reaktivni protein\n(CRP) 3.80" — value is on next line
    "DIMER",  # some labs: "D-dimer\n1.24 ug FEU/mL"
    "FE",     # some labs: "s- Gvozdje\n15.0"
    # NOTE: FER intentionally NOT here.
    # In this PDF, "s- Feritin 123.0" is on one line — standard pattern works.
    # Multiline was causing "Ferozin" (iron method name, appears just above Feritin)
    # to match via the "FER" alias, then skip across the newline to grab "23.9"
    # from the reference range "m: 23.9-336.2" instead of the patient value 123.0.
    # Fix: FER moved to WORD_BOUNDARY_ALIASES (\bFER\b blocks "Ferozin" match).
}


# ─────────────────────────────────────────────────────────────────────────────
# 4. PATTERN BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_pattern(canonical_key: str) -> Optional[re.Pattern]:
    """
    Build a compiled regex pattern for the given canonical parameter key.

    Standard pattern:
        (?:alias1|alias2|...)  [^\\d\\n]{0,25}  (?<![<>])  (\\d+[.,]?\\d*)

    Multiline pattern (for MULTILINE_PARAMETERS):
        (?:alias1|alias2|...)  [\\s\\S]{0,80}?  (?<![<>])  (\\d+[.,]\\d+)

    Returns None if canonical_key is not in PARAMETER_ALIASES.
    """
    aliases = PARAMETER_ALIASES.get(canonical_key)
    if aliases is None:
        return None

    # Sort longest-first: ensures longer aliases win in alternation
    sorted_aliases = sorted(aliases, key=len, reverse=True)

    escaped_aliases = []
    for alias in sorted_aliases:
        if "\\" in alias:
            # Pre-built regex fragment — use as-is
            fragment = alias
        else:
            # Plain string — escape all regex metacharacters (. - ( ) [ ] etc.)
            fragment = re.escape(alias)
            # Wrap short ambiguous aliases in word boundaries
            if alias in WORD_BOUNDARY_ALIASES:
                fragment = r"\b" + fragment + r"\b"

        escaped_aliases.append(fragment)

    alias_group = "|".join(escaped_aliases)
    is_multiline = canonical_key in MULTILINE_PARAMETERS

    if is_multiline:
        # (?:(?![<>])[\s\S]) — each character in the skip zone must NOT be < or >
        # This prevents crossing a reference range boundary like "< 5.00"
        # and accidentally capturing the reference value instead of the patient value.
        pattern_str = (
            r"(?:" + alias_group + r")"
            r"(?:(?![<>])[\s\S]){0,80}?"   # cross newlines, block < >, lazy
            r"(\d+[.,]\d+)"                 # CAPTURE: decimal required
        )
    else:
        pattern_str = (
            r"(?:" + alias_group + r")"
            r"[^\d\n]{0,25}"    # skip non-digit non-newline, max 25 chars
            r"(?<![<>])"        # not immediately after < or >
            r"(\d+[.,]?\d*)"    # CAPTURE: integer or decimal accepted
        )

    return re.compile(pattern_str, re.IGNORECASE | re.UNICODE)


# ─────────────────────────────────────────────────────────────────────────────
# 5. ALIAS RESOLVER
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_alias(name: str) -> Optional[str]:
    """
    Given any alias or canonical key (case-insensitive), return the canonical key.

    Examples:
        _resolve_alias("Leukociti")   -> "WBC"
        _resolve_alias("wbc")         -> "WBC"
        _resolve_alias("eK-LYM")      -> "LYM"
        _resolve_alias("Gvozdje")     -> "FE"
        _resolve_alias("xyz")         -> None
    """
    name_lower = name.lower()
    for canonical_key, alias_list in PARAMETER_ALIASES.items():
        if canonical_key.lower() == name_lower:
            return canonical_key
        for alias in alias_list:
            # Skip regex fragments — they can't be compared as plain strings
            if "\\" not in alias and alias.lower() == name_lower:
                return canonical_key
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 6. CORE EXTRACTION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_parameter(text: str, parameter: str) -> Optional[float]:
    """
    Extract the numeric value of a medical parameter from raw PDF text.

    Args:
        text:      Raw string extracted from a laboratory PDF report.
        parameter: Parameter name or abbreviation (e.g. "WBC", "HGB", "PLT").
                   Case-insensitive. Any alias from PARAMETER_ALIASES works.

    Returns:
        Float value if found, None if not found or not parseable.

    Pipeline:
        1. Resolve input name -> canonical key via _resolve_alias()
        2. Build pattern via _build_pattern() (standard or multiline)
        3. Search the full text with pattern.search()
        4. Extract capture group (the number string)
        5. Normalize decimal separator (comma -> dot)
        6. Convert to float and return
    """
    resolved_key = _resolve_alias(parameter)
    if resolved_key is None:
        resolved_key = parameter.upper()

    pattern = _build_pattern(resolved_key)
    if pattern is None:
        escaped = re.escape(parameter)
        fallback_str = escaped + r"[^\d\n]{0,25}(?<![<>])(\d+[.,]?\d*)"
        pattern = re.compile(fallback_str, re.IGNORECASE | re.UNICODE)

    match = pattern.search(text)
    if match is None:
        return None

    raw_number = match.group(1).replace(",", ".")
    try:
        return float(raw_number)
    except ValueError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 7. BATCH EXTRACTION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_parameters(text: str) -> dict[str, Optional[float]]:
    """Extract all known parameters. Returns canonical_key -> float | None."""
    return {key: extract_parameter(text, key) for key in PARAMETER_ALIASES}


def extract_selected_parameters(
    text: str,
    parameters: list[str]
) -> dict[str, Optional[float]]:
    """Extract a specific subset of parameters by name or alias."""
    results = {}
    for param in parameters:
        canonical = _resolve_alias(param) or param.upper()
        results[canonical] = extract_parameter(text, param)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 8. TEST SUITE — run: python regex_parser.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Real pdfplumber output from NALAZ_Mirjana_Orlic.pdf
    # Key layout details that caused bugs:
    #   - CRP label and value are on different lines
    #   - "Ferozin" (iron test method) appears ABOVE "s- Feritin" in extracted text
    #     and contains "Fer" — without \bFER\b it matched Ferozin and grabbed 23.9
    #   - "m: 23.9-336.2" is the Feritin reference range, appears before the label line
    SAMPLE_TEXT = (
        "eK-Eritrociti (RBC) 4.26 1012/L 3.80-5.60\n"
        "eK-Leukociti (WBC) 10.8 109/L 4.0-10.6\n"
        "eK-Hemoglobin (HGB) 133 g/L 119-155\n"
        "eK-Hematokrit (HCT) 38.5 % 35.0-47.0\n"
        "eK-MCV 90 fL 80-100\n"
        "eK-MCH 31.2 pg 27.0-32.0\n"
        "eK-MCHC 345 g/L 320-360\n"
        "eK-Trombociti (PLT) 677 109/L 150-400\n"
        "eK-LYM 25.6 % 25.0-40.0\n"
        "ek-MID 6.9 % 0.0-12.0\n"
        "ek-GRA 67.5 % 50.0-70.0\n"
        "ek-LYM 2.8 109/L 1.19-3.35\n"
        "ek-MID 0.7 109/L 0.12-0.84\n"
        "ek-GRA 7.3 109/L 2.06-6.49\n"
        "s-C-reaktivni protein turbidimetrija\n"  # CRP label — value on next line
        "3.80 mg/L < 5.00\n"
        "(CRP) sa latex cest.\n"
        "ug imunoturbidimetrija\n"
        "cP- D-dimer 1.24 <0.5\n"                # FEU below, Fe must not match here
        "FEU/mL sa latex cesticama\n"
        "s- Glukoza 5.2 mmol/L 4.1-5.9 ; preko 60 god.:4.6-6.4\n"
        "s- Gvo\u017e\u0111e 15.0 umol/L odrasli zene: 5.8-31.7\n"  # Gvožđe
        "Ferozin\n"                               # iron method name — FER must NOT match this
        "m: 23.9-336.2; z 11.0-306.8\n"          # Feritin ref range — must NOT be captured
        "Referentne vrednosti su\n"
        "s- Feritin 123.0 ng/mL prilagodjene novoj metodi CLIA\n"  # correct FER value
    )

    EXPECTED = {
        "WBC":   10.8,
        "RBC":   4.26,
        "HGB":   133.0,
        "HCT":   38.5,
        "MCV":   90.0,
        "MCH":   31.2,
        "MCHC":  345.0,
        "PLT":   677.0,
        "LYM":   25.6,
        "GRA":   67.5,
        "MID":   6.9,
        "CRP":   3.80,   # BUG 2 FIXED: was None
        "GLU":   5.2,
        "FER":   123.0,
        "FE":    15.0,   # BUG 1 FIXED: was 123.0
        "DIMER": 1.24,
    }

    # ── Regression test ──────────────────────────────────────────────────────
    print("=" * 62)
    print("  REGRESSION TEST — all parameters")
    print("=" * 62)

    all_results = extract_all_parameters(SAMPLE_TEXT)
    passed = failed = 0

    for key, expected_val in EXPECTED.items():
        got = all_results.get(key)
        ok = (got == expected_val)
        icon = "PASS" if ok else "FAIL"
        note = ""
        if key == "CRP" and ok:   note = "  <- BUG 2 FIXED (multiline)"
        if key == "FE"  and ok:   note = "  <- BUG 1 FIXED (false positive)"
        print(f"  {'v' if ok else 'x'} {icon}  {key:<6}  got={str(got):<8}  expected={expected_val}{note}")
        passed += 1 if ok else 0
        failed += 0 if ok else 1

    print()
    print(f"  Result: {passed}/{passed+failed} passed {'OK' if failed == 0 else '--- FAILURES ABOVE'}")

    # ── Alias test ───────────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("  ALIAS TEST — non-canonical names resolve correctly")
    print("=" * 62)
    alias_cases = [
        ("Leukociti",          "WBC",  10.8),
        ("Hemoglobin",         "HGB",  133.0),
        ("Trombociti",         "PLT",  677.0),
        ("Gvo\u017e\u0111e",  "FE",   15.0),
        ("C-reaktivni protein","CRP",  3.80),
    ]
    for alias, expected_key, expected_val in alias_cases:
        resolved = _resolve_alias(alias)
        got = extract_parameter(SAMPLE_TEXT, alias)
        ok = (got == expected_val and resolved == expected_key)
        print(f"  {'v' if ok else 'x'}  '{alias}' -> '{resolved}' = {got}  (expected {expected_val})")

    # ── False positive test ───────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("  FALSE POSITIVE TEST — must NOT match")
    print("=" * 62)
    fp_cases = [
        ("FE",  "FEU/mL <0.5",             None,  "Fe inside FEU/mL unit"),
        ("FE",  "s- Feritin 123.0 ng/mL",  None,  "Fe inside Feritin"),
        ("CRP", "CRP result < 0.5",         None,  "< before value"),
        ("FER", "Ferozin\nm: 23.9-336.2",   None,  "FER inside Ferozin (iron method name)"),
    ]
    for param, test_text, expected, description in fp_cases:
        got = extract_parameter(test_text, param)
        ok = (got == expected)
        print(f"  {'v' if ok else 'x'}  {param} | '{test_text}' -> {got}  ({description})")