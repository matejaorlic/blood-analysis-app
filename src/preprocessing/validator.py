"""
validator.py
------------
Validation layer for the blood analysis pipeline.

Position in pipeline:
    PDF → Text extraction → Regex parsing → [THIS FILE] → ML model

Responsibilities:
    1. Missing value detection  — which parameters failed to parse (None)
    2. Invalid number detection — which values are physiologically impossible
    3. Reference range check    — LOW / NORMAL / HIGH per parameter and sex
    4. Parsing failure analysis — distinguish "not in PDF" from "parser error"
    5. ML feature vector        — clean dict ready for model.predict()

Usage:
    from validator import validate
    report = validate(parsed_dict, sex="F")
    if report.ml_ready:
        prediction = model.predict(report.ml_feature_vector)

Output type: ValidationReport (dataclass)
    report.parameters           → per-parameter detail
    report.ml_ready             → bool: safe to run ML prediction
    report.ml_feature_vector    → dict[str, float] ready for model
    report.missing              → list of keys that were None
    report.invalid              → list of keys outside physiological limits
    report.warnings             → list of keys that are LOW or HIGH
    report.critical             → list of keys with impossible values
    report.summary              → single human-readable status line
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class ParameterStatus(str, Enum):
    """
    Status of a single parameter after validation.

    Inherits from str so it serializes cleanly to JSON and prints without
    the "ParameterStatus." prefix (e.g. "NORMAL" not "ParameterStatus.NORMAL").
    """
    MISSING  = "MISSING"   # value is None — parser returned nothing
    CRITICAL = "CRITICAL"  # value exists but outside physiological limits
                           # → almost certainly a parsing error
    LOW      = "LOW"       # below reference range — clinically low
    NORMAL   = "NORMAL"    # within reference range
    HIGH     = "HIGH"      # above reference range — clinically elevated
    SKIPPED  = "SKIPPED"   # parameter not in reference table (supplementary only)


# ─────────────────────────────────────────────────────────────────────────────
# 2. REFERENCE RANGES
#    Source: https://www.jugolab.rs/biohemija/krvna-slika
#    Format: (female_min, female_max, male_min, male_max)
#
#    LYM / GRA / MID are stored as PERCENTAGE values (% form from CBC diff).
#    This matches the regex parser output which reads the % column first.
# ─────────────────────────────────────────────────────────────────────────────

REFERENCE_RANGES: dict[str, tuple[float, float, float, float]] = {
    #          female_min  female_max  male_min  male_max
    "WBC":    (3.9,   10.6,   3.9,   10.6),
    "RBC":    (3.80,  5.60,   4.30,  5.70),
    "HGB":    (119.0, 155.0,  138.0, 175.0),
    "HCT":    (35.0,  47.0,   41.0,  53.0),
    "MCV":    (80.0,  100.0,  80.0,  100.0),
    "MCH":    (27.0,  32.0,   27.0,  32.0),
    "MCHC":   (320.0, 360.0,  320.0, 360.0),
    "PLT":    (150.0, 400.0,  150.0, 400.0),
    "LYM":    (25.0,  40.0,   25.0,  40.0),   # % of leukocytes
    "GRA":    (50.0,  70.0,   50.0,  70.0),   # % of leukocytes
    "MID":    (0.0,   12.0,   0.0,   12.0),   # % of leukocytes
    "CRP":    (0.0,   5.0,    0.0,   5.0),    # mg/L
    "GLU":    (4.1,   5.9,    4.1,   5.9),    # mmol/L
    "FER":    (11.0,  306.8,  23.9,  336.2),  # ng/mL
    "FE":     (5.8,   31.7,   11.6,  31.3),   # umol/L
    "DIMER":  (0.0,   0.5,    0.0,   0.5),    # ug FEU/mL
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. PHYSIOLOGICAL LIMITS
#    Absolute min/max that a living human can have.
#    Values OUTSIDE these bounds are parsing errors, not medical findings.
#
#    These are intentionally generous — they are NOT clinical alerts,
#    they are sanity checks against impossible numbers like WBC=1234.
# ─────────────────────────────────────────────────────────────────────────────

PHYSIOLOGICAL_LIMITS: dict[str, tuple[float, float]] = {
    #          absolute_min  absolute_max
    "WBC":    (0.1,    200.0),
    "RBC":    (0.5,    10.0),
    "HGB":    (10.0,   250.0),
    "HCT":    (5.0,    75.0),
    "MCV":    (50.0,   150.0),
    "MCH":    (10.0,   60.0),
    "MCHC":   (200.0,  420.0),
    "PLT":    (1.0,    2000.0),
    "LYM":    (0.0,    100.0),
    "GRA":    (0.0,    100.0),
    "MID":    (0.0,    100.0),
    "CRP":    (0.0,    500.0),
    "GLU":    (0.5,    60.0),
    "FER":    (1.0,    50000.0),
    "FE":     (0.5,    200.0),
    "DIMER":  (0.0,    100.0),
}


# ─────────────────────────────────────────────────────────────────────────────
# 4. ML CONFIGURATION
#    ML_REQUIRED: parameters the model needs — if any are missing or CRITICAL,
#    ml_ready = False and prediction is blocked.
#
#    ML_FALLBACK_VALUES: used when an optional parameter is MISSING.
#    Strategy: use the midpoint of the reference range (sex-neutral average).
#    This is a conservative imputation — it tells the model "we have no
#    information about this parameter", not "this parameter is normal".
#    The model was trained on synthetic data that includes all parameters,
#    so None cannot be passed directly.
# ─────────────────────────────────────────────────────────────────────────────

ML_REQUIRED: set[str] = {"WBC", "RBC", "HGB", "HCT", "PLT"}

ML_FALLBACK_VALUES: dict[str, float] = {
    # midpoint of (female_ref + male_ref) / 2 for optional parameters
    key: round((v[0] + v[1] + v[2] + v[3]) / 4, 2)
    for key, v in REFERENCE_RANGES.items()
}


# ─────────────────────────────────────────────────────────────────────────────
# 5. OUTPUT DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ParameterResult:
    """
    Validation result for a single blood parameter.

    Attributes:
        key        Canonical parameter name (e.g. "WBC")
        value      Parsed float value, or None if missing
        status     ParameterStatus enum value
        ref_min    Lower bound of reference range (sex-adjusted), or None
        ref_max    Upper bound of reference range (sex-adjusted), or None
        physio_ok  True if value is within physiological limits
                   False if value is impossible (likely parsing error)
                   None if value is missing (can't check)
        note       Human-readable explanation of the status
    """
    key:       str
    value:     Optional[float]
    status:    ParameterStatus
    ref_min:   Optional[float]
    ref_max:   Optional[float]
    physio_ok: Optional[bool]
    note:      str

    def is_abnormal(self) -> bool:
        """True if status is LOW, HIGH, CRITICAL, or MISSING."""
        return self.status in (
            ParameterStatus.LOW,
            ParameterStatus.HIGH,
            ParameterStatus.CRITICAL,
            ParameterStatus.MISSING,
        )

    def __repr__(self) -> str:
        val_str = f"{self.value}" if self.value is not None else "None"
        ref_str = f"[{self.ref_min}-{self.ref_max}]" if self.ref_min is not None else "N/A"
        return f"ParameterResult({self.key}: {val_str} {self.status} ref={ref_str})"


@dataclass
class ValidationReport:
    """
    Complete validation report for one blood analysis PDF.

    This is the output of validate() and the input to the ML model.

    Attributes:
        parameters         Per-parameter validation detail
        ml_ready           True if all ML_REQUIRED params are present and physio_ok
        ml_feature_vector  Clean dict[str, float] ready for model.predict()
                           Missing optional params are replaced with ML_FALLBACK_VALUES
                           Missing required params → ml_ready = False
        missing            Keys where parser returned None
        invalid            Keys outside physiological limits (parsing errors)
        warnings           Keys that are LOW or HIGH (clinically abnormal)
        critical           Keys that are CRITICAL (physiologically impossible)
        summary            One-line human-readable status
        sex                Sex used for reference range lookup ('M', 'F', or None)
    """
    parameters:        dict[str, ParameterResult]
    ml_ready:          bool
    ml_feature_vector: dict[str, float]
    missing:           list[str]
    invalid:           list[str]
    warnings:          list[str]
    critical:          list[str]
    summary:           str
    sex:               Optional[str]

    def abnormal_parameters(self) -> list[ParameterResult]:
        """Returns all ParameterResult objects that are not NORMAL."""
        return [p for p in self.parameters.values() if p.is_abnormal()]

    def __repr__(self) -> str:
        return (
            f"ValidationReport("
            f"ml_ready={self.ml_ready}, "
            f"missing={self.missing}, "
            f"warnings={self.warnings}, "
            f"critical={self.critical})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6. INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_reference_range(
    key: str,
    sex: Optional[str],
) -> tuple[Optional[float], Optional[float]]:
    """
    Return (ref_min, ref_max) for a parameter, adjusted for sex.

    Sex resolution strategy:
        "F" or "female" → female reference range
        "M" or "male"   → male reference range
        None or unknown → conservative range: min of both mins, max of both maxes
                          This is the widest possible range — reduces false alerts
                          when sex is unknown, at the cost of sensitivity.

    Returns (None, None) if the parameter is not in REFERENCE_RANGES.
    """
    if key not in REFERENCE_RANGES:
        return None, None

    f_min, f_max, m_min, m_max = REFERENCE_RANGES[key]

    if sex is not None and sex.upper() in ("F", "FEMALE"):
        return f_min, f_max
    elif sex is not None and sex.upper() in ("M", "MALE"):
        return m_min, m_max
    else:
        # Unknown sex: use the most inclusive (widest) range
        return min(f_min, m_min), max(f_max, m_max)


def _check_physiological(key: str, value: float) -> bool:
    """
    Return True if value is within physiological limits for this parameter.
    Return False if value is impossible (parsing error territory).
    """
    if key not in PHYSIOLOGICAL_LIMITS:
        return True   # unknown parameter: pass through, can't judge

    phys_min, phys_max = PHYSIOLOGICAL_LIMITS[key]
    return phys_min <= value <= phys_max


def _classify_parameter(
    key: str,
    value: Optional[float],
    sex: Optional[str],
) -> ParameterResult:
    """
    Run all validation checks on a single (key, value) pair.

    Decision tree:
        value is None
            → MISSING

        value exists but outside physiological limits
            → CRITICAL  (physio_ok = False)
              This is almost always a parsing error.

        value exists and physio_ok but outside reference range
            → LOW or HIGH

        value exists and within reference range
            → NORMAL

        key not in REFERENCE_RANGES
            → SKIPPED (supplementary parameter, no reference defined)
    """
    # ── Case 1: missing ──────────────────────────────────────────────────────
    if value is None:
        ref_min, ref_max = _get_reference_range(key, sex)
        return ParameterResult(
            key=key,
            value=None,
            status=ParameterStatus.MISSING,
            ref_min=ref_min,
            ref_max=ref_max,
            physio_ok=None,
            note=f"{key}: not found in PDF — parser returned None",
        )

    # ── Case 2: parameter not in any reference table ─────────────────────────
    if key not in REFERENCE_RANGES:
        return ParameterResult(
            key=key,
            value=value,
            status=ParameterStatus.SKIPPED,
            ref_min=None,
            ref_max=None,
            physio_ok=_check_physiological(key, value),
            note=f"{key}: no reference range defined — supplementary parameter",
        )

    # ── Case 3: physiological check ──────────────────────────────────────────
    physio_ok = _check_physiological(key, value)
    ref_min, ref_max = _get_reference_range(key, sex)

    if not physio_ok:
        phys_min, phys_max = PHYSIOLOGICAL_LIMITS[key]
        return ParameterResult(
            key=key,
            value=value,
            status=ParameterStatus.CRITICAL,
            ref_min=ref_min,
            ref_max=ref_max,
            physio_ok=False,
            note=(
                f"{key}={value} is outside physiological limits "
                f"[{phys_min} – {phys_max}] — likely a parsing error"
            ),
        )

    # ── Case 4: reference range check ────────────────────────────────────────
    if value < ref_min:
        status = ParameterStatus.LOW
        note = (
            f"{key}={value} is BELOW reference range [{ref_min} – {ref_max}]"
            + (f" (sex={sex})" if sex else " (sex unknown, widest range used)")
        )
    elif value > ref_max:
        status = ParameterStatus.HIGH
        note = (
            f"{key}={value} is ABOVE reference range [{ref_min} – {ref_max}]"
            + (f" (sex={sex})" if sex else " (sex unknown, widest range used)")
        )
    else:
        status = ParameterStatus.NORMAL
        note = f"{key}={value} is within reference range [{ref_min} – {ref_max}]"

    return ParameterResult(
        key=key,
        value=value,
        status=status,
        ref_min=ref_min,
        ref_max=ref_max,
        physio_ok=True,
        note=note,
    )


def _build_feature_vector(
    parameters: dict[str, ParameterResult],
    ml_ready: bool,
) -> dict[str, float]:
    """
    Build the ML-ready feature vector from validated parameters.

    Strategy:
        - NORMAL / LOW / HIGH → use actual value
        - MISSING (optional)  → use ML_FALLBACK_VALUES[key] (ref range midpoint)
        - MISSING (required)  → ml_ready is already False; include 0.0 as placeholder
        - CRITICAL            → treat as missing (parsing error, don't trust value)

    The feature vector always contains all keys in REFERENCE_RANGES,
    ensuring consistent shape for the ML model regardless of PDF completeness.
    """
    vector: dict[str, float] = {}

    for key in REFERENCE_RANGES:
        result = parameters.get(key)

        if result is None:
            # Key wasn't even in parsed_dict — use fallback
            vector[key] = ML_FALLBACK_VALUES[key]
            continue

        if result.status in (ParameterStatus.NORMAL,
                              ParameterStatus.LOW,
                              ParameterStatus.HIGH):
            # Trusted value — use it directly
            vector[key] = result.value

        elif result.status == ParameterStatus.CRITICAL:
            # Physiologically impossible — treat as missing, use fallback
            vector[key] = ML_FALLBACK_VALUES[key]

        else:
            # MISSING or SKIPPED
            vector[key] = ML_FALLBACK_VALUES.get(key, 0.0)

    return vector


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN PUBLIC FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def validate(
    parsed: dict[str, Optional[float]],
    sex: Optional[str] = None,
    age: Optional[int] = None,
) -> ValidationReport:
    """
    Validate a parsed blood analysis dictionary and return a ValidationReport.

    Args:
        parsed: Output of regex_parser.extract_all_parameters() or
                extract_selected_parameters(). Keys are canonical parameter
                names (e.g. "WBC", "HGB"), values are float or None.

        sex:    Patient sex for reference range selection.
                Accepted: "M", "F", "male", "female" (case-insensitive).
                None → widest (most inclusive) reference ranges used.

        age:    Patient age in years. Currently stored but not used for
                range adjustment. Reserved for future pediatric/senior ranges.

    Returns:
        ValidationReport dataclass (see class docstring for full field list).

    Example:
        from regex_parser import extract_all_parameters
        from validator import validate

        parsed = extract_all_parameters(raw_pdf_text)
        report = validate(parsed, sex="F")

        if report.ml_ready:
            prediction = model.predict(report.ml_feature_vector)
        else:
            print("Cannot predict:", report.missing, report.critical)
    """
    # Normalize sex input
    if sex is not None:
        sex = sex.strip().upper()
        if sex in ("FEMALE", "F", "ZENA", "Z"):
            sex = "F"
        elif sex in ("MALE", "M", "MUSKARAC", "M"):
            sex = "M"
        else:
            sex = None   # unrecognized → treat as unknown

    # ── Step 1: classify every parameter ─────────────────────────────────────
    parameters: dict[str, ParameterResult] = {}
    for key, value in parsed.items():
        parameters[key] = _classify_parameter(key, value, sex)

    # ── Step 2: categorize results ────────────────────────────────────────────
    missing  = [k for k, r in parameters.items() if r.status == ParameterStatus.MISSING]
    critical = [k for k, r in parameters.items() if r.status == ParameterStatus.CRITICAL]
    warnings = [k for k, r in parameters.items()
                if r.status in (ParameterStatus.LOW, ParameterStatus.HIGH)]
    invalid  = critical   # alias: invalid = physiologically impossible

    # ── Step 3: determine ml_ready ───────────────────────────────────────────
    # ML prediction is blocked if any REQUIRED parameter is:
    #   - MISSING (parser couldn't find it)
    #   - CRITICAL (value is physiologically impossible → don't trust it)
    required_problems = [
        k for k in ML_REQUIRED
        if k in parameters and parameters[k].status in (
            ParameterStatus.MISSING, ParameterStatus.CRITICAL
        )
    ] + [
        k for k in ML_REQUIRED if k not in parameters
    ]
    ml_ready = len(required_problems) == 0

    # ── Step 4: build feature vector ─────────────────────────────────────────
    ml_feature_vector = _build_feature_vector(parameters, ml_ready)

    # ── Step 5: build summary ─────────────────────────────────────────────────
    summary = _build_summary(ml_ready, missing, critical, warnings, required_problems)

    return ValidationReport(
        parameters=parameters,
        ml_ready=ml_ready,
        ml_feature_vector=ml_feature_vector,
        missing=missing,
        invalid=invalid,
        warnings=warnings,
        critical=critical,
        summary=summary,
        sex=sex,
    )


def _build_summary(
    ml_ready: bool,
    missing: list[str],
    critical: list[str],
    warnings: list[str],
    required_problems: list[str],
) -> str:
    """Build a single human-readable summary line for the ValidationReport."""
    if not ml_ready:
        reason = ", ".join(required_problems)
        return f"ML prediction BLOCKED — required parameters unavailable: {reason}"

    parts = []
    if critical:
        parts.append(f"CRITICAL (parsing errors): {', '.join(critical)}")
    if missing:
        parts.append(f"missing (optional): {', '.join(missing)}")
    if warnings:
        parts.append(f"abnormal: {', '.join(warnings)}")

    if not parts:
        return "All parameters valid — ML prediction ready"

    return "ML ready — " + " | ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# 8. TEST SUITE — python validator.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    # ── Test data: real Mirjana Orlic PDF values ──────────────────────────────
    REAL_PARSED = {
        "WBC":   10.8,
        "RBC":   4.26,
        "HGB":   133.0,
        "HCT":   38.5,
        "MCV":   90.0,
        "MCH":   31.2,
        "MCHC":  345.0,
        "PLT":   677.0,   # HIGH — above 400
        "LYM":   25.6,
        "GRA":   67.5,
        "MID":   6.9,
        "CRP":   3.8,
        "GLU":   5.2,
        "FER":   123.0,
        "FE":    15.0,
        "DIMER": 1.24,    # HIGH — above 0.5
    }

    # ── Test 1: real data, female patient ─────────────────────────────────────
    print("=" * 65)
    print("  TEST 1 — Real data, sex=F (Mirjana Orlic)")
    print("=" * 65)

    report = validate(REAL_PARSED, sex="F")

    print(f"\n  Summary: {report.summary}")
    print(f"  ml_ready: {report.ml_ready}")
    print(f"  sex used: {report.sex}")
    print(f"  missing:  {report.missing}")
    print(f"  warnings: {report.warnings}")
    print(f"  critical: {report.critical}")

    print("\n  Per-parameter detail:")
    print(f"  {'KEY':<8} {'VALUE':<10} {'STATUS':<10} {'REF RANGE':<20} {'NOTE'}")
    print("  " + "-" * 80)
    for key, result in report.parameters.items():
        val_str = str(result.value) if result.value is not None else "None"
        ref_str = f"[{result.ref_min}-{result.ref_max}]" if result.ref_min is not None else "N/A"
        flag = " <--" if result.is_abnormal() else ""
        print(f"  {key:<8} {val_str:<10} {result.status.value:<10} {ref_str:<20}{flag}")

    print(f"\n  ML feature vector ({len(report.ml_feature_vector)} keys):")
    for k, v in report.ml_feature_vector.items():
        print(f"    {k:<8} {v}")

    # ── Test 2: missing required parameter ────────────────────────────────────
    print()
    print("=" * 65)
    print("  TEST 2 — Missing required parameter (WBC=None)")
    print("=" * 65)

    broken = dict(REAL_PARSED)
    broken["WBC"] = None
    report2 = validate(broken, sex="F")
    print(f"\n  ml_ready: {report2.ml_ready}  (expected: False)")
    print(f"  missing:  {report2.missing}  (expected: ['WBC'])")
    print(f"  summary:  {report2.summary}")
    assert report2.ml_ready is False
    assert "WBC" in report2.missing
    print("  PASS")

    # ── Test 3: physiologically impossible value ───────────────────────────────
    print()
    print("=" * 65)
    print("  TEST 3 — CRITICAL: WBC=999.9 (impossible value)")
    print("=" * 65)

    critical_parsed = dict(REAL_PARSED)
    critical_parsed["WBC"] = 999.9
    report3 = validate(critical_parsed, sex="F")
    print(f"\n  WBC status: {report3.parameters['WBC'].status}  (expected: CRITICAL)")
    print(f"  ml_ready:   {report3.ml_ready}  (expected: False, WBC is required)")
    print(f"  critical:   {report3.critical}")
    print(f"  note: {report3.parameters['WBC'].note}")
    assert report3.parameters["WBC"].status == ParameterStatus.CRITICAL
    assert report3.ml_ready is False
    print("  PASS")

    # ── Test 4: unknown sex → widest ranges ───────────────────────────────────
    print()
    print("=" * 65)
    print("  TEST 4 — sex=None uses widest reference range")
    print("=" * 65)

    report4 = validate(REAL_PARSED, sex=None)
    rbc_result = report4.parameters["RBC"]
    print(f"\n  RBC value: {rbc_result.value}")
    print(f"  RBC ref range (sex=None): [{rbc_result.ref_min}-{rbc_result.ref_max}]")
    print(f"  (expected: [3.8-5.7], widest of F and M)")
    assert rbc_result.ref_min == 3.8
    assert rbc_result.ref_max == 5.7
    print("  PASS")

    # ── Test 5: all missing ────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  TEST 5 — All parameters None (empty parse)")
    print("=" * 65)

    empty = {k: None for k in REAL_PARSED}
    report5 = validate(empty, sex="F")
    print(f"\n  ml_ready: {report5.ml_ready}  (expected: False)")
    print(f"  missing count: {len(report5.missing)}  (expected: {len(empty)})")
    print(f"  summary: {report5.summary}")
    assert report5.ml_ready is False
    assert len(report5.missing) == len(empty)
    print("  PASS")

    # ── Test 6: feature vector fallback values ─────────────────────────────────
    print()
    print("=" * 65)
    print("  TEST 6 — Optional DIMER missing → fallback in feature vector")
    print("=" * 65)

    partial = dict(REAL_PARSED)
    partial["DIMER"] = None
    report6 = validate(partial, sex="F")
    dimer_in_vector = report6.ml_feature_vector.get("DIMER")
    expected_fallback = ML_FALLBACK_VALUES["DIMER"]
    print(f"\n  DIMER in feature vector: {dimer_in_vector}  (expected fallback: {expected_fallback})")
    print(f"  ml_ready: {report6.ml_ready}  (expected: True — DIMER not required)")
    assert dimer_in_vector == expected_fallback
    assert report6.ml_ready is True
    print("  PASS")

    print()
    print("=" * 65)
    print("  ALL TESTS PASSED")
    print("=" * 65)