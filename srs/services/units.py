"""Concentration-unit conversion helpers used by similarity analysis."""


# Every supported factor converts the submitted value to parts per million.
_TO_PPM_FACTORS = {
    "": 1.0,
    "ppm": 1.0,
    "mg/kg": 1.0,
    "mgkg": 1.0,
    "ug/g": 1.0,
    "µg/g": 1.0,
    "μg/g": 1.0,
    "g/t": 1.0,
    "gpt": 1.0,
    "ppb": 0.001,
    "ug/kg": 0.001,
    "µg/kg": 0.001,
    "μg/kg": 0.001,
    "ng/g": 0.001,
    "ppt": 0.000001,
    "ng/kg": 0.000001,
    "pct": 10000.0,
    "percent": 10000.0,
    "%": 10000.0,
    "wt%": 10000.0,
    "wtpct": 10000.0,
    "mg/g": 1000.0,
}


def concentration_to_ppm(value, unit):
    """
    Convert a supported concentration to ppm.

    Return None for missing, non-positive, non-numeric, or unsupported values.
    Unsupported units are excluded rather than silently compared on the wrong
    scale.
    """
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None

    if numeric_value <= 0:
        return None

    normalised_unit = str(unit or "").strip().lower().replace(" ", "")
    factor = _TO_PPM_FACTORS.get(normalised_unit)
    if factor is None:
        return None

    return numeric_value * factor
