"""Convert supported concentration units to parts per million."""

from math import isfinite


# Each factor converts the supplied value to ppm.
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

    Invalid values and unsupported units return ``None``.
    """
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None

    if not isfinite(numeric_value) or numeric_value <= 0:
        return None

    normalised_unit = str(unit or "").strip().lower().replace(" ", "")
    factor = _TO_PPM_FACTORS.get(normalised_unit)
    if factor is None:
        return None

    converted = numeric_value * factor
    return converted if isfinite(converted) else None
