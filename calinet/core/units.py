import re

# Canonical BIDS unit strings: CMIXF-12 / SI symbols (plus BIDS examples like pixel/arbitrary)
# Micro can be either "u" or "µ" in BIDS examples; CMIXF recommends "u" for portability. :contentReference[oaicite:2]{index=2}
UNIT_MAP = {
    # Conductance / EDA
    "microsiemens": "uS",
    "micro siemens": "uS",
    "µs": "uS",
    "us": "uS",
    "uS": "uS",
    "millisiemens": "mS",
    "msiemens": "mS",
    "ms": "ms",   # note: handled specially below to avoid collision with millisiemens
    "siemens": "S",
    "s": "s",     # same caution; handled below

    # Voltage (EEG/ECG/EMG)
    "microvolt": "uV",
    "microvolts": "uV",
    "µv": "uV",
    "uv": "uV",
    "millivolt": "mV",
    "millivolts": "mV",
    "mv": "mV",
    "volt": "V",
    "volts": "V",
    "v": "V",

    # Current
    "ampere": "A",
    "amps": "A",
    "a": "A",

    # Resistance / impedance
    "ohm": "Ohm",
    "Ω": "Ohm",
    "kohm": "kOhm",
    "kΩ": "kOhm",
    "mohm": "MOhm",
    "mΩ": "mOhm",
    "mω": "mOhm",

    # Time
    "second": "s",
    "seconds": "s",
    "sec": "s",
    "s": "s",
    "millisecond": "ms",
    "milliseconds": "ms",
    "msec": "ms",
    "ms": "ms",
    "microsecond": "us",
    "µs_time": "us",
    "us_time": "us",

    # Frequency
    "hz": "Hz",
    "hertz": "Hz",

    # Length / distance
    "m": "m",
    "meter": "m",
    "metre": "m",
    "cm": "cm",
    "centimeter": "cm",
    "centimetre": "cm",
    "mm": "mm",
    "millimeter": "mm",
    "millimetre": "mm",
    "um": "um",
    "µm": "um",

    # Angle / temperature
    "degc": "oC",
    "°c": "oC",
    "celsius": "oC",
    "degree celsius": "oC",
    "rad": "rad",
    "radian": "rad",
    "deg": "deg",
    "°": "deg",

    # Common physio/eyetracking units in BIDS examples
    "pixel": "pixel",
    "px": "pixel",
    "arbitrary": "arbitrary",
    "au": "arbitrary",
    "a.u.": "arbitrary",

    # Dimensionless / percentages (not SI; still common)
    "%": "%",
    "percent": "%",
    "percentage": "%",

    # Rates often seen in metadata
    "bpm": "1/min",
    "beats/min": "1/min",
    "beats per minute": "1/min",
    "/min": "1/min",
    "per min": "1/min",
}

# Strings where case matters (SI prefixes)
_CANONICAL = set(UNIT_MAP.values())

def normalize_bids_unit(unit: str) -> str:
    """
    Normalize unit strings to BIDS/CMIXF-style unit symbols.
    Returns the original string if not recognized (so you can flag/handle custom units).
    """
    if unit is None:
        return unit

    u = str(unit).strip()
    if u == "":
        return u

    # Normalize unicode micro sign and whitespace
    u = u.replace("μ", "µ")  # normalize to micro sign; mapping will emit "u" form
    u = re.sub(r"\s+", " ", u).strip()

    # Special-case: "ms" can mean milliseconds or millisiemens depending on context.
    # If user typed out "millisiemens"/"mS" we map to mS; if it's exactly "ms" we treat as milliseconds.
    # You can override this by passing "mS" explicitly.
    if u.lower() in ("ms", "msec", "millisecond", "milliseconds"):
        return "ms"

    # If already canonical, keep it (preserve case)
    if u in _CANONICAL:
        return u

    key = u.lower()

    # disambiguate microseconds written as "µs" vs microsiemens "µS" when lowercased:
    # if original contains "µs" and not "siemens", treat as time
    if u in ("µs", "us") and "siem" not in key:
        return "us"

    # Direct lookup
    if key in UNIT_MAP:
        return UNIT_MAP[key]

    # Normalize patterns like "microV", "micro V", "u volt(s)"
    key2 = key.replace("-", " ").replace("_", " ")
    key2 = re.sub(r"\s+", " ", key2).strip()
    if key2 in UNIT_MAP:
        return UNIT_MAP[key2]

    # Common pattern: "<prefix><unit>" spelled out, e.g., "micro siemens"
    # (already covered by map; leave here for extension)

    return u  # unknown: treat as custom/needs review
    