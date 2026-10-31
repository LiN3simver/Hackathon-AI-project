import re

PLATE_PATTERN = re.compile(
    r"^(?:[A-Z]{2,3}-?\d{3,4}|\d{3,4}-?[A-Z]{2,3}|[A-Z0-9]{2,4}-[A-Z0-9]{2,4})$"
)


def normalize_plate(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = re.sub(
        r"[^A-Z0-9-]",
        "",
        value.upper().replace("—", "-").replace("_", "-"),
    )
    compact = cleaned.replace("-", "")
    if re.fullmatch(r"[A-Z]{2,3}\d{3,4}", compact):
        letters = re.match(r"[A-Z]+", compact)
        if letters:
            return f"{letters.group()}-{compact[len(letters.group()):]}"
    if re.fullmatch(r"\d{3,4}[A-Z]{2,3}", compact):
        digits = re.match(r"\d+", compact)
        if digits:
            return f"{digits.group()}-{compact[len(digits.group()):]}"
    if PLATE_PATTERN.fullmatch(cleaned):
        return cleaned
    return None


def is_valid_plate_label(value: str | None) -> bool:
    return normalize_plate(value) is not None


def normalize_ocr_plate(value: str | None) -> str | None:
    """Correct common OCR O/0, I/1, Z/2 confusions in numeric positions."""
    normalized = normalize_plate(value)
    if not value:
        return normalized
    compact = re.sub(r"[^A-Z0-9]", "", value.upper())
    # Modern car plates have three letters followed by four digits. Do this
    # before accepting the permissive legacy mixed-alphanumeric format.
    if len(compact) == 7 and not re.fullmatch(r"[A-Z]{3}[0-9]{4}", compact):
        prefix = compact[:3].translate(str.maketrans("0158", "OISB"))
        suffix = compact[3:].translate(str.maketrans("OIZ", "012"))
        if re.fullmatch(r"[A-Z]{3}", prefix) and re.fullmatch(r"[0-9]{4}", suffix):
            return f"{prefix}-{suffix}"
    if normalized:
        return normalized
    for letters, digits in ((2, 3), (2, 4), (3, 3), (3, 4)):
        if len(compact) == letters + digits:
            prefix, suffix = compact[:letters], compact[letters:]
            if re.fullmatch(r"[A-Z]+", prefix) and re.fullmatch(r"[0-9OIZ]+", suffix):
                corrected = normalize_plate(prefix + suffix.translate(str.maketrans("OIZ", "012")))
                if corrected:
                    return corrected
            prefix, suffix = compact[:digits], compact[digits:]
            if re.fullmatch(r"[0-9OIZ]+", prefix) and re.fullmatch(r"[A-Z]+", suffix):
                corrected = normalize_plate(prefix.translate(str.maketrans("OIZ", "012")) + suffix)
                if corrected:
                    return corrected
    return None
