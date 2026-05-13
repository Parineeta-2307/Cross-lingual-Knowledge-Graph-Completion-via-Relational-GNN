"""Unicode normalization and script detection utilities.

Why this module exists:
    This project handles text in 4 languages with 3 script systems:
      - German: Latin + umlauts (ä, ö, ü, ß)
      - Dutch: Latin + accented chars (ë, ü, ij)
      - Japanese: Hiragana + Katakana + Kanji (CJK ideographs)
      - English: Basic Latin

    Without NFKC normalization, the SAME entity can have different byte
    representations (e.g., "ü" as one codepoint vs "u" + combining diaeresis).
    This breaks deduplication and dictionary lookups silently.

    Script detection tells us which language family an entity belongs to,
    which is essential for alignment quality checks and debugging.

Gap #4 compliance:
    - All text normalized with unicodedata.normalize('NFKC', ...)
    - detect_script() classifies text into latin/japanese/mixed/other
    - Tested with: "SAP AG", "富士通", "Böing", mixed scripts
"""

import unicodedata
from typing import Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Unicode range constants for CJK script detection
# Using code-point ranges is faster than unicodedata.name() for every char
# ---------------------------------------------------------------------------
_CJK_RANGES: list[tuple[int, int]] = [
    (0x3040, 0x309F),   # Hiragana (Japanese phonetic)
    (0x30A0, 0x30FF),   # Katakana (Japanese phonetic, used for foreign words)
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs (Kanji / Chinese characters)
    (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    (0xFF65, 0xFF9F),   # Halfwidth Katakana
    (0x31F0, 0x31FF),   # Katakana Phonetic Extensions
]


def normalize_unicode(text: str) -> str:
    """Normalize text to NFKC form and strip whitespace.

    NFKC normalization does two things:
      1. Compatibility decomposition: fullwidth "Ａ" → "A", "㈱" → "(株)"
      2. Canonical composition: "u" + combining "¨" → "ü"

    This ensures the same entity always has identical byte representation,
    which is critical for deduplication and dictionary lookups.

    Args:
        text: Raw text string, possibly with inconsistent Unicode encoding.

    Returns:
        NFKC-normalized, stripped string. Empty string if input is None/empty.

    Example:
        >>> normalize_unicode("  ＳＡＰ  ")
        'SAP'
        >>> normalize_unicode("Böing")
        'Böing'
    """
    if not text:
        return ""
    return unicodedata.normalize("NFKC", text).strip()


def _is_cjk_codepoint(cp: int) -> bool:
    """Check if a Unicode code point falls in any CJK range.

    Args:
        cp: Integer Unicode code point.

    Returns:
        True if the code point is CJK (Japanese/Chinese/Korean).
    """
    return any(start <= cp <= end for start, end in _CJK_RANGES)


def _is_latin_char(char: str) -> bool:
    """Check if a character is Latin script (including accented variants).

    Uses unicodedata.name() to check for "LATIN" in the character's
    official Unicode name. This correctly identifies:
      - Basic Latin: A-Z, a-z
      - German umlauts: ä, ö, ü, ß
      - Dutch accented: ë, ü
      - Other Latin extended characters

    Args:
        char: Single character string.

    Returns:
        True if the character is Latin script.
    """
    try:
        name = unicodedata.name(char, "")
        return "LATIN" in name
    except ValueError:
        return False


def detect_script(text: str) -> str:
    """Detect the dominant script system in a text string.

    Examines each non-punctuation, non-digit character and classifies
    the overall text based on which scripts are present.

    Classification rules:
      - 'latin': Only Latin characters found (EN, DE, NL)
      - 'japanese': Only CJK/Hiragana/Katakana found (JA)
      - 'mixed': Both Latin AND CJK characters found
      - 'other': Neither Latin nor CJK (e.g., Cyrillic, Arabic)
      - 'empty': Empty or whitespace/punctuation-only string

    Args:
        text: Text string to analyze.

    Returns:
        Script classification: 'latin', 'japanese', 'mixed', 'other', 'empty'.

    Example:
        >>> detect_script("SAP AG")
        'latin'
        >>> detect_script("富士通")
        'japanese'
        >>> detect_script("SAP 富士通")
        'mixed'
        >>> detect_script("Böing")
        'latin'
    """
    if not text or not text.strip():
        return "empty"

    has_latin = False
    has_cjk = False

    for char in text:
        # Skip whitespace, punctuation, digits — they don't indicate script
        if char.isspace() or char.isdigit() or unicodedata.category(char).startswith("P"):
            continue
        # Also skip common symbols (currency, math, etc.)
        if unicodedata.category(char).startswith(("S", "Z", "C")):
            continue

        cp = ord(char)

        if _is_cjk_codepoint(cp):
            has_cjk = True
        elif _is_latin_char(char):
            has_latin = True
        # else: character is from another script (Cyrillic, Arabic, etc.)

    if has_latin and has_cjk:
        return "mixed"
    elif has_cjk:
        return "japanese"
    elif has_latin:
        return "latin"
    else:
        return "other"


def clean_entity_label(label: str, language: Optional[str] = None) -> str:
    """Clean and normalize an entity label from Wikidata.

    Applies NFKC normalization, strips whitespace, and collapses
    multiple spaces. Handles Wikidata URI prefixes if accidentally
    included in label fields.

    Args:
        label: Raw entity label from SPARQL query result.
        language: Optional language code (unused currently, reserved for
            future language-specific cleaning rules).

    Returns:
        Cleaned entity label string. Empty string if input is None/empty.

    Example:
        >>> clean_entity_label("  SAP SE  ")
        'SAP SE'
        >>> clean_entity_label("富士通株式会社")
        '富士通株式会社'
    """
    if not label:
        return ""

    # Step 1: NFKC normalization
    cleaned = normalize_unicode(label)

    # Step 2: Remove Wikidata URI prefixes if accidentally included
    wikidata_prefix = "http://www.wikidata.org/entity/"
    if cleaned.startswith(wikidata_prefix):
        cleaned = cleaned[len(wikidata_prefix):]

    # Step 3: Collapse multiple spaces into one
    cleaned = " ".join(cleaned.split())

    return cleaned


def extract_qid_from_uri(uri: str) -> Optional[str]:
    """Extract Wikidata QID from a full entity URI.

    Wikidata entities have URIs like:
        http://www.wikidata.org/entity/Q80994

    This extracts just 'Q80994'. QIDs are language-invariant identifiers —
    the same real-world entity has the same QID in every language.
    This is crucial for cross-lingual entity alignment verification.

    Args:
        uri: Full Wikidata entity URI string.

    Returns:
        QID string (e.g., 'Q80994') or None if format is unexpected.

    Example:
        >>> extract_qid_from_uri("http://www.wikidata.org/entity/Q80994")
        'Q80994'
        >>> extract_qid_from_uri("not-a-uri")
    """
    if not uri:
        return None

    prefix = "http://www.wikidata.org/entity/"
    if uri.startswith(prefix):
        qid = uri[len(prefix):]
        # Validate: QID must start with Q followed by digits
        if qid and qid[0] == "Q" and qid[1:].isdigit():
            return qid

    # Handle case where just the QID was passed directly
    if uri.startswith("Q") and len(uri) > 1 and uri[1:].isdigit():
        return uri

    return None


def get_script_stats(texts: list[str]) -> dict[str, int]:
    """Compute script distribution statistics for a list of entity names.

    Useful for verifying that the data pipeline is correctly handling
    all three script systems. The stats are logged during preprocessing.

    Args:
        texts: List of entity label strings to analyze.

    Returns:
        Dictionary mapping script types to counts.
        Keys: 'latin', 'japanese', 'mixed', 'other', 'empty'.

    Example:
        >>> get_script_stats(["SAP", "富士通", "Sony ソニー"])
        {'latin': 1, 'japanese': 1, 'mixed': 1}
    """
    stats: dict[str, int] = {}
    for text in texts:
        script = detect_script(text)
        stats[script] = stats.get(script, 0) + 1
    return stats


# ---------------------------------------------------------------------------
# Convenience alias: normalize_text → normalize_unicode
# Some modules reference normalize_text() per the spec naming convention.
# ---------------------------------------------------------------------------
normalize_text = normalize_unicode


def truncate_for_log(text: str, max_chars: int = 200) -> str:
    """Safely truncate a string for logging without breaking Unicode.

    Long entity labels, SPARQL queries, or error messages can overwhelm
    log output. This function truncates to max_chars and appends a
    marker so you know the string was cut.

    Args:
        text: String to potentially truncate.
        max_chars: Maximum character count before truncation.
            Defaults to 200.

    Returns:
        Original string if within limit, or truncated string with
        '...[truncated]' appended.

    Example:
        >>> truncate_for_log("short")
        'short'
        >>> truncate_for_log("a" * 300, max_chars=10)
        'aaaaaaaaaa...[truncated]'
    """
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...[truncated]"


def hash_string(text: str) -> str:
    """Compute SHA-256 hash of a string.

    Used as the primary cache key in the SQLite SPARQL cache.
    Deterministic: same input always produces the same hash.
    Case-sensitive: 'SAP' and 'sap' produce different hashes.

    Args:
        text: String to hash.

    Returns:
        Lowercase hex-encoded SHA-256 hash string (64 characters).

    Example:
        >>> hash_string("test")
        '9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08'
        >>> hash_string("test") == hash_string("test")
        True
        >>> hash_string("test") == hash_string("Test")
        False
    """
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
