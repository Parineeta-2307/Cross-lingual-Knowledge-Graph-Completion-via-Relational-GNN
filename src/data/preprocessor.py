"""Triple preprocessing: validation, deduplication, and script tagging.

This module takes raw triples from the SPARQL extractor and produces
clean, validated per-language JSON files. It handles:
  - Unicode normalization (NFKC) on all entity labels
  - Script detection and tagging for each entity
  - Triple validation: both head and tail must be non-empty
  - Deduplication: identical (head_id, relation, tail_id) triples removed
  - Entity filtering: entities with < min_triples_per_entity are dropped
  - Statistics logging: triple counts, script distribution, entity counts

Why this works:
    Raw SPARQL results often contain duplicates (from OPTIONAL joins),
    empty labels (when a label doesn't exist in the requested language),
    and inconsistent Unicode. This module normalizes everything into a
    clean, consistent format before alignment and graph construction.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any

from loguru import logger

from src.utils.text_utils import (
    normalize_unicode,
    detect_script,
    clean_entity_label,
    get_script_stats,
)


class TriplePreprocessor:
    """Validates, deduplicates, and normalizes extracted triples.

    Args:
        config: Full configuration dictionary (from config.yaml).

    Example:
        >>> import yaml
        >>> config = yaml.safe_load(open("config.yaml"))
        >>> preprocessor = TriplePreprocessor(config)
        >>> cleaned = preprocessor.process_language("de", raw_triples)
        >>> preprocessor.save_triples(cleaned, "de")
    """

    def __init__(self, config: dict) -> None:
        data_cfg = config["data"]
        self.min_triples_per_entity: int = data_cfg["min_triples_per_entity"]
        self.raw_dir: Path = Path(data_cfg["raw_dir"])
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Preprocessor initialized | min_triples={self.min_triples_per_entity} "
            f"| output_dir={self.raw_dir}"
        )

    def validate_triple(self, triple: dict[str, str]) -> bool:
        """Check if a triple has all required non-empty fields.

        A valid triple must have:
          - Non-empty head label and head_id
          - Non-empty tail label and tail_id
          - Non-empty relation
          - Non-empty language code

        Args:
            triple: Triple dict with head, head_id, relation, tail, tail_id, language.

        Returns:
            True if the triple is valid, False otherwise.

        Example:
            >>> t = {"head": "SAP", "head_id": "Q80994", "relation": "located_in",
            ...      "tail": "Germany", "tail_id": "Q183", "language": "de"}
            >>> preprocessor.validate_triple(t)
            True
        """
        required_fields = ["head", "head_id", "relation", "tail", "tail_id", "language"]
        for field in required_fields:
            if not triple.get(field, "").strip():
                return False
        return True

    def normalize_triple(self, triple: dict[str, str]) -> dict[str, str]:
        """Apply Unicode normalization and cleaning to a triple's labels.

        Args:
            triple: Raw triple dict.

        Returns:
            New triple dict with normalized head and tail labels.
        """
        return {
            "head": clean_entity_label(triple["head"], triple.get("language")),
            "head_id": triple["head_id"].strip(),
            "relation": triple["relation"].strip(),
            "tail": clean_entity_label(triple["tail"], triple.get("language")),
            "tail_id": triple["tail_id"].strip(),
            "language": triple["language"].strip(),
        }

    def deduplicate_triples(
        self, triples: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Remove duplicate triples based on (head_id, relation, tail_id).

        Two triples are considered identical if they have the same
        Wikidata QIDs and relation. Labels may differ (e.g., different
        normalization) but the underlying fact is the same.

        Args:
            triples: List of triple dicts.

        Returns:
            Deduplicated list (preserving first occurrence order).

        Example:
            >>> deduped = preprocessor.deduplicate_triples(triples)
            >>> len(deduped) <= len(triples)
            True
        """
        seen: set[tuple[str, str, str]] = set()
        unique: list[dict[str, str]] = []

        for triple in triples:
            key = (triple["head_id"], triple["relation"], triple["tail_id"])
            if key not in seen:
                seen.add(key)
                unique.append(triple)

        removed = len(triples) - len(unique)
        if removed > 0:
            logger.info(f"Deduplication | removed={removed} | remaining={len(unique)}")

        return unique

    def filter_sparse_entities(
        self, triples: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Remove triples involving entities with very few connections.

        Entities that appear in fewer than min_triples_per_entity triples
        are likely noise (obscure entities with almost no info in Wikidata).
        Removing them improves alignment and training quality.

        Args:
            triples: List of triple dicts.

        Returns:
            Filtered list with sparse entities removed.
        """
        # Count appearances of each entity (by QID)
        entity_counts: Counter = Counter()
        for triple in triples:
            entity_counts[triple["head_id"]] += 1
            entity_counts[triple["tail_id"]] += 1

        # Filter triples where BOTH head and tail meet the minimum threshold
        # We keep triples where at least one side is well-connected
        filtered: list[dict[str, str]] = []
        removed = 0
        for triple in triples:
            head_count = entity_counts[triple["head_id"]]
            tail_count = entity_counts[triple["tail_id"]]

            if head_count >= self.min_triples_per_entity or \
               tail_count >= self.min_triples_per_entity:
                filtered.append(triple)
            else:
                removed += 1

        if removed > 0:
            logger.info(
                f"Sparse entity filter | removed={removed} | remaining={len(filtered)} "
                f"| threshold={self.min_triples_per_entity}"
            )

        return filtered

    def add_script_tags(
        self, triples: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Add script type tags to each triple's head and tail labels.

        Tags are added as 'head_script' and 'tail_script' fields.
        This is used for:
          1. Verifying the data pipeline handles all scripts correctly
          2. Debugging alignment issues (e.g., latin entity in JA data)
          3. Statistics reporting

        Args:
            triples: List of triple dicts.

        Returns:
            Same triples with added 'head_script' and 'tail_script' fields.
        """
        for triple in triples:
            triple["head_script"] = detect_script(triple["head"])
            triple["tail_script"] = detect_script(triple["tail"])
        return triples

    def process_language(
        self, language: str, raw_triples: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Full preprocessing pipeline for one language's triples.

        Steps (in order):
          1. Validate (remove incomplete triples)
          2. Normalize Unicode (NFKC)
          3. Deduplicate (by QID + relation)
          4. Filter sparse entities
          5. Add script tags
          6. Log statistics

        Args:
            language: Language code (de, ja, nl, en).
            raw_triples: Raw triples from the SPARQL extractor.

        Returns:
            Cleaned, validated, deduplicated triples with script tags.
        """
        logger.info(
            f"--- Preprocessing: {language.upper()} | input_triples={len(raw_triples)} ---"
        )

        # Step 1: Validate
        valid = [t for t in raw_triples if self.validate_triple(t)]
        invalid_count = len(raw_triples) - len(valid)
        if invalid_count > 0:
            logger.warning(
                f"Validation | lang={language} | dropped={invalid_count} invalid triples"
            )

        # Step 2: Normalize Unicode
        normalized = [self.normalize_triple(t) for t in valid]

        # Step 3: Deduplicate
        deduped = self.deduplicate_triples(normalized)

        # Step 4: Filter sparse entities
        filtered = self.filter_sparse_entities(deduped)

        # Step 5: Add script tags
        tagged = self.add_script_tags(filtered)

        # Step 6: Log statistics
        self._log_stats(language, tagged)

        return tagged

    def _log_stats(self, language: str, triples: list[dict[str, str]]) -> None:
        """Log detailed statistics about processed triples.

        Args:
            language: Language code.
            triples: Processed triple list.
        """
        if not triples:
            logger.warning(f"Stats | lang={language} | NO TRIPLES after processing")
            return

        # Count unique entities and relations
        head_ids = {t["head_id"] for t in triples}
        tail_ids = {t["tail_id"] for t in triples}
        all_entity_ids = head_ids | tail_ids
        relations = {t["relation"] for t in triples}

        # Script distribution for entity labels
        all_labels = [t["head"] for t in triples] + [t["tail"] for t in triples]
        script_stats = get_script_stats(all_labels)

        # Relation type counts
        relation_counts = Counter(t["relation"] for t in triples)

        logger.info(
            f"Stats | lang={language} | triples={len(triples)} "
            f"| unique_entities={len(all_entity_ids)} "
            f"| unique_relations={len(relations)}"
        )
        logger.info(f"Stats | lang={language} | script_distribution={script_stats}")
        logger.info(f"Stats | lang={language} | relation_counts={dict(relation_counts)}")

        # Flag any unexpected script types for this language
        if language == "ja" and script_stats.get("latin", 0) > len(all_labels) * 0.5:
            logger.warning(
                f"Stats | lang=ja | >50% of labels are Latin script — "
                f"check if SERVICE wikibase:label is returning English fallback"
            )

    def save_triples(
        self, triples: list[dict[str, str]], language: str
    ) -> Path:
        """Save processed triples to a JSON file.

        Output format:
          - File: data/raw/{lang}_triples.json
          - Encoding: UTF-8 with ensure_ascii=False (preserves Japanese/German chars)
          - Pretty-printed with indent=2 for human readability

        Args:
            triples: Processed triple list.
            language: Language code (used in filename).

        Returns:
            Path to the saved JSON file.

        Example:
            >>> path = preprocessor.save_triples(cleaned_triples, "de")
            >>> path
            PosixPath('data/raw/de_triples.json')
        """
        output_path = self.raw_dir / f"{language}_triples.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(triples, f, ensure_ascii=False, indent=2)

        logger.info(
            f"Saved | lang={language} | path={output_path} "
            f"| triples={len(triples)} | size={output_path.stat().st_size:,} bytes"
        )
        return output_path

    def compute_overall_stats(
        self, all_triples: dict[str, list[dict[str, str]]]
    ) -> dict[str, Any]:
        """Compute cross-language statistics summary.

        Args:
            all_triples: Dict mapping language codes to triple lists.

        Returns:
            Statistics dictionary with counts per language and totals.

        Example:
            >>> stats = preprocessor.compute_overall_stats({"de": [...], "ja": [...]})
        """
        stats: dict[str, Any] = {
            "total_triples": 0,
            "per_language": {},
            "all_entities": set(),
            "all_relations": set(),
        }

        for lang, triples in all_triples.items():
            entity_ids = {t["head_id"] for t in triples} | {t["tail_id"] for t in triples}
            relations = {t["relation"] for t in triples}

            stats["per_language"][lang] = {
                "triples": len(triples),
                "entities": len(entity_ids),
                "relations": len(relations),
            }
            stats["total_triples"] += len(triples)
            stats["all_entities"].update(entity_ids)
            stats["all_relations"].update(relations)

        # Convert sets to counts for JSON serialization
        stats["total_unique_entities"] = len(stats.pop("all_entities"))
        stats["total_unique_relations"] = len(stats.pop("all_relations"))

        logger.info(
            f"Overall stats | total_triples={stats['total_triples']} "
            f"| unique_entities={stats['total_unique_entities']} "
            f"| unique_relations={stats['total_unique_relations']}"
        )
        for lang, lang_stats in stats["per_language"].items():
            logger.info(f"  {lang.upper()}: {lang_stats}")

        return stats
