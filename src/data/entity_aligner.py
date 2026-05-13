"""Cross-lingual entity alignment via multilingual sentence embeddings.

This module implements Gap #1 from the architecture: entity disambiguation
using both embedding similarity AND structural neighbor overlap.

Why two signals:
    Embedding similarity alone produces false positives when entities have
    similar names but different meanings (e.g., "Apple" the company vs
    "Apple" the fruit). By also checking that aligned entities share
    neighbors in the knowledge graph (structural signal), we filter out
    these ambiguous matches.

Flow:
    1. Extract unique entity lists from each language's triples
    2. Encode all entities with a multilingual sentence-transformer
    3. Compute pairwise cosine similarity matrices
    4. For high-similarity pairs: verify neighbor overlap (disambiguation)
    5. Output anchor pairs: confirmed cross-lingual entity matches

Model used:
    sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    - Supports 50+ languages including DE, JA, NL, EN
    - 12x smaller than mBERT — runs on CPU in ~50ms per batch
    - Auto-downloads from HuggingFace on first use (~470 MB)
"""

import json
import time
from pathlib import Path
from typing import Any

import torch
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    logger.error(
        "sentence-transformers not installed. "
        "Run: pip install sentence-transformers==2.7.0"
    )
    raise

from src.utils.text_utils import normalize_text


class EntityAligner:
    """Finds cross-lingual entity pairs using multilingual embeddings.

    Combines embedding cosine similarity with 1-hop neighbor overlap
    for disambiguation (Gap #1 fix). Uses English as the bridge language
    for transitive alignment, plus direct pairs for coverage.

    Args:
        config: Full configuration dictionary from config.yaml.

    Example:
        >>> from src.utils.config_loader import load_config
        >>> config = load_config()
        >>> aligner = EntityAligner(config)
        >>> pairs = aligner.align_all(processed_triples)
        >>> len(pairs)  # Number of confirmed cross-lingual matches
    """

    def __init__(self, config: dict) -> None:
        align_cfg = config["alignment"]
        self.model_name: str = align_cfg["model_name"]
        self.similarity_threshold: float = align_cfg["similarity_threshold"]
        self.min_neighbor_overlap: int = align_cfg["min_neighbor_overlap"]
        self.batch_size: int = align_cfg["batch_size"]
        self.processed_dir: Path = Path(config["data"]["processed_dir"])
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading alignment model: {self.model_name}")
        load_start = time.time()
        self.model: SentenceTransformer = SentenceTransformer(self.model_name)
        load_elapsed = time.time() - load_start
        logger.info(
            f"Alignment model loaded | time={load_elapsed:.1f}s "
            f"| model={self.model_name}"
        )

    def generate_embeddings(self, entities: list[str]) -> torch.Tensor:
        """Encode entity labels into dense multilingual embeddings.

        Uses the sentence-transformer model to produce fixed-size
        embeddings. Output vectors are L2-normalized to unit length
        so that dot product equals cosine similarity.

        Args:
            entities: List of entity label strings to encode.

        Returns:
            Tensor of shape (n_entities, embedding_dim) with unit-norm
            rows. Returns empty tensor if input list is empty.

        Example:
            >>> emb = aligner.generate_embeddings(["SAP SE", "SAP"])
            >>> emb.shape
            torch.Size([2, 384])
        """
        if not entities:
            logger.warning("generate_embeddings called with empty entity list")
            return torch.empty(0)

        logger.info(
            f"Generating embeddings | entities={len(entities)} "
            f"| batch_size={self.batch_size}"
        )

        embeddings = self.model.encode(
            entities,
            batch_size=self.batch_size,
            show_progress_bar=len(entities) > 100,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        # Ensure we have a 2D tensor
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)

        logger.info(
            f"Embeddings generated | shape={tuple(embeddings.shape)}"
        )
        return embeddings

    def compute_similarity_matrix(
        self, emb_a: torch.Tensor, emb_b: torch.Tensor
    ) -> torch.Tensor:
        """Compute pairwise cosine similarity between two sets of embeddings.

        Since embeddings are already L2-normalized, cosine similarity
        is simply the matrix dot product: emb_a @ emb_b.T.

        For memory safety: if the resulting matrix would exceed 500 MB,
        computation is done in row chunks to avoid OOM errors.

        Args:
            emb_a: Embeddings for entity set A, shape (n_a, dim).
            emb_b: Embeddings for entity set B, shape (n_b, dim).

        Returns:
            Similarity matrix of shape (n_a, n_b), values in [-1, 1].

        Example:
            >>> sim = aligner.compute_similarity_matrix(emb_de, emb_en)
            >>> sim.shape
            torch.Size([312, 387])
        """
        n_a, n_b = emb_a.shape[0], emb_b.shape[0]

        # Estimate memory: each float32 = 4 bytes
        matrix_bytes = n_a * n_b * 4
        max_bytes = 500 * 1024 * 1024  # 500 MB

        if matrix_bytes > max_bytes:
            logger.info(
                f"Large similarity matrix ({n_a}x{n_b} = "
                f"{matrix_bytes / 1e6:.0f} MB) — using chunked computation"
            )
            # Process in row chunks
            chunk_size = max(1, max_bytes // (n_b * 4))
            chunks: list[torch.Tensor] = []
            for start in range(0, n_a, chunk_size):
                end = min(start + chunk_size, n_a)
                chunk_sim = emb_a[start:end] @ emb_b.T
                chunks.append(chunk_sim)
            return torch.cat(chunks, dim=0)

        return emb_a @ emb_b.T

    def _get_neighbors(
        self, entity: str, triples: list[dict[str, str]]
    ) -> set[str]:
        """Get the 1-hop neighborhood of an entity from its triple set.

        An entity's neighbors are all entities connected to it by any
        relation (as head or tail). Labels are normalized before
        returning for consistent comparison.

        Args:
            entity: Entity label to find neighbors for.
            triples: List of triple dicts containing this entity.

        Returns:
            Set of normalized neighbor entity labels.
        """
        neighbors: set[str] = set()
        normalized_entity = normalize_text(entity)

        for triple in triples:
            head_norm = normalize_text(triple["head"])
            tail_norm = normalize_text(triple["tail"])

            if head_norm == normalized_entity:
                neighbors.add(tail_norm)
            elif tail_norm == normalized_entity:
                neighbors.add(head_norm)

        return neighbors

    def _compute_neighbor_overlap(
        self,
        entity_a: str,
        entity_b: str,
        triples_a: list[dict[str, str]],
        triples_b: list[dict[str, str]],
    ) -> int:
        """Count shared neighbors between two entities across languages.

        This is the Gap #1 disambiguation fix. Two entities that are truly
        the same real-world thing should share some graph neighbors even
        across languages (e.g., both connected to the same country or
        industry entities).

        Args:
            entity_a: Entity label from language A.
            entity_b: Entity label from language B.
            triples_a: All triples for language A.
            triples_b: All triples for language B.

        Returns:
            Count of overlapping normalized neighbor labels.

        Example:
            >>> overlap = aligner._compute_neighbor_overlap(
            ...     "SAP SE", "SAP", de_triples, en_triples)
            >>> overlap  # Both connected to Germany, software industry, etc.
            3
        """
        neighbors_a = self._get_neighbors(entity_a, triples_a)
        neighbors_b = self._get_neighbors(entity_b, triples_b)
        return len(neighbors_a & neighbors_b)

    def find_anchor_pairs(
        self,
        lang_a: str,
        lang_b: str,
        triples_a: list[dict[str, str]],
        triples_b: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        """Find confirmed cross-lingual entity matches between two languages.

        Pipeline:
          1. Extract unique entity lists from both triple sets
          2. Generate multilingual embeddings for both
          3. Compute cosine similarity matrix
          4. For each entity in lang_a: find best match in lang_b
          5. Apply similarity threshold
          6. Apply neighbor overlap disambiguation
          7. Log rejected pairs for analysis

        Args:
            lang_a: Source language code (e.g., "de").
            lang_b: Target language code (e.g., "en").
            triples_a: Processed triples for language A.
            triples_b: Processed triples for language B.

        Returns:
            List of confirmed anchor pair dicts with fields:
            entity_a, lang_a, entity_b, lang_b, similarity, neighbor_overlap.
        """
        # Extract unique entities
        entities_a = sorted(
            {t["head"] for t in triples_a} | {t["tail"] for t in triples_a}
        )
        entities_b = sorted(
            {t["head"] for t in triples_b} | {t["tail"] for t in triples_b}
        )

        if not entities_a or not entities_b:
            logger.warning(
                f"Empty entity list | {lang_a}={len(entities_a)} "
                f"| {lang_b}={len(entities_b)}"
            )
            return []

        logger.info(
            f"Aligning {lang_a.upper()}↔{lang_b.upper()} | "
            f"entities: {len(entities_a)} × {len(entities_b)}"
        )

        # Generate embeddings
        emb_a = self.generate_embeddings(entities_a)
        emb_b = self.generate_embeddings(entities_b)

        # Compute similarity matrix
        sim_matrix = self.compute_similarity_matrix(emb_a, emb_b)

        # Find best match for each entity in lang_a
        best_scores, best_indices = sim_matrix.max(dim=1)

        confirmed_pairs: list[dict[str, Any]] = []
        rejected_pairs: list[dict[str, Any]] = []

        for idx_a in range(len(entities_a)):
            score = best_scores[idx_a].item()
            idx_b = best_indices[idx_a].item()

            # Step 1: Similarity threshold
            if score < self.similarity_threshold:
                continue

            entity_a_name = entities_a[idx_a]
            entity_b_name = entities_b[idx_b]

            # Step 2: Neighbor overlap disambiguation
            overlap = self._compute_neighbor_overlap(
                entity_a_name, entity_b_name, triples_a, triples_b
            )

            pair_info: dict[str, Any] = {
                "entity_a": entity_a_name,
                "lang_a": lang_a,
                "entity_b": entity_b_name,
                "lang_b": lang_b,
                "similarity": round(score, 4),
                "neighbor_overlap": overlap,
            }

            if overlap >= self.min_neighbor_overlap:
                confirmed_pairs.append(pair_info)
            else:
                pair_info["rejection_reason"] = (
                    f"neighbor_overlap={overlap} < "
                    f"min_required={self.min_neighbor_overlap}"
                )
                rejected_pairs.append(pair_info)

        # Log results
        logger.info(
            f"Alignment {lang_a.upper()}↔{lang_b.upper()} | "
            f"confirmed={len(confirmed_pairs)} | "
            f"rejected={len(rejected_pairs)} | "
            f"coverage={len(confirmed_pairs)}/{len(entities_a)} "
            f"({100 * len(confirmed_pairs) / max(len(entities_a), 1):.1f}%)"
        )

        # Save rejected pairs for analysis
        if rejected_pairs:
            reject_path = (
                self.processed_dir
                / f"disambiguation_log_{lang_a}_{lang_b}.json"
            )
            with open(reject_path, "w", encoding="utf-8") as f:
                json.dump(rejected_pairs, f, ensure_ascii=False, indent=2)
            logger.debug(
                f"Rejected pairs saved | path={reject_path} "
                f"| count={len(rejected_pairs)}"
            )

        return confirmed_pairs

    def align_all(
        self, processed_triples: dict[str, list[dict[str, str]]]
    ) -> list[dict[str, Any]]:
        """Run full cross-lingual alignment across all language pairs.

        Alignment strategy:
          - Bridge pairs (via EN): DE↔EN, JA↔EN, NL↔EN
          - Direct pairs: DE↔JA, DE↔NL, JA↔NL
          - Deduplicate: same entity pair found via multiple paths

        Args:
            processed_triples: Dict mapping language codes to lists of
                processed triple dicts. Must include all target languages.

        Returns:
            Deduplicated list of all confirmed anchor pairs.
            Also saves to data/processed/anchor_pairs.json and
            data/processed/unmatched_log.json.

        Example:
            >>> pairs = aligner.align_all(processed_triples)
            >>> len(pairs)
            312
        """
        all_pairs: list[dict[str, Any]] = []

        # Define language pairs to align
        # Bridge pairs first (EN as hub), then direct pairs
        language_pairs = [
            ("de", "en"),
            ("ja", "en"),
            ("nl", "en"),
            ("de", "ja"),
            ("de", "nl"),
            ("ja", "nl"),
        ]

        for lang_a, lang_b in language_pairs:
            if lang_a not in processed_triples:
                logger.warning(f"No triples for {lang_a} — skipping {lang_a}↔{lang_b}")
                continue
            if lang_b not in processed_triples:
                logger.warning(f"No triples for {lang_b} — skipping {lang_a}↔{lang_b}")
                continue

            pairs = self.find_anchor_pairs(
                lang_a,
                lang_b,
                processed_triples[lang_a],
                processed_triples[lang_b],
            )
            all_pairs.extend(pairs)

        # Deduplicate: same entity pair might be found via different paths
        seen_pair_keys: set[tuple[str, str]] = set()
        unique_pairs: list[dict[str, Any]] = []
        for pair in all_pairs:
            # Normalize key: sort entity names for order-independent dedup
            key = tuple(sorted([pair["entity_a"], pair["entity_b"]]))
            if key not in seen_pair_keys:
                seen_pair_keys.add(key)
                unique_pairs.append(pair)

        duplicates_removed = len(all_pairs) - len(unique_pairs)
        if duplicates_removed > 0:
            logger.info(
                f"Anchor deduplication | before={len(all_pairs)} "
                f"| after={len(unique_pairs)} | removed={duplicates_removed}"
            )

        # Save anchor pairs
        anchor_path = self.processed_dir / "anchor_pairs.json"
        with open(anchor_path, "w", encoding="utf-8") as f:
            json.dump(unique_pairs, f, ensure_ascii=False, indent=2)
        logger.info(f"Anchor pairs saved | path={anchor_path} | count={len(unique_pairs)}")

        # Compute and save unmatched entities
        self._save_unmatched_log(processed_triples, unique_pairs)

        # Log alignment coverage summary
        self._log_coverage_summary(processed_triples, unique_pairs)

        return unique_pairs

    def _save_unmatched_log(
        self,
        processed_triples: dict[str, list[dict[str, str]]],
        anchor_pairs: list[dict[str, Any]],
    ) -> None:
        """Save entities that were not aligned to any cross-lingual match.

        Args:
            processed_triples: All processed triples by language.
            anchor_pairs: Confirmed anchor pairs.
        """
        # Collect all aligned entities
        aligned_entities: set[str] = set()
        for pair in anchor_pairs:
            aligned_entities.add(normalize_text(pair["entity_a"]))
            aligned_entities.add(normalize_text(pair["entity_b"]))

        # Find unmatched per language
        unmatched: dict[str, list[str]] = {}
        for lang, triples in processed_triples.items():
            all_entities = (
                {t["head"] for t in triples} | {t["tail"] for t in triples}
            )
            lang_unmatched = [
                e for e in sorted(all_entities)
                if normalize_text(e) not in aligned_entities
            ]
            unmatched[lang] = lang_unmatched

        unmatched_path = self.processed_dir / "unmatched_log.json"
        with open(unmatched_path, "w", encoding="utf-8") as f:
            json.dump(unmatched, f, ensure_ascii=False, indent=2)

        total_unmatched = sum(len(v) for v in unmatched.values())
        logger.info(
            f"Unmatched entities saved | path={unmatched_path} "
            f"| total={total_unmatched}"
        )

    def _log_coverage_summary(
        self,
        processed_triples: dict[str, list[dict[str, str]]],
        anchor_pairs: list[dict[str, Any]],
    ) -> None:
        """Log alignment coverage percentages per language pair.

        Args:
            processed_triples: All processed triples by language.
            anchor_pairs: Confirmed anchor pairs.
        """
        # Count entities per language
        entities_per_lang: dict[str, set[str]] = {}
        for lang, triples in processed_triples.items():
            entities_per_lang[lang] = (
                {t["head"] for t in triples} | {t["tail"] for t in triples}
            )

        # Count aligned entities per language
        aligned_per_lang: dict[str, set[str]] = {
            lang: set() for lang in processed_triples
        }
        for pair in anchor_pairs:
            lang_a = pair["lang_a"]
            lang_b = pair["lang_b"]
            if lang_a in aligned_per_lang:
                aligned_per_lang[lang_a].add(pair["entity_a"])
            if lang_b in aligned_per_lang:
                aligned_per_lang[lang_b].add(pair["entity_b"])

        logger.info("--- Alignment Coverage Summary ---")
        for lang in sorted(entities_per_lang.keys()):
            total = len(entities_per_lang[lang])
            aligned = len(aligned_per_lang.get(lang, set()))
            pct = 100 * aligned / max(total, 1)
            logger.info(
                f"  {lang.upper()}→aligned: {aligned}/{total} "
                f"({pct:.1f}%) of {lang.upper()} entities matched"
            )
