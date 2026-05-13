"""End-to-end pipeline runner for Cross-lingual KG Completion.

Usage:
    # Default: sample mode (200 triples/query, fast dev iteration)
    python scripts/run_pipeline.py --phase 1

    # Full extraction (all available triples, takes 15-30 min)
    python scripts/run_pipeline.py --phase 1 --full

    # Force re-fetch from Wikidata (bypass cache)
    python scripts/run_pipeline.py --phase 1 --force-refresh

    # Combine flags
    python scripts/run_pipeline.py --phase 1 --full --force-refresh

Design decision:
    --sample is the DEFAULT behavior (not --full). This means you never
    accidentally wait 30 minutes during development. When you're ready
    for the real dataset, explicitly pass --full.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so `src.` imports work
# when running as: python scripts/run_pipeline.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from loguru import logger

from src.utils.logging_config import setup_logging
from src.data.sparql_extractor import WikidataSPARQLExtractor
from src.data.preprocessor import TriplePreprocessor
from src.data.entity_aligner import EntityAligner
from src.data.graph_builder import GraphBuilder


def load_config(config_path: str = "config.yaml") -> dict:
    """Load and return the YAML configuration file.

    Args:
        config_path: Path to config.yaml (relative to project root).

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config.yaml doesn't exist.
    """
    full_path = PROJECT_ROOT / config_path
    if not full_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {full_path}\n"
            f"Expected location: {PROJECT_ROOT / 'config.yaml'}"
        )

    with open(full_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"Config loaded | path={full_path}")
    return config


def run_phase_1(
    config: dict,
    sample_mode: bool = True,
    force_refresh: bool = False,
) -> None:
    """Phase 1: Extract triples from Wikidata and preprocess.

    Pipeline steps:
      1. Initialize SPARQL extractor with SQLite caching
      2. Query Wikidata for all 4 relation types × all languages
      3. Preprocess: validate, normalize, deduplicate, filter, tag
      4. Save per-language JSON files to data/raw/
      5. Print summary statistics

    Args:
        config: Full configuration dictionary.
        sample_mode: If True, fetch only sample_limit triples per query.
        force_refresh: If True, bypass SQLite cache.
    """
    mode_label = "SAMPLE (200 triples/query)" if sample_mode else "FULL (all triples)"
    logger.info(f"{'='*60}")
    logger.info(f"PHASE 1: Data Extraction + Preprocessing")
    logger.info(f"Mode: {mode_label}")
    logger.info(f"Force refresh: {force_refresh}")
    logger.info(f"{'='*60}")

    start_time = time.time()

    # --- Step 1: Extract raw triples from Wikidata ---
    logger.info("Step 1/3: Extracting triples from Wikidata...")
    extractor = WikidataSPARQLExtractor(config, force_refresh=force_refresh)
    raw_triples = extractor.extract_all_languages(sample_mode=sample_mode)

    # Log cache stats
    cache_stats = extractor.get_cache_stats()
    logger.info(f"Cache state: {cache_stats}")

    # --- Step 2: Preprocess triples ---
    logger.info("Step 2/3: Preprocessing triples...")
    preprocessor = TriplePreprocessor(config)

    processed_triples: dict[str, list[dict[str, str]]] = {}
    for lang, triples in raw_triples.items():
        processed = preprocessor.process_language(lang, triples)
        processed_triples[lang] = processed

    # --- Step 3: Save to JSON files ---
    logger.info("Step 3/3: Saving processed triples...")
    for lang, triples in processed_triples.items():
        preprocessor.save_triples(triples, lang)

    # --- Summary ---
    overall_stats = preprocessor.compute_overall_stats(processed_triples)
    elapsed = time.time() - start_time

    # Save stats for reference
    stats_path = Path(config["data"]["raw_dir"]) / "extraction_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(overall_stats, f, ensure_ascii=False, indent=2)

    logger.info(f"{'='*60}")
    logger.info(f"PHASE 1 COMPLETE")
    logger.info(f"Time elapsed: {elapsed:.1f}s")
    logger.info(f"Total triples: {overall_stats['total_triples']}")
    logger.info(f"Unique entities: {overall_stats['total_unique_entities']}")
    logger.info(f"Unique relations: {overall_stats['total_unique_relations']}")
    for lang, ls in overall_stats["per_language"].items():
        logger.info(f"  {lang.upper()}: {ls['triples']} triples, {ls['entities']} entities")
    logger.info(f"Stats saved to: {stats_path}")
    logger.info(f"{'='*60}")

    # Verification hints
    logger.info("VERIFICATION CHECKLIST:")
    logger.info(f"  [1] Check data/cache/wikidata_cache.db exists: "
                f"{'YES' if Path(config['data']['cache_dir'], config['data']['cache_db_name']).exists() else 'NO'}")
    for lang in config["data"]["languages"]:
        json_path = Path(config["data"]["raw_dir"]) / f"{lang}_triples.json"
        exists = json_path.exists()
        count = len(processed_triples.get(lang, []))
        logger.info(f"  [2] {json_path}: {'EXISTS' if exists else 'MISSING'} ({count} triples)")

    return processed_triples


def run_phase_2(
    config: dict,
    processed_triples: dict[str, list[dict[str, str]]] | None = None,
) -> None:
    """Phase 2: Entity alignment + unified graph construction.

    Steps:
      1. Load processed triples (from Phase 1 output or memory)
      2. Run cross-lingual entity alignment
      3. Build unified NetworkX graph
      4. Convert to PyTorch Geometric format
      5. Perform stratified edge-level splits
      6. Save all artifacts to data/processed/

    Args:
        config: Full configuration dictionary.
        processed_triples: If provided, use these triples directly.
            If None, load from data/raw/*.json files.
    """
    logger.info(f"{'='*60}")
    logger.info("PHASE 2: Entity Alignment + Graph Construction")
    logger.info(f"{'='*60}")

    start_time = time.time()

    # --- Step 1: Load triples (from memory or disk) ---
    if processed_triples is None:
        logger.info("Loading processed triples from disk...")
        processed_triples = {}
        raw_dir = Path(config["data"]["raw_dir"])
        for lang in config["data"]["languages"]:
            json_path = raw_dir / f"{lang}_triples.json"
            if not json_path.exists():
                logger.error(
                    f"Missing {json_path} — run Phase 1 first: "
                    f"python scripts/run_pipeline.py --phase 1"
                )
                return
            with open(json_path, "r", encoding="utf-8") as f:
                processed_triples[lang] = json.load(f)
            logger.info(f"Loaded {lang.upper()}: {len(processed_triples[lang])} triples")
    else:
        logger.info("Using in-memory triples from Phase 1")

    # --- Step 2: Entity alignment ---
    logger.info("Step 1/3: Running cross-lingual entity alignment...")
    aligner = EntityAligner(config)
    anchor_pairs = aligner.align_all(processed_triples)

    # --- Step 3: Build graph ---
    logger.info("Step 2/3: Building unified graph...")
    builder = GraphBuilder(config)
    G = builder.build_networkx_graph(processed_triples, anchor_pairs)

    # --- Step 4: Convert to PyG ---
    logger.info("Step 3/3: Converting to PyTorch Geometric format...")
    data, entity_to_id, relation_to_id = builder.networkx_to_pyg(G)

    # --- Step 5: Split edges ---
    train_data, val_data, test_data = builder.split_edges(data)

    # --- Step 6: Save ---
    builder.save_artifacts(
        data, train_data, val_data, test_data, entity_to_id, relation_to_id
    )

    elapsed = time.time() - start_time

    # --- Summary ---
    logger.info(f"{'='*60}")
    logger.info("PHASE 2 COMPLETE")
    logger.info(f"Time elapsed: {elapsed:.1f}s")
    logger.info(f"  Total entities:     {data.num_nodes}")
    logger.info(f"  Total edges:        {data.edge_index.shape[1]}")
    logger.info(f"  Anchor pairs:       {len(anchor_pairs)}")
    cold = int(data.cold_start_mask.sum().item())
    cold_pct = 100 * cold / max(data.num_nodes, 1)
    logger.info(f"  Cold-start nodes:   {cold} ({cold_pct:.1f}%)")
    logger.info(f"{'='*60}")

    logger.info("NEXT STEPS:")
    logger.info("  1. Upload data/processed/unified_graph.pt to Google Drive")
    logger.info("  2. Run notebooks/colab_training.ipynb on Colab T4")
    logger.info("  3. Download best_model.pt to checkpoints/")
    logger.info("  4. Run: uvicorn app.main:app --reload")

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Cross-lingual KG Completion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_pipeline.py --phase 1              # Sample mode (default, fast)
  python scripts/run_pipeline.py --phase 1 --full       # Full extraction
  python scripts/run_pipeline.py --phase 1 --force-refresh  # Bypass cache
        """,
    )
    parser.add_argument(
        "--phase",
        type=str,
        required=True,
        choices=["1", "2", "all"],
        help="Pipeline phase to run (1=extract, 2=align+graph, all=both)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        default=False,
        help="Run full extraction (all triples). Default is sample mode (200/query).",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        default=False,
        help="Bypass SQLite cache and re-query Wikidata.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml in project root).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log level (default: INFO).",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for the pipeline runner."""
    args = parse_args()

    # Initialize logging first — everything after this uses the logger
    setup_logging(log_level=args.log_level)

    logger.info(f"Pipeline runner started | phase={args.phase} | full={args.full}")

    # Load configuration
    config = load_config(args.config)

    # Dispatch to the appropriate phase
    if args.phase in ("1", "all"):
        processed_triples = run_phase_1(
            config,
            sample_mode=not args.full,
            force_refresh=args.force_refresh,
        )
    else:
        processed_triples = None

    if args.phase in ("2", "all"):
        run_phase_2(config, processed_triples)


if __name__ == "__main__":
    main()
