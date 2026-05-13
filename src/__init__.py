"""Cross-lingual Knowledge Graph Completion via Relational GNN.

This package contains the full pipeline:
  - data/: SPARQL extraction, preprocessing, entity alignment, graph construction
  - models/: R-GCN architecture, RotatE scoring, negative sampling
  - training/: Training loop, evaluation with filtered ranking
  - inference/: Local prediction engine with cold-start fallback
  - utils/: Logging, Unicode handling, configuration utilities
"""

__version__ = "0.1.0"
