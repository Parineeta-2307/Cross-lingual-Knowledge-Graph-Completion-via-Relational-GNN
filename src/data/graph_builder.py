"""Unified graph construction from processed triples and anchor pairs.

Converts per-language triples + cross-lingual anchor pairs into a single
PyTorch Geometric Data object for training the R-GCN model.

Architecture decisions:
    - All languages share ONE graph (not separate graphs per language).
      Cross-lingual SAME_AS edges connect aligned entities, enabling
      the R-GCN to propagate information across languages.
    - Cold-start nodes (degree < threshold) are flagged for special
      handling during inference (Gap #5).
    - Edge-level splits (not node-level) preserve the graph structure
      in all splits — every node appears in training.
    - Relation type stratification ensures each split has a representative
      distribution of all relation types.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
from loguru import logger

try:
    from torch_geometric.data import Data
except ImportError:
    logger.error(
        "torch-geometric not installed. Follow the installation steps in "
        "requirements.txt (Steps 2-4 must be done in order)."
    )
    raise


# Language code → integer mapping (used for node_language tensor)
LANG_TO_ID: dict[str, int] = {"de": 0, "ja": 1, "nl": 2, "en": 3}


class GraphBuilder:
    """Converts processed triples + anchor pairs into a PyTorch Geometric graph.

    Builds a unified directed multigraph where:
      - Nodes = entities (from all languages)
      - Edges = knowledge graph relations + SAME_AS alignment edges
      - Node attributes: language, script, cold-start flag
      - Edge attributes: relation type (integer-encoded)

    Args:
        config: Full configuration dictionary from config.yaml.

    Example:
        >>> from src.utils.config_loader import load_config
        >>> config = load_config()
        >>> builder = GraphBuilder(config)
        >>> G = builder.build_networkx_graph(processed_triples, anchor_pairs)
        >>> data, e2id, r2id = builder.networkx_to_pyg(G)
        >>> data.num_nodes
        847
    """

    def __init__(self, config: dict) -> None:
        graph_cfg = config["graph"]
        self.cold_start_threshold: int = graph_cfg["cold_start_degree_threshold"]
        self.train_ratio: float = graph_cfg["train_ratio"]
        self.val_ratio: float = graph_cfg["val_ratio"]
        self.test_ratio: float = graph_cfg["test_ratio"]
        self.processed_dir: Path = Path(config["data"]["processed_dir"])
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"GraphBuilder initialized | cold_start_threshold={self.cold_start_threshold} "
            f"| split_ratios={self.train_ratio}/{self.val_ratio}/{self.test_ratio}"
        )

    def build_networkx_graph(
        self,
        processed_triples: dict[str, list[dict[str, str]]],
        anchor_pairs: list[dict[str, Any]],
    ) -> nx.DiGraph:
        """Build a unified directed graph from triples and anchor pairs.

        Combines all languages into a single graph. Each triple becomes a
        directed edge with the relation as an edge attribute. Anchor pairs
        become bidirectional SAME_AS edges.

        Node attributes:
          - language: str (de/ja/nl/en or 'cross' for multi-language nodes)
          - script: str (latin/japanese/mixed/other)
          - cold_start: bool (degree < threshold after construction)

        Args:
            processed_triples: Dict mapping language codes to lists of
                processed triple dicts (with head, tail, relation, language,
                head_script, tail_script fields).
            anchor_pairs: List of confirmed anchor pair dicts from alignment.

        Returns:
            NetworkX DiGraph with all nodes, edges, and attributes set.
        """
        G = nx.DiGraph()

        # --- Add triple edges ---
        edge_count = 0
        for lang, triples in processed_triples.items():
            for triple in triples:
                head = triple["head"]
                tail = triple["tail"]
                relation = triple["relation"]

                # Add nodes with language and script attributes
                if head not in G:
                    G.add_node(
                        head,
                        language=lang,
                        script=triple.get("head_script", "unknown"),
                    )
                if tail not in G:
                    G.add_node(
                        tail,
                        language=lang,
                        script=triple.get("tail_script", "unknown"),
                    )

                # Add directed edge with relation type
                G.add_edge(head, tail, relation=relation)
                edge_count += 1

        logger.info(
            f"Triple edges added | edges={edge_count} | "
            f"nodes={G.number_of_nodes()}"
        )

        # --- Add SAME_AS edges for anchor pairs (bidirectional) ---
        same_as_count = 0
        for pair in anchor_pairs:
            entity_a = pair["entity_a"]
            entity_b = pair["entity_b"]

            # Ensure both nodes exist (they should, from triples)
            if entity_a not in G:
                G.add_node(
                    entity_a,
                    language=pair["lang_a"],
                    script="unknown",
                )
            if entity_b not in G:
                G.add_node(
                    entity_b,
                    language=pair["lang_b"],
                    script="unknown",
                )

            # Bidirectional SAME_AS
            G.add_edge(entity_a, entity_b, relation="SAME_AS")
            G.add_edge(entity_b, entity_a, relation="SAME_AS")
            same_as_count += 2

        logger.info(f"SAME_AS edges added | edges={same_as_count}")

        # --- Flag cold-start nodes ---
        cold_start_count = 0
        for node in G.nodes():
            degree = G.degree(node)
            is_cold = degree < self.cold_start_threshold
            G.nodes[node]["cold_start"] = is_cold
            if is_cold:
                cold_start_count += 1

        # --- Log statistics ---
        total_nodes = G.number_of_nodes()
        total_edges = G.number_of_edges()
        cold_pct = 100 * cold_start_count / max(total_nodes, 1)

        # Count edges by language
        edges_by_lang: Counter = Counter()
        for _, _, data in G.edges(data=True):
            edges_by_lang[data.get("relation", "unknown")] += 1

        logger.info(
            f"Graph built | nodes={total_nodes} | edges={total_edges} | "
            f"cold_start={cold_start_count} ({cold_pct:.1f}%)"
        )
        logger.info(f"Edges by relation: {dict(edges_by_lang)}")

        # Count nodes by language
        nodes_by_lang: Counter = Counter()
        for _, attrs in G.nodes(data=True):
            nodes_by_lang[attrs.get("language", "unknown")] += 1
        logger.info(f"Nodes by language: {dict(nodes_by_lang)}")

        return G

    def networkx_to_pyg(
        self, G: nx.DiGraph
    ) -> tuple[Data, dict[str, int], dict[str, int]]:
        """Convert a NetworkX graph to a PyTorch Geometric Data object.

        Builds integer mappings for entities and relations, then constructs
        the sparse edge_index tensor and associated metadata tensors.

        Args:
            G: NetworkX DiGraph from build_networkx_graph().

        Returns:
            Tuple of:
              - data: PyG Data object with edge_index, edge_type,
                      node_language, cold_start_mask
              - entity_to_id: dict mapping entity names to integer IDs
              - relation_to_id: dict mapping relation names to integer IDs
        """
        # Build entity-to-id mapping (sorted for determinism)
        all_nodes = sorted(G.nodes())
        entity_to_id: dict[str, int] = {
            node: idx for idx, node in enumerate(all_nodes)
        }

        # Build relation-to-id mapping
        all_relations = sorted(
            {data["relation"] for _, _, data in G.edges(data=True)}
        )
        relation_to_id: dict[str, int] = {
            rel: idx for idx, rel in enumerate(all_relations)
        }

        # Build edge_index and edge_type tensors
        src_ids: list[int] = []
        dst_ids: list[int] = []
        edge_types: list[int] = []

        for src, dst, data in G.edges(data=True):
            src_ids.append(entity_to_id[src])
            dst_ids.append(entity_to_id[dst])
            edge_types.append(relation_to_id[data["relation"]])

        edge_index = torch.tensor([src_ids, dst_ids], dtype=torch.long)
        edge_type = torch.tensor(edge_types, dtype=torch.long)

        # Build node attribute tensors
        node_languages: list[int] = []
        cold_start_flags: list[bool] = []

        for node in all_nodes:
            attrs = G.nodes[node]
            lang = attrs.get("language", "en")
            node_languages.append(LANG_TO_ID.get(lang, 3))
            cold_start_flags.append(attrs.get("cold_start", False))

        node_language = torch.tensor(node_languages, dtype=torch.long)
        cold_start_mask = torch.tensor(cold_start_flags, dtype=torch.bool)

        # Create PyG Data object
        data = Data(
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=len(all_nodes),
            node_language=node_language,
            cold_start_mask=cold_start_mask,
        )

        logger.info(
            f"PyG Data created | nodes={data.num_nodes} | "
            f"edges={data.edge_index.shape[1]} | "
            f"relations={len(relation_to_id)} | "
            f"cold_start={cold_start_mask.sum().item()}"
        )

        return data, entity_to_id, relation_to_id

    def split_edges(
        self, data: Data
    ) -> tuple[Data, Data, Data]:
        """Split edges into train/val/test sets, stratified by relation type.

        Uses edge-level splitting (NOT node-level) so that all nodes appear
        in the training graph. Stratification ensures each split has a
        representative distribution of all relation types.

        Args:
            data: Full PyG Data object from networkx_to_pyg().

        Returns:
            Tuple of (train_data, val_data, test_data), each a PyG Data
            object with the same num_nodes but different edge subsets.
        """
        num_edges = data.edge_index.shape[1]
        edge_types_np = data.edge_type.numpy()

        # Stratified split: for each relation type, split its edges
        train_indices: list[int] = []
        val_indices: list[int] = []
        test_indices: list[int] = []

        unique_types = np.unique(edge_types_np)

        for rel_type in unique_types:
            # Get indices of edges with this relation type
            type_mask = edge_types_np == rel_type
            type_indices = np.where(type_mask)[0]

            # Shuffle
            rng = np.random.RandomState(42)
            rng.shuffle(type_indices)

            n = len(type_indices)
            n_train = max(1, int(n * self.train_ratio))
            n_val = max(1, int(n * self.val_ratio))
            # Remaining goes to test (handles rounding)

            train_indices.extend(type_indices[:n_train].tolist())
            val_indices.extend(type_indices[n_train:n_train + n_val].tolist())
            test_indices.extend(type_indices[n_train + n_val:].tolist())

        def _make_split(indices: list[int]) -> Data:
            """Create a Data object with a subset of edges."""
            idx = torch.tensor(indices, dtype=torch.long)
            return Data(
                edge_index=data.edge_index[:, idx],
                edge_type=data.edge_type[idx],
                num_nodes=data.num_nodes,
                node_language=data.node_language,
                cold_start_mask=data.cold_start_mask,
            )

        train_data = _make_split(train_indices)
        val_data = _make_split(val_indices)
        test_data = _make_split(test_indices)

        logger.info(
            f"Edge split | train={len(train_indices)} "
            f"({100 * len(train_indices) / max(num_edges, 1):.1f}%) | "
            f"val={len(val_indices)} "
            f"({100 * len(val_indices) / max(num_edges, 1):.1f}%) | "
            f"test={len(test_indices)} "
            f"({100 * len(test_indices) / max(num_edges, 1):.1f}%)"
        )

        return train_data, val_data, test_data

    def save_artifacts(
        self,
        data: Data,
        train_data: Data,
        val_data: Data,
        test_data: Data,
        entity_to_id: dict[str, int],
        relation_to_id: dict[str, int],
    ) -> None:
        """Save all graph artifacts to data/processed/.

        Saved files:
          - unified_graph.pt: full graph (PyG Data)
          - train_graph.pt, val_graph.pt, test_graph.pt: split graphs
          - entity_to_id.json: entity name → integer ID mapping
          - relation_to_id.json: relation name → integer ID mapping
          - graph_stats.json: summary statistics for the dashboard

        Args:
            data: Full PyG Data object.
            train_data: Training split.
            val_data: Validation split.
            test_data: Test split.
            entity_to_id: Entity name → ID mapping.
            relation_to_id: Relation name → ID mapping.
        """
        out = self.processed_dir

        # Save PyG graphs
        torch.save(data, out / "unified_graph.pt")
        torch.save(train_data, out / "train_graph.pt")
        torch.save(val_data, out / "val_graph.pt")
        torch.save(test_data, out / "test_graph.pt")
        logger.info(f"Graphs saved | dir={out}")

        # Save mappings
        with open(out / "entity_to_id.json", "w", encoding="utf-8") as f:
            json.dump(entity_to_id, f, ensure_ascii=False, indent=2)

        with open(out / "relation_to_id.json", "w", encoding="utf-8") as f:
            json.dump(relation_to_id, f, ensure_ascii=False, indent=2)

        logger.info(
            f"Mappings saved | entities={len(entity_to_id)} "
            f"| relations={len(relation_to_id)}"
        )

        # Compute and save statistics
        id_to_relation = {v: k for k, v in relation_to_id.items()}

        # Count entities by language
        entities_by_language: dict[str, int] = {"de": 0, "ja": 0, "nl": 0, "en": 0}
        id_to_lang = {0: "de", 1: "ja", 2: "nl", 3: "en"}
        for lang_id in data.node_language.tolist():
            lang = id_to_lang.get(lang_id, "unknown")
            entities_by_language[lang] = entities_by_language.get(lang, 0) + 1

        # Count triples by relation
        triples_by_relation: dict[str, int] = {}
        for rel_id in data.edge_type.tolist():
            rel_name = id_to_relation.get(rel_id, f"rel_{rel_id}")
            triples_by_relation[rel_name] = (
                triples_by_relation.get(rel_name, 0) + 1
            )

        cold_start_count = int(data.cold_start_mask.sum().item())
        stats: dict[str, Any] = {
            "num_entities": data.num_nodes,
            "num_relations": len(relation_to_id),
            "num_triples": data.edge_index.shape[1],
            "cold_start_entities": cold_start_count,
            "cold_start_pct": round(
                100 * cold_start_count / max(data.num_nodes, 1), 1
            ),
            "entities_by_language": entities_by_language,
            "triples_by_relation": triples_by_relation,
            "split_sizes": {
                "train": train_data.edge_index.shape[1],
                "val": val_data.edge_index.shape[1],
                "test": test_data.edge_index.shape[1],
            },
        }

        with open(out / "graph_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info(
            f"Graph stats saved | entities={stats['num_entities']} | "
            f"triples={stats['num_triples']} | "
            f"cold_start={stats['cold_start_pct']}%"
        )
        logger.info(f"Entities by language: {entities_by_language}")
        logger.info(f"Triples by relation: {triples_by_relation}")
