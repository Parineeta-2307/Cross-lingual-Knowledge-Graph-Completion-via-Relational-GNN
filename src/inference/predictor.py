"""Inference engine for predicting missing links in the knowledge graph.

This module loads the trained R-GCN and RotatE models, along with the
pre-computed unified graph and dictionaries. It serves the FastAPI
backend by answering prediction queries ("SAP", "acquired", "?").

Gap #5 (Cold Start):
    For entities with very few edges (degree < threshold), the RGCN
    embeddings are poor. This module falls back to pure sentence-transformer
    similarity scoring for cold-start entities, ensuring the system fails
    gracefully rather than confidently predicting nonsense.
"""

import json
from pathlib import Path

import torch
from loguru import logger

try:
    from thefuzz import process as fuzz_process
except ImportError:
    logger.error("thefuzz not installed. Run: pip install thefuzz==0.22.1")
    raise

# Conditional import for fallback model
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

from src.models.link_predictor import RotatEScorer
from src.models.rgcn import RGCN
from src.utils.text_utils import normalize_text


class KGPredictor:
    """Loads a trained KG model and serves predictions.

    Pre-computes all entity embeddings at initialization so that
    inference requests are extremely fast (just one RotatE scoring step
    per request, no RGCN forward pass needed).

    Args:
        checkpoint_path: Path to the trained best_model.pt file.
        config: Full configuration dictionary.

    Example:
        >>> predictor = KGPredictor("checkpoints/best_model.pt", config)
        >>> results = predictor.predict("SAP", "acquired", top_k=3)
        >>> print(results[0]["entity"])
        "Sybase"
    """

    def __init__(self, checkpoint_path: str, config: dict) -> None:
        chk_path = Path(checkpoint_path)
        if not chk_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {chk_path.resolve()}.\n"
                f"Run the Colab training notebook first, then download "
                f"best_model.pt to the checkpoints/ directory."
            )

        self.config = config
        self.processed_dir = Path(config["data"]["processed_dir"])
        self.top_k = config["inference"]["top_k_predictions"]
        self.fuzzy_threshold = config["inference"]["fuzzy_match_threshold"]
        self.use_fallback = config["inference"]["cold_start_fallback"]

        # Select device (CPU is usually fine for inference, especially since
        # we precompute embeddings)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading prediction engine | checkpoint={chk_path} | device={self.device}")

        # 1. Load data mappings and stats
        with open(self.processed_dir / "entity_to_id.json", "r", encoding="utf-8") as f:
            self.entity_to_id: dict[str, int] = json.load(f)
            
        with open(self.processed_dir / "relation_to_id.json", "r", encoding="utf-8") as f:
            self.relation_to_id: dict[str, int] = json.load(f)
            
        with open(self.processed_dir / "graph_stats.json", "r", encoding="utf-8") as f:
            self.graph_stats = json.load(f)

        self.id_to_entity = {v: k for k, v in self.entity_to_id.items()}
        self.num_entities = len(self.entity_to_id)
        self.num_relations = len(self.relation_to_id)

        # 2. Load the unified graph (needed for cold-start mask and language info)
        unified_graph_path = self.processed_dir / "unified_graph.pt"
        if not unified_graph_path.exists():
            raise FileNotFoundError(f"Unified graph not found at {unified_graph_path}")
            
        graph_data = torch.load(unified_graph_path, map_location=self.device)
        self.cold_start_mask = graph_data.cold_start_mask.to(self.device)
        self.node_language = graph_data.node_language.to(self.device)

        # 3. Load model weights
        checkpoint = torch.load(chk_path, map_location=self.device)
        
        self.model = RGCN(self.num_entities, self.num_relations, config)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()

        self.scorer = RotatEScorer(self.num_relations, config["model"]["embedding_dim"])
        self.scorer.load_state_dict(checkpoint["scorer_state"])
        self.scorer.to(self.device)
        self.scorer.eval()

        # 4. Pre-compute entity embeddings (HUGE speedup for web app)
        logger.info("Pre-computing RGCN entity embeddings...")
        with torch.no_grad():
            self.all_embeddings = self.model(
                graph_data.edge_index.to(self.device), 
                graph_data.edge_type.to(self.device)
            )

        # 5. Load fallback alignment model if requested
        self.fallback_model = None
        if self.use_fallback:
            if HAS_SENTENCE_TRANSFORMERS:
                model_name = config["alignment"]["model_name"]
                logger.info(f"Loading cold-start fallback model: {model_name}")
                self.fallback_model = SentenceTransformer(model_name)
                
                # Pre-encode all entity names for fast fallback
                # (In a massive graph we'd lazy-load, but for <10k it's fast enough)
                logger.info("Pre-encoding entity labels for fallback mode...")
                all_names = [self.id_to_entity[i] for i in range(self.num_entities)]
                self.fallback_embeddings = self.fallback_model.encode(
                    all_names, convert_to_tensor=True, normalize_embeddings=True
                ).to(self.device)
            else:
                logger.warning(
                    "Cold-start fallback enabled in config, but sentence-transformers "
                    "not installed. Fallback will not be available."
                )

        # Build ID to Language string mapping
        self.id_to_lang_str = {0: "de", 1: "ja", 2: "nl", 3: "en"}

        logger.info(
            f"Predictor ready | entities={self.num_entities} "
            f"| trained_epoch={checkpoint['epoch']} "
            f"| val_mrr={checkpoint['val_mrr']:.4f}"
        )

    def _fuzzy_match_entity(self, query: str) -> tuple[str, int, float]:
        """Find the closest matching entity name in the vocabulary.

        Args:
            query: The user's input entity string.

        Returns:
            Tuple of (matched_entity_name, entity_id, match_score_0_to_100).

        Raises:
            ValueError: If no entity matches above the fuzzy_threshold.
        """
        # Exact match (normalized)
        norm_query = normalize_text(query)
        if norm_query in self.entity_to_id:
            return norm_query, self.entity_to_id[norm_query], 100.0

        # Fuzzy match
        # Extract the best match from the list of all valid entity names
        choices = list(self.entity_to_id.keys())
        best_match, score = fuzz_process.extractOne(norm_query, choices)

        if score < self.fuzzy_threshold:
            raise ValueError(
                f"Entity '{query}' not found. "
                f"Closest match: '{best_match}' ({score}% confidence - below threshold)."
            )

        return best_match, self.entity_to_id[best_match], float(score)

    def predict(
        self, head: str, relation: str, top_k: int = None
    ) -> list[dict]:
        """Predict the most likely tail entities for (head, relation, ?).

        Args:
            head: The head entity name (will be fuzzy matched).
            relation: The relation name (exact match required, case-insensitive).
            top_k: Number of predictions to return (defaults to config value).

        Returns:
            List of dictionary results containing rank, entity name, score,
            confidence percentage, language, and cold_start_fallback flag.
        """
        if top_k is None:
            top_k = self.top_k

        # 1. Look up relation
        relation_lower = relation.lower()
        if relation_lower not in self.relation_to_id:
            valid_rels = list(self.relation_to_id.keys())
            raise ValueError(
                f"Unknown relation '{relation}'. Valid options: {valid_rels}"
            )
        r_idx = self.relation_to_id[relation_lower]

        # 2. Look up head entity
        head_name, head_id, _ = self._fuzzy_match_entity(head)

        # 3. Check for cold-start (Gap #5)
        is_cold = self.cold_start_mask[head_id].item()
        
        # We need a 1D tensor for relation and 2D for head embedding
        r_tensor = torch.tensor([r_idx], device=self.device).expand(self.num_entities)
        
        if is_cold and self.use_fallback and self.fallback_model is not None:
            # --- COLD START FALLBACK ---
            logger.warning(f"Cold-start entity '{head_name}' — using embedding fallback")
            
            # Since the RGCN embedding is poor, we can't trust RotatE.
            # Instead, we just find entities that are semantically similar to the 
            # query entity, but we lack the relation transformation. 
            # In a real system, we'd have a separate fallback relation model. 
            # Here, we just use cosine similarity of the entity names as a baseline.
            head_fallback_emb = self.fallback_embeddings[head_id].unsqueeze(0)
            
            # Dot product for cosine similarity
            scores = (head_fallback_emb @ self.fallback_embeddings.T).squeeze(0)
            
            # Ensure the head itself doesn't show up in predictions
            scores[head_id] = -float('inf')
            
            used_fallback = True
        else:
            # --- NORMAL RGCN + RotatE PREDICTION ---
            head_emb = self.all_embeddings[head_id].unsqueeze(0).expand(self.num_entities, -1)
            
            with torch.no_grad():
                scores = self.scorer.score(head_emb, r_tensor, self.all_embeddings)
                
            used_fallback = False

        # 4. Extract top K
        top_scores, top_indices = torch.topk(scores, k=top_k)
        
        # Softmax over the top-k to get a readable "confidence" percentage
        # (RotatE scores are negative distances, so standard softmax works)
        if not used_fallback:
            probs = torch.softmax(top_scores, dim=0) * 100
        else:
            # Fallback uses cosine similarity (-1 to 1)
            # Scale up so softmax is peaky
            probs = torch.softmax(top_scores * 10.0, dim=0) * 100

        # 5. Format results
        results = []
        for i in range(top_k):
            idx = top_indices[i].item()
            score_val = top_scores[i].item()
            conf_val = probs[i].item()
            
            # Skip if score is -inf (can happen if graph is smaller than top_k)
            if score_val == -float('inf'):
                continue
                
            tail_name = self.id_to_entity[idx]
            lang_id = self.node_language[idx].item()
            lang_str = self.id_to_lang_str.get(lang_id, "en")
            
            results.append({
                "rank": i + 1,
                "entity": tail_name,
                "entity_id": idx,
                "language": lang_str,
                "score": round(score_val, 4),
                "confidence_pct": round(conf_val, 1),
                "cold_start_fallback": used_fallback
            })

        return results

    def get_neighborhood(self, entity_id_or_name: str | int, max_hops: int = 1) -> dict:
        """Get the immediate graph neighborhood for visualization.

        Retrieves all edges connected to the specified entity.
        Formatted specifically for Cytoscape.js in the frontend.

        Args:
            entity_id_or_name: The ID (int) or name (str) of the center entity.
            max_hops: Currently only 1-hop is supported.

        Returns:
            Dictionary with 'center', 'nodes', and 'edges' lists.
        """
        # Resolve entity ID
        if isinstance(entity_id_or_name, str):
            try:
                # First try exact parse if it's a string of digits
                if entity_id_or_name.isdigit():
                    center_id = int(entity_id_or_name)
                    center_name = self.id_to_entity[center_id]
                else:
                    center_name, center_id, _ = self._fuzzy_match_entity(entity_id_or_name)
            except (ValueError, KeyError):
                return {"error": f"Entity '{entity_id_or_name}' not found."}
        else:
            center_id = entity_id_or_name
            try:
                center_name = self.id_to_entity[center_id]
            except KeyError:
                return {"error": f"Entity ID {center_id} not found."}

        # Load edges from unified_graph.pt
        unified_graph_path = self.processed_dir / "unified_graph.pt"
        graph_data = torch.load(unified_graph_path, map_location="cpu")
        
        edge_index = graph_data.edge_index
        edge_types = graph_data.edge_type

        # Find all edges where center is head or tail
        head_mask = edge_index[0] == center_id
        tail_mask = edge_index[1] == center_id
        connected_edge_mask = head_mask | tail_mask
        
        connected_edges = edge_index[:, connected_edge_mask]
        connected_types = edge_types[connected_edge_mask]
        
        # Collect unique nodes in neighborhood
        neighbor_ids = torch.unique(connected_edges).tolist()
        
        nodes_list = []
        for nid in neighbor_ids:
            name = self.id_to_entity[nid]
            lang_id = self.node_language[nid].item()
            lang_str = self.id_to_lang_str.get(lang_id, "en")
            is_cold = self.cold_start_mask[nid].item()
            
            node_dict = {
                "data": {
                    "id": str(nid),
                    "label": name,
                    "language": lang_str,
                    "cold_start": is_cold,
                }
            }
            if nid == center_id:
                node_dict["classes"] = "center"
                
            nodes_list.append(node_dict)
            
        # Collect edges
        edges_list = []
        id_to_rel = {v: k for k, v in self.relation_to_id.items()}
        
        for i in range(connected_edges.shape[1]):
            src = connected_edges[0, i].item()
            dst = connected_edges[1, i].item()
            rel_type = connected_types[i].item()
            rel_name = id_to_rel.get(rel_type, f"rel_{rel_type}")
            
            edges_list.append({
                "data": {
                    "source": str(src),
                    "target": str(dst),
                    "relation": rel_name
                }
            })
            
        center_lang_id = self.node_language[center_id].item()

        return {
            "center": {
                "id": str(center_id),
                "label": center_name,
                "language": self.id_to_lang_str.get(center_lang_id, "en")
            },
            "elements": nodes_list + edges_list
        }

    def batch_predict(self, queries: list[dict[str, str]]) -> list[dict]:
        """Run predictions for multiple queries.

        Args:
            queries: List of dicts, e.g., [{"head": "SAP", "relation": "acquired"}, ...]

        Returns:
            List of prediction result lists.
        """
        results = []
        for q in queries:
            try:
                res = self.predict(q["head"], q["relation"])
                results.append({
                    "query": q,
                    "success": True,
                    "predictions": res
                })
            except Exception as e:
                results.append({
                    "query": q,
                    "success": False,
                    "error": str(e)
                })
        return results
