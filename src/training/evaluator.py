"""Knowledge Graph Evaluator with Filtered Ranking.

This module evaluates link prediction performance using the standard
Hits@k and Mean Reciprocal Rank (MRR) metrics. Crucially, it implements
"filtered" ranking (Gap #6 from architecture).

Why filtered ranking:
    If a query "SAP -> acquired -> ?" returns "Sybase" as rank 1 and
    "Concur" as rank 2, and the test triple is (SAP, acquired, Concur),
    a naive (raw) evaluation would penalize the model and say it ranked
    the true answer 2nd. However, SAP DID acquire Sybase. "Sybase" is
    a known true answer from the training set.
    
    Filtered evaluation removes ALL known true tails (except the target
    test tail) from the candidate ranking. This ensures the model isn't
    penalized for predicting other correct facts.

Reference:
    Bordes et al. (2013) — "Translating Embeddings for Modeling
    Multi-relational Data" (TransE paper, introduced filtered metrics)
"""

import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm


class KGEvaluator:
    """Evaluates KG link prediction performance using filtered ranking.

    Args:
        all_true_triples: Tensor of shape (N, 3) containing ALL known
            true triples across train, validation, and test sets.
        num_entities: Total number of entities in the graph.
        
    Example:
        >>> evaluator = KGEvaluator(all_triples, num_entities=847)
        >>> metrics = evaluator.evaluate(model, scorer, test_triples, ...)
        >>> print(metrics["mrr_filtered"])
        0.312
    """

    def __init__(self, all_true_triples: torch.Tensor, num_entities: int) -> None:
        self.num_entities = num_entities
        
        # Build filter index: maps (head, relation) -> set of valid tails
        # and (tail, relation) -> set of valid heads
        self.tail_filter_index: dict[tuple[int, int], set[int]] = {}
        self.head_filter_index: dict[tuple[int, int], set[int]] = {}
        
        logger.info(f"Building filter index from {all_true_triples.shape[0]} triples...")
        
        for row in all_true_triples:
            h, r, t = row[0].item(), row[1].item(), row[2].item()
            
            # Add to tail filter index
            hr = (h, r)
            if hr not in self.tail_filter_index:
                self.tail_filter_index[hr] = set()
            self.tail_filter_index[hr].add(t)
            
            # Add to head filter index
            tr = (t, r)
            if tr not in self.head_filter_index:
                self.head_filter_index[tr] = set()
            self.head_filter_index[tr].add(h)
            
        logger.info(
            f"Filter index built | unique (h,r) pairs: {len(self.tail_filter_index)} | "
            f"unique (t,r) pairs: {len(self.head_filter_index)}"
        )

    def _filtered_rank(
        self, query_id: int, relation: int, target_id: int, all_scores: torch.Tensor, is_tail_prediction: bool = True
    ) -> int:
        """Compute the rank of the target entity in a filtered candidate list.

        Args:
            query_id: The ID of the known entity (head if predicting tail, tail if predicting head).
            relation: The relation ID.
            target_id: The true target entity ID to find the rank for.
            all_scores: Tensor of shape (num_entities,) with scores for all candidate entities.
                Higher score means more likely.
            is_tail_prediction: If True, query_id is head and we're predicting tail.
                If False, query_id is tail and we're predicting head.

        Returns:
            Integer rank (1-indexed) of the target_id among all valid candidates.
        """
        # Get the set of all known true answers for this query
        if is_tail_prediction:
            filter_set = self.tail_filter_index.get((query_id, relation), set())
        else:
            filter_set = self.head_filter_index.get((query_id, relation), set())
            
        # We want to remove all known true answers EXCEPT the current target
        entities_to_mask = filter_set - {target_id}
        
        if entities_to_mask:
            # Create a mask tensor
            mask_indices = torch.tensor(list(entities_to_mask), dtype=torch.long, device=all_scores.device)
            # Set scores of true-but-not-target entities to negative infinity
            # so they drop to the bottom of the ranking
            all_scores[mask_indices] = -float('inf')
            
        # Get rank of target_id
        # We count how many entities have a STRICTLY HIGHER score than the target
        target_score = all_scores[target_id].item()
        
        # Count entities with higher scores (rank 1 means 0 entities are higher)
        higher_scores = (all_scores > target_score).sum().item()
        
        # Tie-breaking: half of the ties are considered ranked higher on average.
        # This prevents the model from predicting the exact same score for everything
        # and getting a good rank.
        ties = (all_scores == target_score).sum().item() - 1
        
        # Final rank (1-indexed)
        rank = higher_scores + (ties / 2) + 1
        return int(rank)

    def evaluate(
        self,
        model: torch.nn.Module,
        scorer: torch.nn.Module,
        test_triples: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        batch_size: int = 256,
    ) -> dict[str, float]:
        """Run full evaluation on a test set.

        Computes both head-prediction (predict h given r, t) and 
        tail-prediction (predict t given h, r) for every test triple.
        Reports Hits@1, 3, 10 and MRR for both raw and filtered settings.

        Args:
            model: Trained RGCN model.
            scorer: Trained RotatEScorer.
            test_triples: Tensor of shape (N, 3) to evaluate on.
            edge_index: Full graph edge index (needed for model forward pass).
            edge_type: Full graph edge types.
            batch_size: Number of queries to process simultaneously.

        Returns:
            Dictionary of metrics (mrr_filtered, hits_at_1, etc.).
        """
        model.eval()
        scorer.eval()
        
        num_triples = test_triples.shape[0]
        if num_triples == 0:
            return {}

        # Pre-compute all entity embeddings once for the entire graph
        with torch.no_grad():
            all_embeddings = model(edge_index, edge_type)
            
        # Create a tensor of all entity IDs [0, 1, 2, ..., num_entities-1]
        all_entity_ids = torch.arange(self.num_entities, device=all_embeddings.device)
        
        # Metrics trackers
        ranks_filtered_tail = []
        ranks_raw_tail = []
        ranks_filtered_head = []
        ranks_raw_head = []

        logger.info(f"Evaluating {num_triples} test triples...")
        
        with torch.no_grad():
            # Process triples one by one (or could batch, but one-by-one is easier for filtering)
            for i in tqdm(range(num_triples), desc="Evaluating", leave=False):
                h = test_triples[i, 0].item()
                r = test_triples[i, 1].item()
                t = test_triples[i, 2].item()
                
                # --- 1. Tail Prediction: (?, r, t) given h ---
                # Score all entities as candidates for the tail
                h_emb = all_embeddings[h].unsqueeze(0).expand(self.num_entities, -1)
                r_tensor = torch.tensor([r], device=all_embeddings.device).expand(self.num_entities)
                
                tail_scores = scorer.score(h_emb, r_tensor, all_embeddings)
                
                # Compute Raw Rank
                t_score = tail_scores[t].item()
                higher = (tail_scores > t_score).sum().item()
                ties = (tail_scores == t_score).sum().item() - 1
                raw_rank_tail = higher + (ties / 2) + 1
                ranks_raw_tail.append(raw_rank_tail)
                
                # Compute Filtered Rank
                # We pass a clone of the scores because _filtered_rank modifies it inplace
                filt_rank_tail = self._filtered_rank(
                    query_id=h, relation=r, target_id=t, 
                    all_scores=tail_scores.clone(), is_tail_prediction=True
                )
                ranks_filtered_tail.append(filt_rank_tail)
                
                # Sanity check: filtered rank must be <= raw rank (lower rank is better)
                assert filt_rank_tail <= raw_rank_tail, f"Filtered rank {filt_rank_tail} > Raw {raw_rank_tail}"
                
                # --- 2. Head Prediction: (h, r, ?) given t ---
                # Score all entities as candidates for the head
                t_emb = all_embeddings[t].unsqueeze(0).expand(self.num_entities, -1)
                
                head_scores = scorer.score(all_embeddings, r_tensor, t_emb)
                
                # Compute Raw Rank
                h_score = head_scores[h].item()
                higher = (head_scores > h_score).sum().item()
                ties = (head_scores == h_score).sum().item() - 1
                raw_rank_head = higher + (ties / 2) + 1
                ranks_raw_head.append(raw_rank_head)
                
                # Compute Filtered Rank
                filt_rank_head = self._filtered_rank(
                    query_id=t, relation=r, target_id=h, 
                    all_scores=head_scores.clone(), is_tail_prediction=False
                )
                ranks_filtered_head.append(filt_rank_head)

        # Aggregate metrics
        # We combine head and tail predictions for the overall metrics
        all_ranks_filtered = ranks_filtered_tail + ranks_filtered_head
        all_ranks_raw = ranks_raw_tail + ranks_raw_head
        
        # Convert to tensors for easy metric calculation
        ranks_filt_t = torch.tensor(all_ranks_filtered, dtype=torch.float)
        ranks_raw_t = torch.tensor(all_ranks_raw, dtype=torch.float)
        
        mrr_filt = (1.0 / ranks_filt_t).mean().item()
        mrr_raw = (1.0 / ranks_raw_t).mean().item()
        
        mrr_head = (1.0 / torch.tensor(ranks_filtered_head, dtype=torch.float)).mean().item()
        mrr_tail = (1.0 / torch.tensor(ranks_filtered_tail, dtype=torch.float)).mean().item()
        
        hits_1 = (ranks_filt_t <= 1.0).float().mean().item()
        hits_3 = (ranks_filt_t <= 3.0).float().mean().item()
        hits_10 = (ranks_filt_t <= 10.0).float().mean().item()
        
        return {
            "mrr_filtered": round(mrr_filt, 4),
            "mrr_raw": round(mrr_raw, 4),
            "hits_at_1": round(hits_1, 4),
            "hits_at_3": round(hits_3, 4),
            "hits_at_10": round(hits_10, 4),
            "mrr_filtered_head": round(mrr_head, 4),
            "mrr_filtered_tail": round(mrr_tail, 4),
            "num_test_triples": num_triples,
            "filter_benefit": round(mrr_filt - mrr_raw, 4),
        }

    def evaluate_by_relation(
        self,
        model: torch.nn.Module,
        scorer: torch.nn.Module,
        test_triples: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        id_to_relation: dict[int, str],
    ) -> pd.DataFrame:
        """Evaluate metrics broken down by relation type.

        Returns:
            A pandas DataFrame with metrics for each relation type.
        """
        results = []
        
        # Group test triples by relation
        unique_relations = torch.unique(test_triples[:, 1]).tolist()
        
        for r_id in unique_relations:
            # Extract triples for this relation
            mask = test_triples[:, 1] == r_id
            rel_triples = test_triples[mask]
            
            # Evaluate just this subset
            rel_metrics = self.evaluate(
                model, scorer, rel_triples, edge_index, edge_type
            )
            
            rel_name = id_to_relation.get(r_id, f"rel_{r_id}")
            
            results.append({
                "relation": rel_name,
                "hits@1": rel_metrics["hits_at_1"],
                "hits@10": rel_metrics["hits_at_10"],
                "mrr_filtered": rel_metrics["mrr_filtered"],
                "count": rel_metrics["num_test_triples"]
            })
            
        df = pd.DataFrame(results)
        return df.sort_values(by="mrr_filtered", ascending=False).reset_index(drop=True)

    def evaluate_cold_start(
        self,
        model: torch.nn.Module,
        scorer: torch.nn.Module,
        test_triples: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        cold_start_mask: torch.Tensor,
    ) -> dict[str, dict[str, float]]:
        """Evaluate cold-start vs non-cold-start performance separately.
        
        A test triple is considered "cold start" if either the head OR
        the tail node is flagged in the cold_start_mask.
        """
        heads = test_triples[:, 0]
        tails = test_triples[:, 2]
        
        # Boolean mask: true if head or tail is cold-start
        is_cold_start = cold_start_mask[heads] | cold_start_mask[tails]
        
        cold_triples = test_triples[is_cold_start]
        warm_triples = test_triples[~is_cold_start]
        
        logger.info(
            f"Cold-start split | cold: {cold_triples.shape[0]} triples | "
            f"warm: {warm_triples.shape[0]} triples"
        )
        
        cold_metrics = self.evaluate(model, scorer, cold_triples, edge_index, edge_type)
        warm_metrics = self.evaluate(model, scorer, warm_triples, edge_index, edge_type)
        
        return {
            "cold_start": cold_metrics,
            "warm_start": warm_metrics
        }
