"""Frequency-weighted negative sampling for knowledge graph training.

This module implements Gap #2: replacing uniform random negative sampling
with frequency-weighted corrupt-tail sampling.

Why frequency-weighted:
    Uniform negative sampling disproportionately selects rare entities
    (there are many of them). The model wastes capacity learning to
    distinguish common entities from extremely rare ones it will never
    see at test time. Frequency-weighted sampling (with smoothing) focuses
    negatives on entities the model is likely to confuse — producing
    harder, more informative negative examples.

    The smoothing exponent (default 0.75) comes from Word2Vec's negative
    sampling trick: sample by count^0.75. This dampens hub-node dominance
    while still preferring common entities over rare ones.

Reference:
    Mikolov et al. (2013) — "Distributed Representations of Words and
    Phrases and their Compositionality" (negative sampling section)
"""

import torch
from loguru import logger


class NegativeSampler:
    """Frequency-weighted corrupt-tail negative sampler.

    For each positive triple (h, r, t), generates k negative triples by
    replacing the tail with randomly sampled entities. Sampling probability
    is proportional to entity frequency ^ smoothing.

    Known-true triples are filtered out (up to max_resample attempts)
    to avoid training on false negatives.

    Args:
        num_entities: Total entity count in the knowledge graph.
        true_triples: All known true (h, r, t) triples, shape (N, 3).
            Used to avoid sampling known-true triples as negatives.
        num_negatives: Number of negative samples per positive triple.
        smoothing: Frequency smoothing exponent. Lower values flatten the
            distribution; 1.0 = proportional to raw frequency.

    Example:
        >>> sampler = NegativeSampler(
        ...     num_entities=847,
        ...     true_triples=train_triples,
        ...     num_negatives=64
        ... )
        >>> neg = sampler.sample(batch)  # batch shape (B, 3)
        >>> neg.shape
        torch.Size([B*64, 3])
    """

    def __init__(
        self,
        num_entities: int,
        true_triples: torch.Tensor,
        num_negatives: int = 64,
        smoothing: float = 0.75,
    ) -> None:
        self.num_entities: int = num_entities
        self.num_negatives: int = num_negatives
        self.smoothing: float = smoothing

        # Build true triple set for fast lookup during filtering
        self.true_triple_set: set[tuple[int, int, int]] = set()
        for row in true_triples:
            h, r, t = row[0].item(), row[1].item(), row[2].item()
            self.true_triple_set.add((h, r, t))

        # Build frequency distribution over tail entities
        # Count how often each entity appears as a tail
        tail_counts = torch.zeros(num_entities, dtype=torch.float32)
        if true_triples.shape[0] > 0:
            tails = true_triples[:, 2]
            for t in tails:
                tail_counts[t.item()] += 1.0

        # Apply smoothing and ensure minimum count of 1 for every entity
        # (so entities with zero tail occurrences still have nonzero probability)
        tail_counts = tail_counts + 1.0
        smoothed = tail_counts.pow(smoothing)

        # Normalize to probability distribution
        self.sampling_weights: torch.Tensor = smoothed / smoothed.sum()

        logger.info(
            f"NegativeSampler initialized | entities={num_entities} "
            f"| true_triples={len(self.true_triple_set)} "
            f"| k={num_negatives} | smoothing={smoothing}"
        )

    def sample(self, batch: torch.Tensor) -> torch.Tensor:
        """Generate negative triples by corrupting tails.

        For each positive triple in the batch, samples num_negatives
        replacement tail entities. Avoids sampling tails that would
        create known-true triples (with up to 3 re-sample attempts).

        Args:
            batch: Positive triples of shape (batch_size, 3).
                Columns: (head, relation, tail).

        Returns:
            Negative triples of shape (batch_size * num_negatives, 3).
            Each row is (head, relation, corrupted_tail).
        """
        batch_size = batch.shape[0]
        heads = batch[:, 0]
        relations = batch[:, 1]

        # Sample candidate tail replacements
        # Shape: (batch_size, num_negatives)
        neg_tails = torch.multinomial(
            self.sampling_weights.expand(batch_size, -1),
            num_samples=self.num_negatives,
            replacement=True,
        )

        # Filter out known true triples (up to 3 re-sample attempts)
        max_resample = 3
        for attempt in range(max_resample):
            needs_resample = torch.zeros_like(neg_tails, dtype=torch.bool)

            for i in range(batch_size):
                h = heads[i].item()
                r = relations[i].item()
                for j in range(self.num_negatives):
                    t = neg_tails[i, j].item()
                    if (h, r, t) in self.true_triple_set:
                        needs_resample[i, j] = True

            resample_count = needs_resample.sum().item()
            if resample_count == 0:
                break

            # Re-sample only the positions that hit true triples
            if resample_count > 0:
                new_samples = torch.multinomial(
                    self.sampling_weights,
                    num_samples=int(resample_count),
                    replacement=True,
                )
                neg_tails[needs_resample] = new_samples

        # Build negative triple tensor
        # Expand heads and relations to match negative count
        neg_heads = heads.unsqueeze(1).expand(-1, self.num_negatives).reshape(-1)
        neg_relations = relations.unsqueeze(1).expand(-1, self.num_negatives).reshape(-1)
        neg_tails_flat = neg_tails.reshape(-1)

        negatives = torch.stack([neg_heads, neg_relations, neg_tails_flat], dim=1)
        return negatives

    def get_sampling_stats(self) -> dict:
        """Compute statistics about the sampling distribution.

        Useful for the "What Didn't Work" analysis in the README and for
        verifying that frequency-weighted sampling isn't degenerate.

        Returns:
            Dict with sampling distribution statistics:
              - mean_weight: average sampling probability
              - top_5_entity_ids: 5 most likely entities to be sampled
              - top_5_weights: their sampling probabilities
              - hub_selection_rate: fraction of total probability mass
                held by hub nodes (weight > mean + 2*std)
        """
        weights = self.sampling_weights

        mean_w = weights.mean().item()
        std_w = weights.std().item()

        # Top-5 most sampled entities
        top_k_values, top_k_indices = weights.topk(min(5, len(weights)))

        # Hub node analysis: entities with weight > mean + 2*std
        hub_threshold = mean_w + 2 * std_w
        hub_mask = weights > hub_threshold
        hub_rate = weights[hub_mask].sum().item()

        return {
            "mean_weight": round(mean_w, 6),
            "std_weight": round(std_w, 6),
            "top_5_entity_ids": top_k_indices.tolist(),
            "top_5_weights": [round(v, 6) for v in top_k_values.tolist()],
            "hub_node_count": int(hub_mask.sum().item()),
            "hub_selection_rate": round(hub_rate, 4),
        }
