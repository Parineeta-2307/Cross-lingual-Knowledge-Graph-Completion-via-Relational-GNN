"""RotatE scoring function for knowledge graph link prediction.

This module implements the RotatE scorer, which models relations as
rotations in complex vector space.

Why RotatE:
    Traditional models like TransE (translation: h + r = t) cannot model
    symmetric relations (a = b → b = a) or inversion relations
    (a includes b → b is part of a). RotatE solves this by mapping entities
    to complex numbers and relations to complex rotations.
    
    Formula: score(h, r, t) = -|| h ∘ r - t ||
    where ∘ is element-wise complex multiplication, and |r_i| = 1.

Reference:
    Sun et al. (2019) — "RotatE: Knowledge Graph Embedding by
    Relational Rotation in Complex Space" (https://arxiv.org/abs/1902.10197)
"""

import torch
import torch.nn as nn
from loguru import logger


class RotatEScorer(nn.Module):
    """RotatE scorer for computing triple probabilities.

    Models entities as complex vectors and relations as phase rotations.
    The input entity embeddings (from the R-GCN) are treated as having
    concatenated real and imaginary parts. Therefore, the embedding_dim
    MUST be an even number.

    Args:
        num_relations: Total number of unique relation types.
        embedding_dim: Dimensionality of entity embeddings from R-GCN.
            Must be an even number.

    Example:
        >>> scorer = RotatEScorer(num_relations=8, embedding_dim=128)
        >>> # h_emb and t_emb shape: (batch_size, 128)
        >>> # r_idx shape: (batch_size,)
        >>> scores = scorer.score(h_emb, r_idx, t_emb)
        >>> scores.shape
        torch.Size([batch_size])
    """

    def __init__(self, num_relations: int, embedding_dim: int) -> None:
        super().__init__()

        if embedding_dim % 2 != 0:
            raise ValueError(
                f"RotatE requires an even embedding_dim (got {embedding_dim}). "
                f"The dimension is split equally into real and imaginary parts."
            )

        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        
        # Relations only need phase representations (angles).
        # We store them as a real number representing the phase angle
        # in the range [-pi, pi]. The complex rotation will be e^(i*phase).
        self.phase_dim = embedding_dim // 2
        
        # Relation embeddings are just the phase angles
        self.relation_phases = nn.Embedding(num_relations, self.phase_dim)
        
        # Initialize uniformly between -pi and pi
        nn.init.uniform_(self.relation_phases.weight, a=-3.14159, b=3.14159)

        logger.info(
            f"RotatEScorer initialized | relations={num_relations} "
            f"| dim={embedding_dim} (split into {self.phase_dim} real/imag)"
        )

    def _get_relation_rotation(self, relations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert relation phase angles into complex rotation components.
        
        Using Euler's formula: e^(i*θ) = cos(θ) + i*sin(θ)
        This guarantees that the relation modulus |r| = 1.
        
        Args:
            relations: Relation indices, shape (batch_size,).
            
        Returns:
            Tuple of (real_part, imaginary_part), each of shape (batch_size, phase_dim).
        """
        phases = self.relation_phases(relations)
        
        # r = cos(theta) + i*sin(theta)
        re_relation = torch.cos(phases)
        im_relation = torch.sin(phases)
        
        return re_relation, im_relation

    def score(
        self, heads: torch.Tensor, relations: torch.Tensor, tails: torch.Tensor
    ) -> torch.Tensor:
        """Compute the RotatE score for a batch of triples.

        Score = -distance(h ∘ r, t)
        Higher score (closer to 0) means the triple is more likely to be true.

        Args:
            heads: Head entity embeddings, shape (batch_size, embedding_dim).
            relations: Relation indices, shape (batch_size,).
            tails: Tail entity embeddings, shape (batch_size, embedding_dim).

        Returns:
            Scores tensor of shape (batch_size,).
        """
        # Split entity embeddings into real and imaginary halves
        re_head, im_head = torch.chunk(heads, 2, dim=-1)
        re_tail, im_tail = torch.chunk(tails, 2, dim=-1)

        # Get relation rotation components
        re_relation, im_relation = self._get_relation_rotation(relations)

        # Complex multiplication: h ∘ r
        # (re_h + i*im_h) * (re_r + i*im_r)
        # = (re_h * re_r - im_h * im_r) + i*(re_h * im_r + im_h * re_r)
        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation

        # Distance: || h ∘ r - t ||
        re_diff = re_score - re_tail
        im_diff = im_score - im_tail
        
        # Stack real and imaginary differences: shape (batch_size, phase_dim, 2)
        diff_stack = torch.stack([re_diff, im_diff], dim=-1)
        
        # Norm over the complex components, then sum over the phase_dim
        # Using L1 norm (sum of absolute differences) is common and stable in RotatE
        distance = torch.norm(diff_stack, p=2, dim=-1).sum(dim=-1)

        # Score is negative distance (so higher is better)
        return -distance

    def loss(
        self, pos_scores: torch.Tensor, neg_scores: torch.Tensor, margin: float = 1.0
    ) -> torch.Tensor:
        """Compute margin ranking loss for a batch.

        Loss = max(0, margin - positive_score + negative_score)
        
        If there are multiple negative samples per positive, neg_scores
        should be reshaped to (batch_size, num_negatives) before computing
        the mean over negatives.

        Args:
            pos_scores: Scores for true triples, shape (batch_size,).
            neg_scores: Scores for corrupted triples, shape (batch_size * k,).
            margin: The margin hyperparameter.

        Returns:
            Scalar loss tensor.
        """
        batch_size = pos_scores.shape[0]
        num_negatives = neg_scores.shape[0] // batch_size
        
        # Reshape negatives to align with positives: (batch_size, num_negatives)
        neg_scores = neg_scores.view(batch_size, num_negatives)
        
        # Expand pos_scores to match neg_scores shape: (batch_size, num_negatives)
        pos_scores = pos_scores.unsqueeze(1).expand_as(neg_scores)
        
        # Compute margin ranking loss for each positive-negative pair
        pair_loss = torch.relu(margin - pos_scores + neg_scores)
        
        # Mean over all pairs
        return pair_loss.mean()
