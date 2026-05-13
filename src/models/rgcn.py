"""Relational Graph Convolutional Network (R-GCN) architecture.

This module implements the core graph neural network used for learning
entity representations. Standard GCNs only work on simple graphs with
one edge type. R-GCNs are designed for knowledge graphs with multiple
relation types (e.g., 'acquired', 'headquartered_in').

Why basis decomposition:
    With many relation types and high hidden dimensions, learning a full
    weight matrix per relation causes massive parameter bloat and overfitting.
    Basis decomposition regularizes this by learning a small set of 'basis'
    matrices; each relation's specific weight matrix is just a linear
    combination of these shared bases.

Reference:
    Schlichtkrull et al. (2018) — "Modeling Relational Data with Graph
    Convolutional Networks" (https://arxiv.org/abs/1703.06103)
"""

import torch
import torch.nn as nn
from loguru import logger

try:
    from torch_geometric.nn import RGCNConv
except ImportError:
    logger.error(
        "torch-geometric not installed. Follow the installation steps in "
        "requirements.txt (Steps 2-4 must be done in order)."
    )
    raise


class RGCN(nn.Module):
    """2-layer Relational Graph Convolutional Network.

    Architecture:
      Input: Entity indices → Embedding(num_entities, embedding_dim)
      Layer 1: RGCNConv(embedding_dim, hidden_dim) → ReLU → Dropout
      Layer 2: RGCNConv(hidden_dim, embedding_dim)
      Output: Updated entity embeddings of shape (num_entities, embedding_dim)

    Args:
        num_entities: Total number of unique entities in the graph.
        num_relations: Total number of unique relation types.
        config: Full configuration dictionary from config.yaml.

    Example:
        >>> model = RGCN(num_entities=847, num_relations=8, config=cfg)
        >>> embeddings = model(data.edge_index, data.edge_type)
        >>> embeddings.shape
        torch.Size([847, 128])
    """

    def __init__(
        self, num_entities: int, num_relations: int, config: dict
    ) -> None:
        super().__init__()
        
        model_cfg = config["model"]
        self.embedding_dim: int = model_cfg["embedding_dim"]
        self.hidden_dim: int = model_cfg["hidden_dim"]
        self.num_bases: int = model_cfg["num_bases"]
        self.dropout: float = model_cfg["dropout"]
        
        self.num_entities = num_entities
        self.num_relations = num_relations

        # Initial node embeddings (learned parameters)
        self.node_embedding = nn.Embedding(num_entities, self.embedding_dim)
        
        # Initialize embeddings with Xavier normal (standard for KG embeddings)
        nn.init.xavier_normal_(self.node_embedding.weight)

        # Ensure num_bases isn't larger than num_relations
        # If relations are few, we just use num_relations as the base count
        actual_bases = min(self.num_bases, num_relations)

        # Layer 1: Transforms from embedding_dim to hidden_dim
        self.conv1 = RGCNConv(
            in_channels=self.embedding_dim,
            out_channels=self.hidden_dim,
            num_relations=num_relations,
            num_bases=actual_bases,
        )

        # Layer 2: Transforms from hidden_dim back to embedding_dim
        self.conv2 = RGCNConv(
            in_channels=self.hidden_dim,
            out_channels=self.embedding_dim,
            num_relations=num_relations,
            num_bases=actual_bases,
        )

        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(p=self.dropout)

        logger.info(
            f"RGCN initialized | entities={num_entities} | relations={num_relations} "
            f"| dims={self.embedding_dim}→{self.hidden_dim}→{self.embedding_dim} "
            f"| bases={actual_bases}"
        )

    def forward(
        self, edge_index: torch.Tensor, edge_type: torch.Tensor
    ) -> torch.Tensor:
        """Full forward pass to compute updated embeddings for all entities.

        In a full-batch graph setting (like ours), we pass the entire graph
        structure through the GCN layers at once.

        Args:
            edge_index: Tensor of shape (2, num_edges) containing source
                and destination node indices.
            edge_type: Tensor of shape (num_edges,) containing the relation
                type integer for each edge.

        Returns:
            Updated entity embeddings tensor of shape (num_entities, embedding_dim).
        """
        # Start with the learned initial embeddings for all nodes
        x = self.node_embedding.weight

        # Layer 1
        x = self.conv1(x, edge_index, edge_type)
        x = self.relu(x)
        x = self.dropout_layer(x)

        # Layer 2
        x = self.conv2(x, edge_index, edge_type)

        return x

    def get_entity_embeddings(
        self,
        entity_ids: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """Compute embeddings only for specific entities.

        During inference, we don't always need the entire graph's embeddings.
        This computes the full graph pass once, then slices out the requested
        entities. (In a production setting with millions of nodes, this would
        use neighbor sampling instead of a full pass).

        Args:
            entity_ids: Tensor of entity indices to extract.
            edge_index: Full graph edge index.
            edge_type: Full graph edge types.

        Returns:
            Embeddings for the requested entities, shape (len(entity_ids), dim).
        """
        all_embeddings = self.forward(edge_index, edge_type)
        return all_embeddings[entity_ids]
