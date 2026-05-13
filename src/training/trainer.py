"""Training loop for Cross-lingual Knowledge Graph Completion.

This module orchestrates the training of the R-GCN and RotatE scorer,
incorporating negative sampling, margin ranking loss, and early stopping
based on filtered validation MRR.
"""

import copy
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm

from src.models.negative_sampler import NegativeSampler
from src.training.evaluator import KGEvaluator

try:
    from torch_geometric.data import Data
except ImportError:
    pass  # Handled elsewhere


class Trainer:
    """Orchestrates the training of the KG completion model.

    Args:
        model: The RGCN model instance.
        scorer: The RotatEScorer instance.
        sampler: The NegativeSampler instance.
        evaluator: The KGEvaluator instance.
        config: Full configuration dictionary.
    """

    def __init__(
        self,
        model: nn.Module,
        scorer: nn.Module,
        sampler: NegativeSampler,
        evaluator: KGEvaluator,
        config: dict,
    ) -> None:
        self.model = model
        self.scorer = scorer
        self.sampler = sampler
        self.evaluator = evaluator
        self.config = config

        train_cfg = config["training"]
        self.epochs: int = train_cfg["epochs"]
        self.batch_size: int = train_cfg["batch_size"]
        self.lr: float = train_cfg["learning_rate"]
        self.weight_decay: float = train_cfg["weight_decay"]
        self.margin: float = train_cfg["margin"]
        self.patience: int = train_cfg["early_stopping_patience"]
        
        # Save directory
        self.checkpoint_dir = Path(train_cfg["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.checkpoint_dir / train_cfg["checkpoint_name"]

        # Optional: Google Drive sync (for Colab)
        self.gdrive_path = None
        if train_cfg.get("gdrive_checkpoint_path"):
            self.gdrive_path = Path(train_cfg["gdrive_checkpoint_path"])

        # Select device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.scorer = self.scorer.to(self.device)

        # Initialize optimizer and scheduler
        # We optimize both the RGCN parameters and the RotatE phase embeddings
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.scorer.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
        # Reduce LR when validation MRR stops improving
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )

        logger.info(
            f"Trainer initialized | device={self.device} | epochs={self.epochs} "
            f"| batch_size={self.batch_size} | lr={self.lr} | margin={self.margin}"
        )

    def train(self, train_data: "Data", val_data: "Data") -> dict[str, list[float]]:
        """Execute the full training loop with early stopping.

        Args:
            train_data: PyG Data object containing training edges.
            val_data: PyG Data object containing validation edges.

        Returns:
            Dictionary containing training history metrics (loss, val_mrr).
        """
        # Convert edge_index back to (head, relation, tail) triples for batching
        # train_data.edge_index is shape (2, E)
        train_h = train_data.edge_index[0]
        train_t = train_data.edge_index[1]
        train_r = train_data.edge_type
        train_triples = torch.stack([train_h, train_r, train_t], dim=1).to(self.device)
        
        val_h = val_data.edge_index[0]
        val_t = val_data.edge_index[1]
        val_r = val_data.edge_type
        val_triples = torch.stack([val_h, val_r, val_t], dim=1).to(self.device)

        # Full graph edge index needs to be on device for RGCN forward pass
        full_edge_index = train_data.edge_index.to(self.device)
        full_edge_type = train_data.edge_type.to(self.device)

        num_train_triples = train_triples.shape[0]
        num_batches = (num_train_triples + self.batch_size - 1) // self.batch_size

        history: dict[str, list[float]] = {"loss": [], "val_mrr": []}
        
        best_val_mrr = 0.0
        epochs_without_improvement = 0
        best_model_state = None
        best_scorer_state = None

        logger.info(f"Starting training loop... | batches_per_epoch={num_batches}")

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            self.scorer.train()
            
            total_loss = 0.0
            
            # Shuffle training triples at the start of each epoch
            indices = torch.randperm(num_train_triples, device=self.device)
            shuffled_triples = train_triples[indices]

            epoch_iterator = tqdm(
                range(0, num_train_triples, self.batch_size), 
                desc=f"Epoch {epoch:03d}/{self.epochs}", 
                leave=False
            )

            for i in epoch_iterator:
                # 1. Get positive batch
                pos_batch = shuffled_triples[i:i + self.batch_size]
                
                # 2. Sample negative triples (on CPU, then move to device)
                neg_batch = self.sampler.sample(pos_batch.cpu()).to(self.device)

                # 3. Forward pass: compute entity embeddings using the FULL training graph
                # (RGCN aggregates neighborhood information)
                all_embeddings = self.model(full_edge_index, full_edge_type)

                # 4. Extract embeddings for the positive and negative batches
                # Positive
                pos_h_emb = all_embeddings[pos_batch[:, 0]]
                pos_r_idx = pos_batch[:, 1]
                pos_t_emb = all_embeddings[pos_batch[:, 2]]
                
                # Negative
                neg_h_emb = all_embeddings[neg_batch[:, 0]]
                neg_r_idx = neg_batch[:, 1]
                neg_t_emb = all_embeddings[neg_batch[:, 2]]

                # 5. Score triples using RotatE
                pos_scores = self.scorer.score(pos_h_emb, pos_r_idx, pos_t_emb)
                neg_scores = self.scorer.score(neg_h_emb, neg_r_idx, neg_t_emb)

                # 6. Compute loss and optimize
                loss = self.scorer.loss(pos_scores, neg_scores, margin=self.margin)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                epoch_iterator.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = total_loss / num_batches
            history["loss"].append(avg_loss)

            # Evaluate every 10 epochs
            if epoch % 10 == 0 or epoch == self.epochs:
                val_metrics = self.evaluator.evaluate(
                    self.model, 
                    self.scorer, 
                    val_triples, 
                    full_edge_index, 
                    full_edge_type
                )
                
                val_mrr = val_metrics.get("mrr_filtered", 0.0)
                history["val_mrr"].append(val_mrr)
                
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(
                    f"Epoch {epoch:03d}/{self.epochs} | Loss: {avg_loss:.4f} | "
                    f"Val MRR (filt): {val_mrr:.4f} | Hits@10: {val_metrics.get('hits_at_10', 0):.4f} | "
                    f"LR: {current_lr}"
                )

                self.scheduler.step(val_mrr)

                # Save best model
                if val_mrr > best_val_mrr:
                    best_val_mrr = val_mrr
                    epochs_without_improvement = 0
                    
                    # Store best weights in memory
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    best_scorer_state = copy.deepcopy(self.scorer.state_dict())
                    
                    # Save to disk
                    self.save_checkpoint(epoch, val_mrr)
                else:
                    epochs_without_improvement += 10
                    
                # Early stopping
                if epochs_without_improvement >= self.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch} (patience={self.patience})")
                    break

        # Training complete, restore best weights
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.scorer.load_state_dict(best_scorer_state)
            logger.info(f"Training complete. Restored best model with Val MRR: {best_val_mrr:.4f}")

        return history

    def save_checkpoint(self, epoch: int, val_mrr: float) -> None:
        """Save model checkpoint to disk and optionally to Google Drive.
        
        The checkpoint contains everything needed for inference.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "scorer_state": self.scorer.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "val_mrr": val_mrr,
            "config": self.config,
        }

        # Save locally
        torch.save(checkpoint, self.checkpoint_path)
        logger.debug(f"Checkpoint saved | path={self.checkpoint_path} | val_mrr={val_mrr:.4f}")

        # Save to Google Drive if configured (useful for Colab)
        if self.gdrive_path is not None:
            try:
                # Ensure the parent directory exists on GDrive
                self.gdrive_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint, self.gdrive_path)
                logger.debug(f"Checkpoint synced to Google Drive | path={self.gdrive_path}")
            except Exception as e:
                logger.warning(f"Failed to sync checkpoint to Google Drive: {e}")
