"""
SimCSE-style Contrastive Learning for Seq2Seq Text Detoxification

Adds an auxiliary NT-Xent loss on mean-pooled encoder representations.
For each input, the encoder is run twice (different dropout masks) to
create two views that form a positive pair. Other batch samples serve
as negatives.

Reference: Gao et al., "SimCSE: Simple Contrastive Learning of Sentence Embeddings" (EMNLP 2021)

Usage:
    # 启用对比学习
    proj_head = ProjectionHead(hidden_dim=768, proj_dim=256)
    model.add_module("proj_head", proj_head)   # 注册到 model 上，优化器自动包含

    trainer = ContrastiveSeq2SeqTrainer(
        proj_head=proj_head,
        contrastive_weight=0.1,
        contrastive_temperature=0.05,
        model=model, args=training_args,
        train_dataset=train_dataset, ...
    )

    # 禁用对比学习（proj_head=None 或 weight=0，等同于标准 Seq2SeqTrainer）
    trainer = ContrastiveSeq2SeqTrainer(
        contrastive_weight=0.0,
        model=model, args=training_args,
        train_dataset=train_dataset, ...
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Seq2SeqTrainer


# ── Projection Head ────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """
    2-layer MLP: Linear → GELU → Dropout → Linear
    Maps mean-pooled encoder output (768-d) to contrastive space (256-d).
    No activation after final linear (SimCSE convention).
    """

    def __init__(self, hidden_dim: int = 768, proj_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.linear2 = nn.Linear(hidden_dim, proj_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


# ── NT-Xent Loss ───────────────────────────────────────────────────

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    """
    Normalized Temperature-scaled Cross Entropy loss.

    Positive pairs: (z1[i], z2[i]) — two views of the same input.
    Negative pairs: all other (2*B - 2) samples in the concatenated set.

    Args:
        z1: (B, D) first-view embeddings (should already be L2-normalized)
        z2: (B, D) second-view embeddings
        temperature: scaling factor (default 0.05)

    Returns:
        scalar loss averaged over 2*B rows
    """
    B = z1.size(0)

    # Concat: [z1; z2] → (2B, D), normalize
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)

    # Cosine similarity scaled by temperature
    sim = torch.mm(z, z.t()) / temperature  # (2B, 2B)

    # Labels: row i → its positive is at i+B (for i<B) or i-B (for i>=B)
    labels = torch.arange(B, device=z1.device)
    labels = torch.cat([labels + B, labels])  # (2B,)

    # Mask self-comparisons
    sim = sim.masked_fill(torch.eye(2 * B, device=z1.device, dtype=torch.bool), float("-inf"))

    return F.cross_entropy(sim, labels)


# ── Mean Pooling ───────────────────────────────────────────────────

def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average-pool encoder hidden states over real (non-padding) tokens."""
    mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


# ── Trainer ────────────────────────────────────────────────────────

class ContrastiveSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Seq2SeqTrainer with optional SimCSE contrastive loss.

    When proj_head is None or contrastive_weight == 0, behaves identically
    to a standard Seq2SeqTrainer (no overhead).

    Loss = CE + weight * NT-Xent(proj_head(mean_pool(enc_view1)),
                                  proj_head(mean_pool(enc_view2)))
    """

    def __init__(
        self,
        proj_head: nn.Module = None,
        contrastive_weight: float = 0.0,
        contrastive_temperature: float = 0.05,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.proj_head = proj_head
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature

    @property
    def _use_contrastive(self) -> bool:
        """Only compute contrastive loss during training, with a valid head and weight."""
        return (
            self.model.training
            and self.proj_head is not None
            and self.contrastive_weight > 0
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # ── Main forward: CE loss + encoder hidden states (view 1) ──
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs.get("labels"),
            return_dict=True,
        )
        ce_loss = outputs.loss

        if not self._use_contrastive:
            return (ce_loss, outputs) if return_outputs else ce_loss

        # ── View 1: from the main forward pass ──
        h1 = outputs.encoder_last_hidden_state  # (B, L, 768)

        # ── View 2: second encoder pass (different dropout masks) ──
        h2 = model.get_encoder()(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
        ).last_hidden_state  # (B, L, 768)

        # ── Mean pool + project ──
        z1 = self.proj_head(mean_pool(h1, inputs["attention_mask"]))
        z2 = self.proj_head(mean_pool(h2, inputs["attention_mask"]))

        # ── Contrastive loss ──
        cl_loss = nt_xent_loss(z1, z2, temperature=self.contrastive_temperature)
        total_loss = ce_loss + self.contrastive_weight * cl_loss

        return (total_loss, outputs) if return_outputs else total_loss
