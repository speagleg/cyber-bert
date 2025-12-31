#!/usr/bin/env python3
"""
SecBERT + Advanced KG-GNN Integrated Model.

Combines:
1. SecBERT text encoder (security-domain pretrained)
2. Advanced KG-GNN with:
   - R-GCN (edge-type aware message passing)
   - HGT (heterogeneous graph transformer)
   - TransE pre-training
   - Dynamic text-conditioned attention
3. Cross-attention fusion
4. Multi-label classification head
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

from advanced_kg_gnn import AdvancedKnowledgeGraphGNN


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention to fuse text embeddings with technique embeddings.

    Text queries attend to technique keys/values from the KG.
    """

    def __init__(
        self,
        text_dim: int,
        kg_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.d_k = hidden_dim // num_heads

        # Project text to query
        self.text_to_q = nn.Linear(text_dim, hidden_dim)

        # Project KG to key/value
        self.kg_to_k = nn.Linear(kg_dim, hidden_dim)
        self.kg_to_v = nn.Linear(kg_dim, hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        text_emb: torch.Tensor,      # [batch, text_dim]
        technique_emb: torch.Tensor,  # [batch, num_techniques, kg_dim] or [num_techniques, kg_dim]
    ) -> torch.Tensor:
        """
        Cross-attention from text to techniques.

        Returns:
            Fused representation [batch, hidden_dim]
        """
        batch_size = text_emb.size(0)

        # Handle non-batched technique embeddings
        if technique_emb.dim() == 2:
            technique_emb = technique_emb.unsqueeze(0).expand(batch_size, -1, -1)

        num_techniques = technique_emb.size(1)

        # Compute Q, K, V
        Q = self.text_to_q(text_emb)  # [B, H*D]
        K = self.kg_to_k(technique_emb)  # [B, T, H*D]
        V = self.kg_to_v(technique_emb)  # [B, T, H*D]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, 1, self.num_heads, self.d_k).transpose(1, 2)  # [B, H, 1, D]
        K = K.view(batch_size, num_techniques, self.num_heads, self.d_k).transpose(1, 2)  # [B, H, T, D]
        V = V.view(batch_size, num_techniques, self.num_heads, self.d_k).transpose(1, 2)  # [B, H, T, D]

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # [B, H, 1, T]
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Aggregate
        out = torch.matmul(attn, V)  # [B, H, 1, D]
        out = out.transpose(1, 2).contiguous().view(batch_size, -1)  # [B, H*D]

        # Project and residual
        out = self.out_proj(out)

        # Note: No residual here since dimensions may differ
        return self.layer_norm(out)


class SecBERTAdvancedGNNModel(nn.Module):
    """
    SecBERT + Advanced KG-GNN for ATT&CK technique classification.

    Architecture:
    1. SecBERT encodes security log text -> 768-dim
    2. Project text to shared dimension -> 512-dim
    3. Advanced GNN encodes ATT&CK KG with text conditioning -> 512-dim per technique
    4. Cross-attention fuses text with technique embeddings
    5. Classification head outputs technique probabilities
    """

    def __init__(
        self,
        kg_dir: str,
        bert_model: str = 'jackaduma/SecBERT',
        hidden_dim: int = 512,
        gnn_layers: int = 4,
        gnn_heads: int = 8,
        dropout: float = 0.1,
        use_transe_pretrain: bool = True,
        transe_epochs: int = 100,
    ):
        super().__init__()

        self.kg_dir = Path(kg_dir)
        self.hidden_dim = hidden_dim

        # Load BERT (SecBERT)
        print(f"Loading text encoder: {bert_model}")
        self.bert_config = AutoConfig.from_pretrained(bert_model)
        self.bert = AutoModel.from_pretrained(bert_model)
        self.bert_dim = self.bert_config.hidden_size  # Usually 768

        # Text projection to shared dimension
        self.text_projection = nn.Sequential(
            nn.Linear(self.bert_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Advanced KG-GNN
        print(f"Loading Advanced KG-GNN from {kg_dir}")
        self.gnn = AdvancedKnowledgeGraphGNN(
            kg_dir=kg_dir,
            hidden_dim=hidden_dim,
            num_layers=gnn_layers,
            num_heads=gnn_heads,
            dropout=dropout,
            use_transe_pretrain=use_transe_pretrain,
            transe_epochs=transe_epochs,
        )

        # Cross-attention fusion
        self.cross_attention = CrossAttentionFusion(
            text_dim=hidden_dim,
            kg_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=gnn_heads,
            dropout=dropout,
        )

        # Classification head
        self.num_techniques = self.gnn.num_techniques
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Concat text + cross-attn
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_techniques),
        )

        # Technique list for label mapping
        self.technique_list = self.gnn.technique_list

        # Cache for GNN embeddings (recomputed when text changes)
        self._cached_node_emb = None
        self._cached_device = None

    def pretrain_gnn(self, device: torch.device):
        """Pre-train GNN with TransE embeddings."""
        self.gnn.pretrain_with_transe(device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_text_conditioning: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            use_text_conditioning: Whether to use dynamic text-conditioned GNN

        Returns:
            Logits for each technique [batch, num_techniques]
        """
        device = input_ids.device
        batch_size = input_ids.size(0)

        # 1. BERT encoding
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = bert_output.last_hidden_state[:, 0, :]  # CLS token [B, 768]

        # 2. Project to shared dimension
        text_proj = self.text_projection(text_emb)  # [B, hidden_dim]

        # 3. GNN encoding
        if use_text_conditioning:
            # Dynamic: GNN attention conditioned on text
            node_emb = self.gnn(device, text_emb=text_proj)  # [B, N, hidden]
            technique_emb = self.gnn.get_technique_embeddings(node_emb)  # [B, T, hidden]
        else:
            # Static: Same GNN output for all inputs (cached)
            if self._cached_node_emb is None or self._cached_device != device:
                self._cached_node_emb = self.gnn(device)  # [N, hidden]
                self._cached_device = device
            technique_emb = self.gnn.get_technique_embeddings(self._cached_node_emb)  # [T, hidden]

        # 4. Cross-attention fusion
        cross_attn_out = self.cross_attention(text_proj, technique_emb)  # [B, hidden]

        # 5. Classification
        combined = torch.cat([text_proj, cross_attn_out], dim=-1)  # [B, hidden*2]
        logits = self.classifier(combined)  # [B, num_techniques]

        return logits

    def get_technique_attention(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get technique attention weights for interpretability.

        Returns:
            (logits, attention_weights)
        """
        device = input_ids.device

        # BERT encoding
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = bert_output.last_hidden_state[:, 0, :]
        text_proj = self.text_projection(text_emb)

        # GNN with text conditioning
        node_emb = self.gnn(device, text_emb=text_proj)
        technique_emb = self.gnn.get_technique_embeddings(node_emb)

        # Cross-attention with attention weights
        batch_size = text_proj.size(0)
        num_techniques = technique_emb.size(1)

        Q = self.cross_attention.text_to_q(text_proj)
        K = self.cross_attention.kg_to_k(technique_emb)

        d_k = self.cross_attention.d_k
        num_heads = self.cross_attention.num_heads

        Q = Q.view(batch_size, 1, num_heads, d_k).transpose(1, 2)
        K = K.view(batch_size, num_techniques, num_heads, d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        attn_weights = F.softmax(scores, dim=-1).mean(dim=1).squeeze(1)  # Average over heads [B, T]

        # Full forward for logits
        logits = self.forward(input_ids, attention_mask)

        return logits, attn_weights


def create_advanced_model(
    kg_dir: str,
    bert_model: str = 'jackaduma/SecBERT',
    hidden_dim: int = 512,
    gnn_layers: int = 4,
    gnn_heads: int = 8,
    dropout: float = 0.1,
    use_transe_pretrain: bool = True,
    transe_epochs: int = 100,
    device: torch.device = None,
) -> SecBERTAdvancedGNNModel:
    """
    Factory function to create and initialize the model.
    """
    model = SecBERTAdvancedGNNModel(
        kg_dir=kg_dir,
        bert_model=bert_model,
        hidden_dim=hidden_dim,
        gnn_layers=gnn_layers,
        gnn_heads=gnn_heads,
        dropout=dropout,
        use_transe_pretrain=use_transe_pretrain,
        transe_epochs=transe_epochs,
    )

    if device is not None:
        model = model.to(device)
        if use_transe_pretrain:
            model.pretrain_gnn(device)

    return model


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument('--kg_dir', type=str, default='../data/attack_framework')
    parser.add_argument('--bert_model', type=str, default='jackaduma/SecBERT')
    args = parser.parse_args()

    print("Testing SecBERT + Advanced KG-GNN Model...")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.bert_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    # Create model
    print(f"\nCreating model...")
    model = create_advanced_model(
        kg_dir=args.kg_dir,
        bert_model=args.bert_model,
        hidden_dim=512,
        gnn_layers=4,
        gnn_heads=8,
        use_transe_pretrain=True,
        transe_epochs=50,  # Reduced for testing
        device=device,
    )

    print(f"\nModel Statistics:")
    print(f"  - Techniques: {model.num_techniques}")
    print(f"  - BERT dim: {model.bert_dim}")
    print(f"  - Hidden dim: {model.hidden_dim}")

    # Test forward pass
    print("\n" + "-" * 40)
    print("Testing forward pass...")

    test_texts = [
        "powershell.exe -enc SQBFAFgAIAAoAE4AZQB3AC0ATwBiAGoAZQBjAHQA",
        "Process Create: svchost.exe spawned cmd.exe /c whoami",
        "Network connection to 192.168.1.100:443 from malware.exe",
        "Registry key HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Run modified",
    ]

    # Tokenize
    encoding = tokenizer(
        test_texts,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(input_ids, attention_mask)

    print(f"  - Input shape: {input_ids.shape}")
    print(f"  - Output shape: {logits.shape}")

    # Get predictions
    probs = torch.sigmoid(logits)
    top_k = 5
    for i, text in enumerate(test_texts):
        print(f"\n  Sample {i+1}: {text[:60]}...")
        top_probs, top_indices = torch.topk(probs[i], top_k)
        for j in range(top_k):
            tech_id = model.technique_list[top_indices[j]]
            prob = top_probs[j].item()
            print(f"    {tech_id}: {prob:.3f}")

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameters:")
    print(f"  - Total: {total_params:,}")
    print(f"  - Trainable: {trainable_params:,}")

    print("\n" + "=" * 70)
    print("SecBERT + Advanced KG-GNN ready!")
