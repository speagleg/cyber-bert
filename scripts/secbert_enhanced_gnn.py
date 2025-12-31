#!/usr/bin/env python3
"""
SecBERT + Enhanced KG-GNN: Hybrid Model for MITRE ATT&CK Technique Classification.

This model combines:
1. SecBERT for encoding security log text
2. Enhanced KG-GNN with semantic features and hierarchical attention
3. Cross-modal attention between text and knowledge graph
4. Hierarchical loss for tactic → technique → sub-technique classification

Key improvements:
- Semantic node embeddings (not random) from ATT&CK descriptions
- Ontology-aware message passing respecting edge types
- Hierarchical attention propagating tactic→technique→sub-technique
- Multi-task learning with hierarchical auxiliary losses
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from enhanced_kg_gnn import EnhancedKnowledgeGraphGNN


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention between text embeddings and knowledge graph embeddings.

    Allows the model to dynamically attend to relevant parts of the KG
    based on the input text.
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

        # Project both modalities to same dimension
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        self.kg_projection = nn.Linear(kg_dim, hidden_dim)

        # Multi-head attention: text queries KG
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Gating to control information flow
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        text_emb: torch.Tensor,     # [batch, text_dim]
        kg_emb: torch.Tensor,       # [num_kg_nodes, kg_dim]
    ) -> torch.Tensor:
        """
        Cross-modal attention from text to knowledge graph.

        Args:
            text_emb: Text embeddings [batch_size, text_dim]
            kg_emb: KG node embeddings [num_nodes, kg_dim]

        Returns:
            Attended KG representation [batch_size, hidden_dim]
        """
        batch_size = text_emb.size(0)

        # Project to shared space
        text_h = self.text_projection(text_emb)  # [batch, hidden]
        kg_h = self.kg_projection(kg_emb)        # [num_nodes, hidden]

        # Expand text for attention
        text_h = text_h.unsqueeze(1)  # [batch, 1, hidden]

        # Expand KG for batch
        kg_h = kg_h.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, num_nodes, hidden]

        # Cross attention: text queries KG
        attended_kg, attention_weights = self.cross_attention(
            query=text_h,
            key=kg_h,
            value=kg_h,
        )
        attended_kg = attended_kg.squeeze(1)  # [batch, hidden]

        # Gated combination
        text_h = text_h.squeeze(1)  # [batch, hidden]
        gate = self.gate(torch.cat([text_h, attended_kg], dim=-1))
        output = self.norm(text_h + gate * attended_kg)

        return output, attention_weights


class HierarchicalClassifier(nn.Module):
    """
    Hierarchical classifier that predicts at multiple levels:
    1. Tactic level (coarse)
    2. Technique level (medium)
    3. Sub-technique level (fine)

    Uses auxiliary losses at each level for better gradient flow,
    similar to hierarchical classification in ICD-10 coding.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_tactics: int,
        num_techniques: int,
        num_subtechniques: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_tactics = num_tactics
        self.num_techniques = num_techniques
        self.num_subtechniques = num_subtechniques

        # Tactic classifier (coarsest level)
        self.tactic_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_tactics),
        )

        # Technique classifier (conditioned on tactic context)
        self.technique_classifier = nn.Sequential(
            nn.Linear(hidden_dim + num_tactics, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_techniques),
        )

        # Sub-technique classifier (conditioned on technique context)
        self.subtechnique_classifier = nn.Sequential(
            nn.Linear(hidden_dim + num_techniques, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_subtechniques),
        )

    def forward(
        self,
        x: torch.Tensor,  # [batch, hidden]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Hierarchical classification.

        Returns:
            Tuple of (tactic_logits, technique_logits, subtechnique_logits)
        """
        # Tactic prediction
        tactic_logits = self.tactic_classifier(x)  # [batch, num_tactics]
        tactic_probs = torch.sigmoid(tactic_logits)

        # Technique prediction (conditioned on tactic)
        technique_input = torch.cat([x, tactic_probs], dim=-1)
        technique_logits = self.technique_classifier(technique_input)  # [batch, num_techniques]
        technique_probs = torch.sigmoid(technique_logits)

        # Sub-technique prediction (conditioned on technique)
        subtechnique_input = torch.cat([x, technique_probs], dim=-1)
        subtechnique_logits = self.subtechnique_classifier(subtechnique_input)  # [batch, num_subtechniques]

        return tactic_logits, technique_logits, subtechnique_logits


class SecBERTEnhancedGNN(nn.Module):
    """
    SecBERT + Enhanced KG-GNN: State-of-the-art MITRE ATT&CK classifier.

    Architecture:
    1. SecBERT encodes security log text
    2. Enhanced KG-GNN encodes knowledge graph with semantic features
    3. Cross-modal attention fuses text and KG representations
    4. Hierarchical classifier predicts at tactic/technique/sub-technique levels

    This mirrors the successful ICD-10 approach:
    - Semantic understanding from descriptions (like morphemes/etymology)
    - Hierarchical structure (like ICD chapters→blocks→codes)
    - Relational reasoning (like etymological relationships)
    """

    def __init__(
        self,
        kg_dir: str,
        bert_model: str = "bert-base-uncased",
        hidden_dim: int = 512,
        gnn_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        freeze_bert_layers: int = 0,
        use_hierarchical_loss: bool = True,
        hierarchical_loss_weights: Tuple[float, float, float] = (0.1, 0.3, 0.6),
    ):
        super().__init__()

        self.kg_dir = Path(kg_dir)
        self.hidden_dim = hidden_dim
        self.use_hierarchical_loss = use_hierarchical_loss
        self.hierarchical_loss_weights = hierarchical_loss_weights

        # BERT encoder for text
        self.bert = AutoModel.from_pretrained(bert_model)
        self.bert_dim = self.bert.config.hidden_size

        # Optionally freeze early BERT layers
        if freeze_bert_layers > 0:
            for i, layer in enumerate(self.bert.encoder.layer):
                if i < freeze_bert_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

        # Text projection
        self.text_projection = nn.Sequential(
            nn.Linear(self.bert_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Enhanced KG-GNN (shares BERT with text encoder)
        self.kg_gnn = EnhancedKnowledgeGraphGNN(
            kg_dir=kg_dir,
            hidden_dim=hidden_dim,
            num_layers=gnn_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_semantic_features=True,
            bert_dim=self.bert_dim,  # Pass BERT hidden size, not the model itself
        )

        # Store KG info
        self.num_techniques = self.kg_gnn.num_techniques
        self.technique_list = self.kg_gnn.technique_list
        self.node_to_idx = self.kg_gnn.node_to_idx

        # Count hierarchical levels
        self.num_tactics = len(self.kg_gnn.tactic_indices)
        self.num_parent_techniques = len(self.kg_gnn.technique_indices)
        self.num_subtechniques = len(self.kg_gnn.subtechnique_indices)

        # Cross-modal attention
        self.cross_attention = CrossModalAttention(
            text_dim=hidden_dim,
            kg_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Main classifier (for all techniques)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, self.num_techniques),
        )

        # Hierarchical classifier (optional)
        if use_hierarchical_loss:
            self.hierarchical_classifier = HierarchicalClassifier(
                hidden_dim=hidden_dim,
                num_tactics=self.num_tactics,
                num_techniques=self.num_parent_techniques,
                num_subtechniques=self.num_subtechniques,
                dropout=dropout,
            )

        # Build technique to hierarchy mappings
        self._build_hierarchy_mappings()

    def _build_hierarchy_mappings(self):
        """Build mappings from techniques to their tactic/parent indices."""
        # Map technique IDs to tactic indices
        self.technique_to_tactic = {}
        self.technique_to_parent = {}

        for node_id, node in self.kg_gnn.nodes.items():
            if node['type'] in ['technique', 'subtechnique']:
                tactics = node.get('tactics', [])
                tactic_indices = []
                for tactic in tactics:
                    if tactic in self.kg_gnn.node_to_idx:
                        # Find index in tactic list
                        try:
                            idx = self.kg_gnn.tactic_indices.index(self.kg_gnn.node_to_idx[tactic])
                            tactic_indices.append(idx)
                        except ValueError:
                            pass
                self.technique_to_tactic[node_id] = tactic_indices

                # For sub-techniques, find parent technique
                if node['type'] == 'subtechnique' and '.' in node_id:
                    parent_id = node_id.split('.')[0]
                    if parent_id in self.kg_gnn.node_to_idx:
                        try:
                            parent_idx = self.kg_gnn.technique_indices.index(self.kg_gnn.node_to_idx[parent_id])
                            self.technique_to_parent[node_id] = parent_idx
                        except ValueError:
                            pass

    def precompute_kg_embeddings(self, device: torch.device, tokenizer=None):
        """Pre-compute semantic embeddings for KG nodes using shared BERT."""
        self.kg_gnn.precompute_semantic_embeddings(
            device=device,
            bert_model=self.bert,  # Share the BERT model
            tokenizer=tokenizer,
        )

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode text using BERT."""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.text_projection(cls_embedding)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        tactic_labels: Optional[torch.Tensor] = None,
        technique_labels: Optional[torch.Tensor] = None,
        subtechnique_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Multi-hot labels for all techniques [batch_size, num_techniques]
            tactic_labels: Multi-hot tactic labels [batch_size, num_tactics] (optional)
            technique_labels: Multi-hot technique labels [batch_size, num_parent_techniques] (optional)
            subtechnique_labels: Multi-hot subtechnique labels [batch_size, num_subtechniques] (optional)

        Returns:
            Dictionary with logits and optionally losses
        """
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Encode text
        text_emb = self.encode_text(input_ids, attention_mask)  # [batch, hidden]

        # Encode knowledge graph
        kg_node_emb = self.kg_gnn(device)  # [num_nodes, hidden]

        # Get technique-only embeddings for classification
        technique_kg_emb = self.kg_gnn.get_technique_embeddings(kg_node_emb)  # [num_techniques, hidden]

        # Cross-modal attention between text and KG
        attended_kg, attention_weights = self.cross_attention(text_emb, technique_kg_emb)

        # Combine text and attended KG
        combined = torch.cat([text_emb, attended_kg], dim=-1)  # [batch, hidden*2]

        # Main classification
        logits = self.classifier(combined)  # [batch, num_techniques]

        output = {
            'logits': logits,
            'attention_weights': attention_weights,
        }

        # Compute losses if labels provided
        if labels is not None:
            # Main BCE loss
            main_loss = F.binary_cross_entropy_with_logits(logits, labels)
            output['main_loss'] = main_loss

            # Hierarchical losses (optional)
            if self.use_hierarchical_loss:
                tactic_logits, technique_logits, subtechnique_logits = self.hierarchical_classifier(text_emb)

                output['tactic_logits'] = tactic_logits
                output['technique_logits'] = technique_logits
                output['subtechnique_logits'] = subtechnique_logits

                # Compute hierarchical labels from main labels if not provided
                if tactic_labels is None:
                    tactic_labels = self._compute_tactic_labels(labels, device)
                if technique_labels is None:
                    technique_labels = self._compute_technique_labels(labels, device)
                if subtechnique_labels is None:
                    subtechnique_labels = self._compute_subtechnique_labels(labels, device)

                tactic_loss = F.binary_cross_entropy_with_logits(tactic_logits, tactic_labels)
                technique_loss = F.binary_cross_entropy_with_logits(technique_logits, technique_labels)
                subtechnique_loss = F.binary_cross_entropy_with_logits(subtechnique_logits, subtechnique_labels)

                output['tactic_loss'] = tactic_loss
                output['technique_loss'] = technique_loss
                output['subtechnique_loss'] = subtechnique_loss

                # Combined hierarchical loss
                w_tactic, w_technique, w_subtechnique = self.hierarchical_loss_weights
                hierarchical_loss = (
                    w_tactic * tactic_loss +
                    w_technique * technique_loss +
                    w_subtechnique * subtechnique_loss
                )

                # Total loss: main + hierarchical
                output['loss'] = main_loss + 0.3 * hierarchical_loss
            else:
                output['loss'] = main_loss

        return output

    def _compute_tactic_labels(self, labels: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Compute tactic labels from technique labels."""
        batch_size = labels.size(0)
        tactic_labels = torch.zeros(batch_size, self.num_tactics, device=device)

        for i, tech_id in enumerate(self.technique_list):
            tactic_indices = self.technique_to_tactic.get(tech_id, [])
            for tactic_idx in tactic_indices:
                tactic_labels[:, tactic_idx] = torch.max(tactic_labels[:, tactic_idx], labels[:, i])

        return tactic_labels

    def _compute_technique_labels(self, labels: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Compute parent technique labels from all technique labels."""
        batch_size = labels.size(0)
        technique_labels = torch.zeros(batch_size, self.num_parent_techniques, device=device)

        for i, tech_id in enumerate(self.technique_list):
            # Check if this is a parent technique (not sub-technique)
            if '.' not in tech_id and tech_id in self.node_to_idx:
                try:
                    parent_idx = self.kg_gnn.technique_indices.index(self.node_to_idx[tech_id])
                    technique_labels[:, parent_idx] = torch.max(technique_labels[:, parent_idx], labels[:, i])
                except ValueError:
                    pass
            # For sub-techniques, also set parent technique label
            elif '.' in tech_id:
                parent_id = tech_id.split('.')[0]
                if parent_id in self.node_to_idx:
                    try:
                        parent_idx = self.kg_gnn.technique_indices.index(self.node_to_idx[parent_id])
                        technique_labels[:, parent_idx] = torch.max(technique_labels[:, parent_idx], labels[:, i])
                    except ValueError:
                        pass

        return technique_labels

    def _compute_subtechnique_labels(self, labels: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Compute sub-technique labels from all technique labels."""
        batch_size = labels.size(0)
        subtechnique_labels = torch.zeros(batch_size, self.num_subtechniques, device=device)

        for i, tech_id in enumerate(self.technique_list):
            if '.' in tech_id and tech_id in self.node_to_idx:
                try:
                    subtech_idx = self.kg_gnn.subtechnique_indices.index(self.node_to_idx[tech_id])
                    subtechnique_labels[:, subtech_idx] = labels[:, i]
                except ValueError:
                    pass

        return subtechnique_labels


def create_enhanced_model(
    kg_dir: str,
    bert_model: str = "bert-base-uncased",
    hidden_dim: int = 512,
    gnn_layers: int = 4,
    num_heads: int = 8,
    dropout: float = 0.1,
    freeze_bert_layers: int = 0,
    use_hierarchical_loss: bool = True,
) -> SecBERTEnhancedGNN:
    """
    Create the enhanced SecBERT + KG-GNN model.

    Args:
        kg_dir: Path to knowledge graph directory
        bert_model: HuggingFace BERT model name
        hidden_dim: Hidden dimension for projections
        gnn_layers: Number of GNN layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        freeze_bert_layers: Number of BERT layers to freeze
        use_hierarchical_loss: Whether to use hierarchical auxiliary losses

    Returns:
        SecBERTEnhancedGNN model
    """
    model = SecBERTEnhancedGNN(
        kg_dir=kg_dir,
        bert_model=bert_model,
        hidden_dim=hidden_dim,
        gnn_layers=gnn_layers,
        num_heads=num_heads,
        dropout=dropout,
        freeze_bert_layers=freeze_bert_layers,
        use_hierarchical_loss=use_hierarchical_loss,
    )

    return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--kg_dir', type=str, default='../data/attack_framework')
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased')
    args = parser.parse_args()

    print("Testing SecBERT + Enhanced KG-GNN Model...")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    model = create_enhanced_model(
        kg_dir=args.kg_dir,
        bert_model=args.bert_model,
        hidden_dim=512,
        gnn_layers=4,
        use_hierarchical_loss=True,
    )

    print(f"\nModel Statistics:")
    print(f"  - Num techniques: {model.num_techniques}")
    print(f"  - Num tactics: {model.num_tactics}")
    print(f"  - Num parent techniques: {model.num_parent_techniques}")
    print(f"  - Num sub-techniques: {model.num_subtechniques}")

    # Pre-compute KG embeddings
    model.to(device)
    print("\nPre-computing KG semantic embeddings...")
    model.precompute_kg_embeddings(device)

    # Test forward pass
    print("\nTesting forward pass...")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    test_text = "The malware used mimikatz to dump credentials from LSASS memory"
    inputs = tokenizer(
        test_text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    ).to(device)

    # Create dummy labels
    labels = torch.zeros(1, model.num_techniques, device=device)
    labels[0, 0] = 1.0  # Mark one technique as positive

    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=labels,
        )

    print(f"  - Logits shape: {outputs['logits'].shape}")
    print(f"  - Loss: {outputs['loss'].item():.4f}")
    if 'tactic_logits' in outputs:
        print(f"  - Tactic logits shape: {outputs['tactic_logits'].shape}")
        print(f"  - Technique logits shape: {outputs['technique_logits'].shape}")
        print(f"  - Subtechnique logits shape: {outputs['subtechnique_logits'].shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameters:")
    print(f"  - Total: {total_params:,}")
    print(f"  - Trainable: {trainable_params:,}")

    print("\n" + "=" * 70)
    print("Model ready for training!")
