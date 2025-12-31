#!/usr/bin/env python3
"""
Enhanced Knowledge Graph GNN with Semantic Features and Hierarchical Structure.

Key improvements over basic GNN:
1. Semantic node features from ATT&CK descriptions (not random embeddings)
2. Heterogeneous message passing respecting edge types
3. Hierarchical attention for Tactic → Technique → Sub-technique propagation
4. Ontological relationship awareness (USES, MITIGATES, SUBTECHNIQUE_OF, etc.)

Inspired by the hierarchical classification success in ICD-10 medical coding.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, GATv2Conv, MessagePassing
from torch_geometric.data import HeteroData
from transformers import AutoModel, AutoTokenizer


class SemanticNodeEncoder(nn.Module):
    """
    Encode node descriptions into semantic embeddings using a shared BERT model.

    Unlike random embeddings, this captures the actual meaning of each ATT&CK entity.
    Uses the shared BERT model from the main SecBERT encoder to avoid loading twice.
    """

    def __init__(
        self,
        bert_dim: int = 768,  # BERT hidden size
        output_dim: int = 512,
    ):
        super().__init__()
        self.bert_dim = bert_dim
        self.output_dim = output_dim

        # Projection to desired dimension
        self.projection = nn.Linear(bert_dim, output_dim)

    @torch.no_grad()
    def encode_descriptions(
        self,
        descriptions: List[str],
        bert_model: nn.Module,  # Shared BERT model
        tokenizer,  # Shared tokenizer
        batch_size: int = 32,
        device: torch.device = torch.device('cpu'),
    ) -> torch.Tensor:
        """
        Encode a list of descriptions into semantic embeddings using shared BERT.

        Args:
            descriptions: List of text descriptions
            bert_model: Shared BERT model (from main encoder)
            tokenizer: Shared tokenizer
            batch_size: Batch size for encoding
            device: Device to use

        Returns:
            Tensor of shape [num_descriptions, output_dim]
        """
        self.projection.to(device)
        bert_model.eval()

        all_embeddings = []

        for i in range(0, len(descriptions), batch_size):
            batch = descriptions[i:i + batch_size]

            # Tokenize
            inputs = tokenizer(
                batch,
                max_length=256,
                padding=True,
                truncation=True,
                return_tensors='pt',
            ).to(device)

            # Encode with shared BERT
            outputs = bert_model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]

            # Project to output dimension
            projected = self.projection(cls_embeddings)
            all_embeddings.append(projected.cpu())

        return torch.cat(all_embeddings, dim=0)


class HierarchicalAttention(nn.Module):
    """
    Hierarchical attention mechanism for Tactic → Technique → Sub-technique propagation.

    This mirrors the morpheme/etymology approach that worked for ICD-10:
    - Tactics are like "roots" providing high-level semantic meaning
    - Techniques are like "stems" adding specific context
    - Sub-techniques are like "affixes" for precise specification
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Attention for each hierarchical level
        self.tactic_to_technique = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.technique_to_subtechnique = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Level-specific projections
        self.tactic_proj = nn.Linear(hidden_dim, hidden_dim)
        self.technique_proj = nn.Linear(hidden_dim, hidden_dim)
        self.subtechnique_proj = nn.Linear(hidden_dim, hidden_dim)

        # Gating mechanism to control information flow
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        tactic_emb: torch.Tensor,      # [num_tactics, hidden]
        technique_emb: torch.Tensor,    # [num_techniques, hidden]
        subtechnique_emb: torch.Tensor, # [num_subtechniques, hidden]
        tactic_technique_idx: torch.Tensor,  # [2, num_edges]
        technique_subtechnique_idx: torch.Tensor,  # [2, num_edges]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Propagate information down the hierarchy.

        Returns:
            Updated embeddings for tactics, techniques, and sub-techniques
        """
        # Project each level (creates new tensors, avoids in-place issues)
        tactic_h = self.tactic_proj(tactic_emb)
        technique_h = self.technique_proj(technique_emb).clone()  # Clone to avoid in-place issues
        subtechnique_h = self.subtechnique_proj(subtechnique_emb).clone()

        # Tactic → Technique attention using batched operations
        if tactic_technique_idx.size(1) > 0 and technique_h.size(0) > 0:
            # Aggregate messages from tactics to techniques using scatter_add (non-inplace)
            src_tactics = tactic_technique_idx[0]  # Tactic indices
            tgt_techniques = tactic_technique_idx[1]  # Technique indices

            # Get tactic embeddings for each edge (clone to avoid in-place issues)
            tactic_msgs = tactic_h[src_tactics].clone().to(technique_h.dtype)

            # Use scatter_add (non-inplace) for aggregation
            tgt_expanded = tgt_techniques.unsqueeze(1).expand(-1, technique_h.size(1))
            technique_updates = torch.zeros_like(technique_h).scatter_add(0, tgt_expanded, tactic_msgs)

            # Count for normalization using scatter_add
            count_ones = torch.ones(tgt_techniques.size(0), dtype=technique_h.dtype, device=technique_h.device)
            technique_counts = torch.zeros(technique_h.size(0), dtype=technique_h.dtype, device=technique_h.device)
            technique_counts = technique_counts.scatter_add(0, tgt_techniques, count_ones)
            technique_counts = technique_counts.clamp(min=1).unsqueeze(1)
            technique_updates = technique_updates / technique_counts

            # Gated update (no in-place operations)
            gate_input = torch.cat([technique_h, technique_updates], dim=-1)
            gate = self.gate(gate_input)
            technique_h = self.norm1(technique_h + gate * technique_updates)

        # Technique → Sub-technique attention using batched operations
        if technique_subtechnique_idx.size(1) > 0 and subtechnique_h.size(0) > 0:
            src_techniques = technique_subtechnique_idx[0]
            tgt_subtechniques = technique_subtechnique_idx[1]

            # Clone to avoid in-place issues with mixed precision
            technique_msgs = technique_h[src_techniques].clone().to(subtechnique_h.dtype)

            # Use scatter_add (non-inplace) for aggregation
            tgt_expanded = tgt_subtechniques.unsqueeze(1).expand(-1, subtechnique_h.size(1))
            subtechnique_updates = torch.zeros_like(subtechnique_h).scatter_add(0, tgt_expanded, technique_msgs)

            # Count for normalization
            count_ones = torch.ones(tgt_subtechniques.size(0), dtype=subtechnique_h.dtype, device=subtechnique_h.device)
            subtechnique_counts = torch.zeros(subtechnique_h.size(0), dtype=subtechnique_h.dtype, device=subtechnique_h.device)
            subtechnique_counts = subtechnique_counts.scatter_add(0, tgt_subtechniques, count_ones)
            subtechnique_counts = subtechnique_counts.clamp(min=1).unsqueeze(1)
            subtechnique_updates = subtechnique_updates / subtechnique_counts

            gate_input = torch.cat([subtechnique_h, subtechnique_updates], dim=-1)
            gate = self.gate(gate_input)
            subtechnique_h = self.norm2(subtechnique_h + gate * subtechnique_updates)

        return tactic_h, technique_h, subtechnique_h


class OntologyAwareConv(MessagePassing):
    """
    Message passing layer that is aware of ontological relationships.

    Different edge types carry different semantic meaning:
    - USES: Threat actors/malware use techniques
    - MITIGATES: Defenses counter techniques
    - SUBTECHNIQUE_OF: Hierarchical refinement
    - TACTIC_HAS_TECHNIQUE: Category membership
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_edge_types: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__(aggr='mean')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_edge_types = num_edge_types

        # Edge-type specific transformations
        self.edge_type_transforms = nn.ModuleList([
            nn.Linear(in_channels, out_channels)
            for _ in range(num_edge_types)
        ])

        # Self-loop transformation
        self.self_loop = nn.Linear(in_channels, out_channels)

        # Attention weights per edge type
        self.edge_attention = nn.Parameter(torch.ones(num_edge_types))

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with edge-type aware message passing.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_type: Edge type for each edge [num_edges]
        """
        # Self-loop
        out = self.self_loop(x)

        # Normalize attention weights
        attention = F.softmax(self.edge_attention, dim=0)

        # Message passing for each edge type
        for etype in range(self.num_edge_types):
            mask = edge_type == etype
            if mask.sum() > 0:
                edge_subset = edge_index[:, mask]

                # Transform source nodes with edge-specific weights
                x_transformed = self.edge_type_transforms[etype](x)

                # Aggregate messages
                messages = self.propagate(edge_subset, x=x_transformed)
                out = out + attention[etype] * messages

        out = self.norm(out)
        out = F.relu(out)
        out = self.dropout(out)

        return out

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        return x_j


class EnhancedKnowledgeGraphGNN(nn.Module):
    """
    Enhanced GNN for ATT&CK Knowledge Graph with:
    1. Semantic node embeddings from descriptions
    2. Heterogeneous message passing respecting edge types
    3. Hierarchical attention for tactic/technique/sub-technique
    4. Ontology-aware convolutions

    This architecture mirrors the successful ICD-10 approach:
    - Semantic understanding (descriptions as morphemes)
    - Hierarchical structure (tactic→technique→subtechnique like ICD chapters→blocks→codes)
    - Relational reasoning (uses/mitigates as etymological connections)
    """

    def __init__(
        self,
        kg_dir: str,
        hidden_dim: int = 512,
        num_layers: int = 4,  # More layers for deeper reasoning
        num_heads: int = 8,
        dropout: float = 0.1,
        use_semantic_features: bool = True,
        bert_dim: int = 768,  # BERT hidden size (to avoid loading BERT again)
    ):
        super().__init__()

        self.kg_dir = Path(kg_dir)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_semantic_features = use_semantic_features

        # Load knowledge graph
        self._load_kg()

        # Semantic encoder for node descriptions (uses shared BERT)
        if use_semantic_features:
            self.semantic_encoder = SemanticNodeEncoder(
                bert_dim=bert_dim,
                output_dim=hidden_dim,
            )
        else:
            # Fallback to learnable embeddings
            self.node_embeddings = nn.Embedding(self.num_nodes, hidden_dim)
            nn.init.xavier_uniform_(self.node_embeddings.weight)

        # Node type embeddings (adds type-specific bias)
        self.node_type_embedding = nn.Embedding(len(self.node_types), hidden_dim)

        # Hierarchical attention
        self.hierarchical_attention = HierarchicalAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Ontology-aware GNN layers
        self.gnn_layers = nn.ModuleList([
            OntologyAwareConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                num_edge_types=len(self.edge_types),
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Final attention layer with graph attention
        self.final_attention = GATv2Conv(
            hidden_dim, hidden_dim,
            heads=num_heads,
            concat=False,
            dropout=dropout,
        )

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Cached semantic embeddings
        self.register_buffer('semantic_embeddings', None)

    def _load_kg(self):
        """Load knowledge graph structure."""
        # Load nodes
        with open(self.kg_dir / 'graph_nodes.json', 'r') as f:
            self.nodes = json.load(f)

        # Load edges
        with open(self.kg_dir / 'graph_edges.json', 'r') as f:
            self.edges = json.load(f)

        # Load node index mapping
        with open(self.kg_dir / 'node_to_index.json', 'r') as f:
            self.node_to_idx = json.load(f)

        # Load technique list
        with open(self.kg_dir / 'technique_list.json', 'r') as f:
            self.technique_list = json.load(f)

        self.idx_to_node = {v: k for k, v in self.node_to_idx.items()}
        self.num_nodes = len(self.nodes)
        self.num_techniques = len(self.technique_list)

        # Extract node types and create type mapping
        self.node_types = ['tactic', 'technique', 'subtechnique', 'group', 'malware', 'tool', 'mitigation']
        self.type_to_idx = {t: i for i, t in enumerate(self.node_types)}

        # Create node type tensor
        self.node_type_list = []
        for node_id in sorted(self.node_to_idx.keys(), key=lambda x: self.node_to_idx[x]):
            node_type = self.nodes[node_id]['type']
            self.node_type_list.append(self.type_to_idx.get(node_type, 0))

        # Extract edge types
        self.edge_types = ['USES', 'MITIGATES', 'SUBTECHNIQUE_OF', 'TACTIC_HAS_TECHNIQUE', 'TECHNIQUE_HAS_SUBTECHNIQUE']
        self.edge_type_to_idx = {t: i for i, t in enumerate(self.edge_types)}

        # Collect node descriptions for semantic encoding
        self.node_descriptions = []
        for node_id in sorted(self.node_to_idx.keys(), key=lambda x: self.node_to_idx[x]):
            node = self.nodes[node_id]
            # Combine name and description for richer semantics
            desc = f"{node.get('name', '')}. {node.get('description', '')}"
            self.node_descriptions.append(desc[:512])  # Truncate for efficiency

        # Build hierarchical indices
        self._build_hierarchical_indices()

    def _build_hierarchical_indices(self):
        """Build indices for hierarchical propagation."""
        tactic_indices = []
        technique_indices = []
        subtechnique_indices = []

        tactic_technique_src = []
        tactic_technique_tgt = []
        technique_subtechnique_src = []
        technique_subtechnique_tgt = []

        # Map node IDs to their type-specific indices
        tactic_map = {}
        technique_map = {}
        subtechnique_map = {}

        for node_id, node in self.nodes.items():
            idx = self.node_to_idx[node_id]
            node_type = node['type']

            if node_type == 'tactic':
                tactic_map[node_id] = len(tactic_indices)
                tactic_indices.append(idx)
            elif node_type == 'technique':
                technique_map[node_id] = len(technique_indices)
                technique_indices.append(idx)
            elif node_type == 'subtechnique':
                subtechnique_map[node_id] = len(subtechnique_indices)
                subtechnique_indices.append(idx)

        # Build hierarchical edges
        for edge in self.edges:
            if edge['type'] == 'TACTIC_HAS_TECHNIQUE':
                if edge['source'] in tactic_map and edge['target'] in technique_map:
                    tactic_technique_src.append(tactic_map[edge['source']])
                    tactic_technique_tgt.append(technique_map[edge['target']])
            elif edge['type'] == 'TECHNIQUE_HAS_SUBTECHNIQUE':
                if edge['source'] in technique_map and edge['target'] in subtechnique_map:
                    technique_subtechnique_src.append(technique_map[edge['source']])
                    technique_subtechnique_tgt.append(subtechnique_map[edge['target']])

        self.tactic_indices = tactic_indices
        self.technique_indices = technique_indices
        self.subtechnique_indices = subtechnique_indices

        self.tactic_technique_idx = torch.tensor([tactic_technique_src, tactic_technique_tgt], dtype=torch.long)
        self.technique_subtechnique_idx = torch.tensor([technique_subtechnique_src, technique_subtechnique_tgt], dtype=torch.long)

    def precompute_semantic_embeddings(
        self,
        device: torch.device,
        bert_model: nn.Module = None,
        tokenizer = None,
    ):
        """
        Pre-compute semantic embeddings for all nodes using shared BERT.
        Call this once before training.

        Args:
            device: Device to use
            bert_model: Shared BERT model from the main encoder
            tokenizer: Shared tokenizer
        """
        if self.use_semantic_features and self.semantic_embeddings is None:
            if bert_model is None or tokenizer is None:
                print("Warning: Shared BERT/tokenizer not provided, using learnable embeddings")
                self.use_semantic_features = False
                self.node_embeddings = nn.Embedding(self.num_nodes, self.hidden_dim)
                nn.init.xavier_uniform_(self.node_embeddings.weight)
                return

            print("Computing semantic embeddings for knowledge graph nodes...")
            embeddings = self.semantic_encoder.encode_descriptions(
                self.node_descriptions,
                bert_model=bert_model,
                tokenizer=tokenizer,
                batch_size=32,
                device=device,
            )
            self.semantic_embeddings = embeddings.to(device)
            print(f"  - Computed {embeddings.shape[0]} semantic embeddings")

    def get_edge_index_and_types(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get edge index and edge type tensors."""
        sources = []
        targets = []
        edge_types = []

        for edge in self.edges:
            src = edge['source']
            tgt = edge['target']
            etype = edge['type']

            if src in self.node_to_idx and tgt in self.node_to_idx:
                sources.append(self.node_to_idx[src])
                targets.append(self.node_to_idx[tgt])
                edge_types.append(self.edge_type_to_idx.get(etype, 0))

        # Make bidirectional
        edge_index = torch.tensor([
            sources + targets,
            targets + sources
        ], dtype=torch.long, device=device)

        edge_type = torch.tensor(
            edge_types + edge_types,
            dtype=torch.long,
            device=device
        )

        return edge_index, edge_type

    def forward(self, device: torch.device) -> torch.Tensor:
        """
        Forward pass through the enhanced GNN.

        Returns:
            Node embeddings [num_nodes, hidden_dim]
        """
        # Get initial node features
        if self.use_semantic_features and self.semantic_embeddings is not None:
            x = self.semantic_embeddings.to(device)
        else:
            node_ids = torch.arange(self.num_nodes, device=device)
            x = self.node_embeddings(node_ids)

        # Add node type embeddings
        node_types = torch.tensor(self.node_type_list, dtype=torch.long, device=device)
        type_emb = self.node_type_embedding(node_types)
        x = x + type_emb

        # Get edge information
        edge_index, edge_type = self.get_edge_index_and_types(device)

        # Hierarchical attention (propagate from tactics → techniques → sub-techniques)
        if len(self.tactic_indices) > 0 and len(self.technique_indices) > 0:
            # Clone all sliced embeddings to avoid in-place modification issues during backprop
            tactic_emb = x[self.tactic_indices].clone()
            technique_emb = x[self.technique_indices].clone()
            subtechnique_emb = x[self.subtechnique_indices].clone() if self.subtechnique_indices else torch.zeros(0, self.hidden_dim, device=device)

            tactic_h, technique_h, subtechnique_h = self.hierarchical_attention(
                tactic_emb,
                technique_emb,
                subtechnique_emb,
                self.tactic_technique_idx.to(device),
                self.technique_subtechnique_idx.to(device),
            )

            # Update node embeddings with hierarchical information
            # Use scatter_add which returns new tensor (no in-place modification)
            tactic_idx_tensor = torch.tensor(self.tactic_indices, dtype=torch.long, device=device)
            technique_idx_tensor = torch.tensor(self.technique_indices, dtype=torch.long, device=device)

            # Create expanded index tensors for scatter_add [N] -> [N, hidden_dim]
            tactic_idx_expanded = tactic_idx_tensor.unsqueeze(1).expand(-1, x.size(1))
            technique_idx_expanded = technique_idx_tensor.unsqueeze(1).expand(-1, x.size(1))

            # Use scatter_add to create new tensors (not in-place)
            x = x.scatter_add(0, tactic_idx_expanded, tactic_h.to(x.dtype))
            x = x.scatter_add(0, technique_idx_expanded, technique_h.to(x.dtype))
            if len(self.subtechnique_indices) > 0:
                subtechnique_idx_tensor = torch.tensor(self.subtechnique_indices, dtype=torch.long, device=device)
                subtechnique_idx_expanded = subtechnique_idx_tensor.unsqueeze(1).expand(-1, x.size(1))
                x = x.scatter_add(0, subtechnique_idx_expanded, subtechnique_h.to(x.dtype))

        # Ontology-aware GNN layers
        for layer in self.gnn_layers:
            x_new = layer(x, edge_index, edge_type)
            x = x + x_new  # Residual connection

        # Final attention layer
        x = self.final_attention(x, edge_index)

        # Output projection
        x = self.output_projection(x)

        return x

    def get_technique_embeddings(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings for technique/sub-technique nodes only.

        Args:
            node_embeddings: Full node embeddings from forward()

        Returns:
            Technique embeddings [num_techniques, hidden_dim]
        """
        technique_embs = []
        for tech_id in self.technique_list:
            if tech_id in self.node_to_idx:
                idx = self.node_to_idx[tech_id]
                technique_embs.append(node_embeddings[idx])
            else:
                # Fallback for missing techniques
                technique_embs.append(torch.zeros(self.hidden_dim, device=node_embeddings.device))

        return torch.stack(technique_embs, dim=0)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--kg_dir', type=str, default='../data/attack_framework')
    args = parser.parse_args()

    print("Testing Enhanced Knowledge Graph GNN...")
    print("=" * 70)

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EnhancedKnowledgeGraphGNN(
        kg_dir=args.kg_dir,
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        use_semantic_features=True,  # Use descriptions
    )

    print(f"\nModel Statistics:")
    print(f"  - Nodes: {model.num_nodes}")
    print(f"  - Techniques: {model.num_techniques}")
    print(f"  - Node types: {model.node_types}")
    print(f"  - Edge types: {model.edge_types}")
    print(f"  - Tactics: {len(model.tactic_indices)}")
    print(f"  - Techniques: {len(model.technique_indices)}")
    print(f"  - Sub-techniques: {len(model.subtechnique_indices)}")

    # Pre-compute semantic embeddings
    model.to(device)
    model.precompute_semantic_embeddings(device)

    # Forward pass
    print("\nRunning forward pass...")
    node_embeddings = model(device)
    print(f"  - Node embeddings shape: {node_embeddings.shape}")

    technique_embeddings = model.get_technique_embeddings(node_embeddings)
    print(f"  - Technique embeddings shape: {technique_embeddings.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameters:")
    print(f"  - Total: {total_params:,}")
    print(f"  - Trainable: {trainable_params:,}")

    print("\n" + "=" * 70)
    print("Enhanced KG-GNN ready for integration with SecBERT!")
