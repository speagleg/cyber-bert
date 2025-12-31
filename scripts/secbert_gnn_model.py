#!/usr/bin/env python3
"""
SecBERT-GNN: Hybrid BERT + Graph Neural Network for MITRE ATT&CK Classification.

This model combines:
1. SecBERT/BERT for encoding security log text
2. GNN for encoding MITRE ATT&CK knowledge graph relationships
3. Multi-label classification head for technique prediction

No pretraining required - trains end-to-end on labeled security data.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATv2Conv, global_mean_pool
from torch_geometric.data import Data
from transformers import AutoModel, AutoTokenizer


class AttackKnowledgeGraph:
    """Load and manage MITRE ATT&CK knowledge graph."""

    def __init__(self, kg_dir: str):
        self.kg_dir = Path(kg_dir)
        self.nodes: Dict = {}
        self.edges: List = []
        self.node_to_idx: Dict[str, int] = {}
        self.idx_to_node: Dict[int, str] = {}
        self.technique_list: List[str] = []
        self.num_nodes = 0
        self.num_techniques = 0

        self._load_graph()

    def _load_graph(self):
        """Load graph from files."""
        # Load nodes
        nodes_file = self.kg_dir / 'graph_nodes.json'
        with open(nodes_file, 'r') as f:
            self.nodes = json.load(f)

        # Load edges
        edges_file = self.kg_dir / 'graph_edges.json'
        with open(edges_file, 'r') as f:
            self.edges = json.load(f)

        # Load node index mapping
        index_file = self.kg_dir / 'node_to_index.json'
        with open(index_file, 'r') as f:
            self.node_to_idx = json.load(f)

        self.idx_to_node = {v: k for k, v in self.node_to_idx.items()}
        self.num_nodes = len(self.nodes)

        # Load technique list (classification labels)
        techniques_file = self.kg_dir / 'technique_list.json'
        with open(techniques_file, 'r') as f:
            self.technique_list = json.load(f)

        self.num_techniques = len(self.technique_list)
        self.technique_to_idx = {t: i for i, t in enumerate(self.technique_list)}

    def get_edge_index(self) -> torch.Tensor:
        """Get edge index tensor for PyG."""
        sources = []
        targets = []

        for edge in self.edges:
            src = edge['source']
            tgt = edge['target']

            if src in self.node_to_idx and tgt in self.node_to_idx:
                sources.append(self.node_to_idx[src])
                targets.append(self.node_to_idx[tgt])

        # Make bidirectional
        edge_index = torch.tensor([
            sources + targets,
            targets + sources
        ], dtype=torch.long)

        return edge_index

    def get_node_features(self, hidden_dim: int) -> torch.Tensor:
        """Get initial node features (learnable embeddings)."""
        return torch.randn(self.num_nodes, hidden_dim) * 0.02

    def techniques_to_labels(self, techniques: List[str]) -> torch.Tensor:
        """Convert technique IDs to multi-hot label vector."""
        labels = torch.zeros(self.num_techniques)
        for tech in techniques:
            if tech in self.technique_to_idx:
                labels[self.technique_to_idx[tech]] = 1.0
        return labels

    def labels_to_techniques(self, labels: torch.Tensor, threshold: float = 0.5) -> List[str]:
        """Convert predictions to technique IDs."""
        indices = (labels > threshold).nonzero(as_tuple=True)[0]
        return [self.technique_list[i] for i in indices.tolist()]


class GNNEncoder(nn.Module):
    """Graph Neural Network encoder for ATT&CK knowledge graph."""

    def __init__(
        self,
        num_nodes: int,
        input_dim: int = 256,
        hidden_dim: int = 512,
        output_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        # Learnable node embeddings
        self.node_embeddings = nn.Embedding(num_nodes, input_dim)
        nn.init.xavier_uniform_(self.node_embeddings.weight)

        # GNN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))

        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Final layer with attention
        self.convs.append(GATv2Conv(hidden_dim, output_dim, heads=4, concat=False))
        self.norms.append(nn.LayerNorm(output_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GNN.

        Args:
            edge_index: Edge index tensor [2, num_edges]

        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        # Get initial node features
        node_ids = torch.arange(self.num_nodes, device=edge_index.device)
        x = self.node_embeddings(node_ids)

        # Apply GNN layers
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)

        return x


class SecBERTGNNModel(nn.Module):
    """
    SecBERT-GNN: Hybrid model for MITRE ATT&CK technique classification.

    Combines BERT text encoding with GNN-encoded knowledge graph
    for multi-label classification of security events.
    """

    def __init__(
        self,
        kg: AttackKnowledgeGraph,
        bert_model: str = "bert-base-uncased",
        hidden_dim: int = 512,
        gnn_layers: int = 3,
        dropout: float = 0.1,
        freeze_bert_layers: int = 0,
    ):
        super().__init__()

        self.kg = kg
        self.hidden_dim = hidden_dim
        self.num_techniques = kg.num_techniques

        # BERT encoder
        self.bert = AutoModel.from_pretrained(bert_model)
        self.bert_dim = self.bert.config.hidden_size  # Usually 768

        # Optionally freeze early BERT layers
        if freeze_bert_layers > 0:
            for i, layer in enumerate(self.bert.encoder.layer):
                if i < freeze_bert_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

        # Project BERT output to shared dimension
        self.text_projection = nn.Sequential(
            nn.Linear(self.bert_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # GNN encoder for knowledge graph
        self.gnn = GNNEncoder(
            num_nodes=kg.num_nodes,
            input_dim=256,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=gnn_layers,
            dropout=dropout,
        )

        # Attention mechanism to weight graph nodes based on text
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )

        # Classification head
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

        # Store edge index (will be set during training)
        self.register_buffer('edge_index', None)

    def set_edge_index(self, edge_index: torch.Tensor):
        """Set the graph edge index."""
        self.edge_index = edge_index

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
        # Use [CLS] token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.text_projection(cls_embedding)

    def encode_graph(self) -> torch.Tensor:
        """Encode knowledge graph using GNN."""
        return self.gnn(self.edge_index)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Multi-hot labels [batch_size, num_techniques] (optional)

        Returns:
            Dictionary with logits and optionally loss
        """
        batch_size = input_ids.size(0)

        # Encode text
        text_emb = self.encode_text(input_ids, attention_mask)  # [batch, hidden]

        # Encode graph
        graph_emb = self.encode_graph()  # [num_nodes, hidden]

        # Get technique node embeddings only
        technique_indices = [
            self.kg.node_to_idx[t] for t in self.kg.technique_list
            if t in self.kg.node_to_idx
        ]
        technique_indices = torch.tensor(technique_indices, device=graph_emb.device)
        technique_emb = graph_emb[technique_indices]  # [num_techniques, hidden]

        # Cross-attention: text attends to technique embeddings
        text_emb_expanded = text_emb.unsqueeze(1)  # [batch, 1, hidden]
        technique_emb_expanded = technique_emb.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch, num_techniques, hidden]

        attended_graph, _ = self.cross_attention(
            query=text_emb_expanded,
            key=technique_emb_expanded,
            value=technique_emb_expanded,
        )
        attended_graph = attended_graph.squeeze(1)  # [batch, hidden]

        # Combine text and graph representations
        combined = torch.cat([text_emb, attended_graph], dim=-1)  # [batch, hidden*2]

        # Classification
        logits = self.classifier(combined)  # [batch, num_techniques]

        output = {'logits': logits}

        if labels is not None:
            # Multi-label binary cross entropy loss
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            output['loss'] = loss

        return output

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, List[List[str]]]:
        """
        Make predictions.

        Returns:
            Tuple of (probabilities, list of technique IDs per sample)
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(input_ids, attention_mask)
            probs = torch.sigmoid(output['logits'])

            techniques = []
            for i in range(probs.size(0)):
                sample_techniques = self.kg.labels_to_techniques(probs[i], threshold)
                techniques.append(sample_techniques)

            return probs, techniques


class SecBERTGNNForInference:
    """Wrapper for easy inference."""

    def __init__(
        self,
        model_path: str,
        kg_dir: str,
        bert_model: str = "bert-base-uncased",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device

        # Load knowledge graph
        self.kg = AttackKnowledgeGraph(kg_dir)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)

        # Load model
        self.model = SecBERTGNNModel(
            kg=self.kg,
            bert_model=bert_model,
        )

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Set edge index
        edge_index = self.kg.get_edge_index().to(device)
        self.model.set_edge_index(edge_index)

        self.model.to(device)
        self.model.eval()

    def predict(
        self,
        text: str,
        threshold: float = 0.5,
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Predict ATT&CK techniques for input text.

        Returns:
            List of dicts with technique ID, name, and confidence
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        # Predict
        with torch.no_grad():
            output = self.model(input_ids, attention_mask)
            probs = torch.sigmoid(output['logits'])[0]

        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))

        results = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            tech_id = self.kg.technique_list[idx]
            tech_node = self.kg.nodes.get(tech_id, {})

            results.append({
                'technique_id': tech_id,
                'name': tech_node.get('name', ''),
                'confidence': prob,
                'above_threshold': prob >= threshold,
            })

        return results


def create_model(
    kg_dir: str,
    bert_model: str = "bert-base-uncased",
    hidden_dim: int = 512,
    gnn_layers: int = 3,
    dropout: float = 0.1,
    freeze_bert_layers: int = 0,
) -> Tuple[SecBERTGNNModel, AttackKnowledgeGraph]:
    """
    Create SecBERT-GNN model.

    Args:
        kg_dir: Path to knowledge graph directory
        bert_model: HuggingFace BERT model name
        hidden_dim: Hidden dimension for projections
        gnn_layers: Number of GNN layers
        dropout: Dropout rate
        freeze_bert_layers: Number of BERT layers to freeze

    Returns:
        Tuple of (model, knowledge_graph)
    """
    kg = AttackKnowledgeGraph(kg_dir)

    model = SecBERTGNNModel(
        kg=kg,
        bert_model=bert_model,
        hidden_dim=hidden_dim,
        gnn_layers=gnn_layers,
        dropout=dropout,
        freeze_bert_layers=freeze_bert_layers,
    )

    return model, kg


if __name__ == '__main__':
    # Quick test
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--kg_dir', type=str, default='data/attack_framework')
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased')
    args = parser.parse_args()

    print("Creating model...")
    model, kg = create_model(args.kg_dir, args.bert_model)

    print(f"\nModel created successfully!")
    print(f"  - Number of nodes: {kg.num_nodes}")
    print(f"  - Number of techniques: {kg.num_techniques}")
    print(f"  - BERT model: {args.bert_model}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
