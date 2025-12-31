#!/usr/bin/env python3
"""
Advanced Knowledge Graph GNN with High-Impact Enhancements.

Four key improvements:
1. R-GCN (Relational GCN) - Edge-type aware message passing
2. HGT (Heterogeneous Graph Transformer) - Node/edge type-specific attention
3. TransE Pre-training - Knowledge graph embeddings for initialization
4. Dynamic Text-Conditioned Attention - Input-dependent graph attention

These enhancements are designed to significantly improve precision by:
- Better distinguishing between similar techniques
- Leveraging relational semantics (USES, MITIGATES, etc.)
- Learning richer KG representations before task training
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATv2Conv
from torch_geometric.utils import softmax


# =============================================================================
# 1. R-GCN: Relational Graph Convolutional Network
# =============================================================================

class RGCNConv(MessagePassing):
    """
    Relational Graph Convolutional Network layer.

    Different edge types get different weight matrices, allowing the model
    to learn type-specific transformations:
    - USES edges: How threat actors employ techniques
    - MITIGATES edges: How defenses counter techniques
    - SUBTECHNIQUE_OF: Hierarchical refinement relationships
    - TACTIC_HAS_TECHNIQUE: Category membership

    Reference: Schlichtkrull et al. "Modeling Relational Data with GCNs"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        num_bases: int = None,  # Basis decomposition for parameter efficiency
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__(aggr='mean')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases or num_relations  # Default: no decomposition

        # Basis weight matrices (for decomposition)
        self.basis = nn.Parameter(torch.Tensor(self.num_bases, in_channels, out_channels))

        # Coefficients to combine bases for each relation
        if num_bases < num_relations:
            self.att = nn.Parameter(torch.Tensor(num_relations, self.num_bases))
        else:
            self.register_parameter('att', None)

        # Self-loop weight
        self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.basis)
        nn.init.xavier_uniform_(self.root)
        if self.att is not None:
            nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_type: Edge type for each edge [num_edges]
        """
        out = x @ self.root  # Self-loop

        # Compute relation-specific weights
        if self.att is not None:
            # Basis decomposition: W_r = sum_b(a_rb * B_b)
            weight = torch.einsum('rb,bio->rio', self.att, self.basis)
        else:
            weight = self.basis

        # Message passing for each relation type
        for r in range(self.num_relations):
            mask = edge_type == r
            if mask.sum() == 0:
                continue

            edge_index_r = edge_index[:, mask]

            # Transform source nodes
            x_r = x @ weight[r]

            # Aggregate
            out = out + self.propagate(edge_index_r, x=x_r, size=None)

        if self.bias is not None:
            out = out + self.bias

        return self.dropout(F.relu(out))

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        return x_j


# =============================================================================
# 2. HGT: Heterogeneous Graph Transformer
# =============================================================================

class HGTConv(nn.Module):
    """
    Heterogeneous Graph Transformer layer.

    Computes attention based on (source_type, edge_type, target_type) triplets,
    allowing different attention patterns for different meta-relations.

    For ATT&CK KG:
    - (group, USES, technique): How threat actors use techniques
    - (mitigation, MITIGATES, technique): How defenses counter techniques
    - (technique, SUBTECHNIQUE_OF, technique): Hierarchical relationships

    Reference: Hu et al. "Heterogeneous Graph Transformer"
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_node_types: int,
        num_edge_types: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.d_k = hidden_dim // num_heads
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        # Type-specific linear transformations for Q, K, V
        self.q_linear = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_node_types)
        ])
        self.k_linear = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_node_types)
        ])
        self.v_linear = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_node_types)
        ])

        # Edge-type specific attention weights
        self.relation_att = nn.Parameter(torch.Tensor(num_edge_types, num_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(num_edge_types, num_heads, self.d_k, self.d_k))

        # Prior importance per relation
        self.relation_pri = nn.Parameter(torch.ones(num_edge_types, num_heads))

        # Output projection
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_type: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            node_type: Node type for each node [num_nodes]
            edge_type: Edge type for each edge [num_edges]
        """
        num_nodes = x.size(0)
        input_dtype = x.dtype

        # Compute Q, K, V based on node types using list accumulation (AMP-safe)
        q_list = [None] * num_nodes
        k_list = [None] * num_nodes
        v_list = [None] * num_nodes

        for ntype in range(self.num_node_types):
            mask = node_type == ntype
            if mask.sum() == 0:
                continue

            indices = mask.nonzero(as_tuple=True)[0]
            x_ntype = x[mask]

            q_ntype = self.q_linear[ntype](x_ntype).view(-1, self.num_heads, self.d_k)
            k_ntype = self.k_linear[ntype](x_ntype).view(-1, self.num_heads, self.d_k)
            v_ntype = self.v_linear[ntype](x_ntype).view(-1, self.num_heads, self.d_k)

            for i, idx in enumerate(indices.tolist()):
                q_list[idx] = q_ntype[i]
                k_list[idx] = k_ntype[i]
                v_list[idx] = v_ntype[i]

        # Stack into tensors (will use dtype from linear outputs)
        q = torch.stack(q_list, dim=0)  # [N, H, D]
        k = torch.stack(k_list, dim=0)  # [N, H, D]
        v = torch.stack(v_list, dim=0)  # [N, H, D]

        # Compute attention for each edge
        src, dst = edge_index

        # Get edge-specific transformations
        k_rel = torch.einsum('nhd,rhde->nrhe', k[src], self.relation_att)  # [E, R, H, D]
        v_rel = torch.einsum('nhd,rhde->nrhe', v[src], self.relation_msg)  # [E, R, H, D]

        # Select based on actual edge type
        edge_type_expanded = edge_type.view(-1, 1, 1, 1).expand(-1, 1, self.num_heads, self.d_k)
        k_edge = k_rel.gather(1, edge_type_expanded).squeeze(1)  # [E, H, D]
        v_edge = v_rel.gather(1, edge_type_expanded).squeeze(1)  # [E, H, D]

        # Attention scores
        att = (q[dst] * k_edge).sum(dim=-1) / math.sqrt(self.d_k)  # [E, H]

        # Add relation prior
        pri = self.relation_pri[edge_type]  # [E, H]
        att = att * pri

        # Softmax over incoming edges
        att = softmax(att, dst, num_nodes=num_nodes)
        att = self.dropout(att)

        # Aggregate messages (use dtype from q which matches linear output dtype)
        out = torch.zeros(num_nodes, self.num_heads, self.d_k, device=x.device, dtype=q.dtype)
        out.scatter_add_(0, dst.view(-1, 1, 1).expand(-1, self.num_heads, self.d_k), att.unsqueeze(-1) * v_edge)

        # Reshape and project
        out = out.view(num_nodes, self.hidden_dim)
        out = self.out_linear(out)

        # Residual + LayerNorm (cast x to match out dtype for AMP compatibility)
        out = self.layer_norm(x.to(out.dtype) + self.dropout(out))

        return out


# =============================================================================
# 3. TransE Pre-training for Knowledge Graph Embeddings
# =============================================================================

class TransE(nn.Module):
    """
    TransE knowledge graph embedding model.

    Learns embeddings where: head + relation ≈ tail
    This captures relational patterns in the KG before task-specific training.

    For ATT&CK:
    - Learns that "APT28 + USES ≈ T1566" (spearphishing)
    - Captures technique similarity through shared relations

    Reference: Bordes et al. "Translating Embeddings for Modeling Multi-relational Data"
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        margin: float = 1.0,
        p_norm: int = 2,
    ):
        super().__init__()

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        self.margin = margin
        self.p_norm = p_norm

        # Initialize
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

        # Normalize relation embeddings
        with torch.no_grad():
            self.relation_embeddings.weight.data = F.normalize(
                self.relation_embeddings.weight.data, p=2, dim=-1
            )

    def forward(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """Compute TransE scores (lower is better)."""
        h = self.entity_embeddings(head)
        r = self.relation_embeddings(relation)
        t = self.entity_embeddings(tail)

        # Normalize entity embeddings
        h = F.normalize(h, p=2, dim=-1)
        t = F.normalize(t, p=2, dim=-1)

        # Score: ||h + r - t||
        score = torch.norm(h + r - t, p=self.p_norm, dim=-1)
        return score

    def loss(
        self,
        pos_head: torch.Tensor,
        pos_rel: torch.Tensor,
        pos_tail: torch.Tensor,
        neg_head: torch.Tensor,
        neg_tail: torch.Tensor,
    ) -> torch.Tensor:
        """Margin-based ranking loss."""
        pos_score = self.forward(pos_head, pos_rel, pos_tail)
        neg_score = self.forward(neg_head, pos_rel, neg_tail)

        # Margin loss: max(0, margin + pos_score - neg_score)
        loss = F.relu(self.margin + pos_score - neg_score)
        return loss.mean()

    def get_entity_embeddings(self) -> torch.Tensor:
        """Get learned entity embeddings for GNN initialization."""
        return self.entity_embeddings.weight.detach()


class TransEPretrainer:
    """Pre-train TransE embeddings on the ATT&CK knowledge graph."""

    def __init__(
        self,
        edges: List[Dict],
        node_to_idx: Dict[str, int],
        edge_type_to_idx: Dict[str, int],
        embedding_dim: int = 512,
        device: torch.device = None,
    ):
        self.edges = edges
        self.node_to_idx = node_to_idx
        self.edge_type_to_idx = edge_type_to_idx
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = TransE(
            num_entities=len(node_to_idx),
            num_relations=len(edge_type_to_idx),
            embedding_dim=embedding_dim,
        ).to(self.device)

        # Prepare training triples
        self.triples = []
        for edge in edges:
            if edge['source'] in node_to_idx and edge['target'] in node_to_idx:
                h = node_to_idx[edge['source']]
                r = edge_type_to_idx.get(edge['type'], 0)
                t = node_to_idx[edge['target']]
                self.triples.append((h, r, t))

        self.triples = torch.tensor(self.triples, dtype=torch.long, device=self.device)

    def train(self, epochs: int = 100, lr: float = 0.01, batch_size: int = 256) -> torch.Tensor:
        """Train TransE and return learned embeddings."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        num_entities = len(self.node_to_idx)

        print(f"Pre-training TransE embeddings on {len(self.triples)} triples...")

        for epoch in range(epochs):
            # Shuffle triples
            perm = torch.randperm(len(self.triples))
            total_loss = 0

            for i in range(0, len(self.triples), batch_size):
                batch = self.triples[perm[i:i + batch_size]]

                pos_head = batch[:, 0]
                pos_rel = batch[:, 1]
                pos_tail = batch[:, 2]

                # Negative sampling (corrupt head or tail)
                neg_head = torch.randint(0, num_entities, pos_head.shape, device=self.device)
                neg_tail = torch.randint(0, num_entities, pos_tail.shape, device=self.device)

                # Random choice: corrupt head or tail
                mask = torch.rand(len(batch), device=self.device) > 0.5
                neg_head = torch.where(mask, neg_head, pos_head)
                neg_tail = torch.where(~mask, neg_tail, pos_tail)

                optimizer.zero_grad()
                loss = self.model.loss(pos_head, pos_rel, pos_tail, neg_head, neg_tail)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

        return self.model.get_entity_embeddings()


# =============================================================================
# 4. Dynamic Text-Conditioned Graph Attention
# =============================================================================

class DynamicTextConditionedAttention(nn.Module):
    """
    Graph attention conditioned on input text.

    Different security logs should attend to different parts of the KG.
    For example:
    - PowerShell logs → higher attention to T1059.001 and related techniques
    - Network logs → higher attention to C2 and lateral movement techniques

    This goes beyond static cross-attention by making the GNN itself text-aware.
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

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.d_k = hidden_dim // num_heads

        # Text-to-attention projection
        self.text_to_att = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_heads),  # One attention weight per head
        )

        # Query, Key, Value projections
        self.q_proj = nn.Linear(kg_dim, hidden_dim)
        self.k_proj = nn.Linear(kg_dim, hidden_dim)
        self.v_proj = nn.Linear(kg_dim, hidden_dim)

        # Text-conditioned key modulation
        self.text_key_mod = nn.Linear(text_dim, hidden_dim)

        # Output
        self.out_proj = nn.Linear(hidden_dim, kg_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(kg_dim)

    def forward(
        self,
        kg_emb: torch.Tensor,       # [num_nodes, kg_dim]
        text_emb: torch.Tensor,     # [batch_size, text_dim]
        edge_index: torch.Tensor,   # [2, num_edges]
    ) -> torch.Tensor:
        """
        Apply text-conditioned attention to KG embeddings.

        Returns:
            Text-conditioned KG embeddings [batch_size, num_nodes, kg_dim]
        """
        batch_size = text_emb.size(0)
        num_nodes = kg_emb.size(0)

        # Compute base Q, K, V
        Q = self.q_proj(kg_emb)  # [N, H*D]
        K = self.k_proj(kg_emb)  # [N, H*D]
        V = self.v_proj(kg_emb)  # [N, H*D]

        # Text-conditioned key modulation
        text_mod = self.text_key_mod(text_emb)  # [B, H*D]

        # Modulate keys based on text (makes attention text-dependent)
        # K_cond[b, n] = K[n] * sigmoid(text_mod[b])
        K_cond = K.unsqueeze(0) * torch.sigmoid(text_mod.unsqueeze(1))  # [B, N, H*D]

        # Reshape for multi-head attention
        Q = Q.view(num_nodes, self.num_heads, self.d_k)  # [N, H, D]
        K_cond = K_cond.view(batch_size, num_nodes, self.num_heads, self.d_k)  # [B, N, H, D]
        V = V.view(num_nodes, self.num_heads, self.d_k)  # [N, H, D]

        # Compute attention scores along edges
        src, dst = edge_index

        # For each batch, compute edge attention
        outputs = []
        for b in range(batch_size):
            # Attention: Q[dst] * K_cond[b, src]
            q_dst = Q[dst]  # [E, H, D]
            k_src = K_cond[b, src]  # [E, H, D]

            att = (q_dst * k_src).sum(dim=-1) / math.sqrt(self.d_k)  # [E, H]
            att = softmax(att, dst, num_nodes=num_nodes)
            att = self.dropout(att)

            # Aggregate
            v_src = V[src]  # [E, H, D]
            out = torch.zeros(num_nodes, self.num_heads, self.d_k, device=kg_emb.device, dtype=kg_emb.dtype)
            out.scatter_add_(0, dst.view(-1, 1, 1).expand(-1, self.num_heads, self.d_k), att.unsqueeze(-1) * v_src)

            out = out.view(num_nodes, self.hidden_dim)
            out = self.out_proj(out)
            out = self.layer_norm(kg_emb + self.dropout(out))
            outputs.append(out)

        return torch.stack(outputs, dim=0)  # [B, N, kg_dim]


# =============================================================================
# Complete Advanced GNN Model
# =============================================================================

class AdvancedKnowledgeGraphGNN(nn.Module):
    """
    Advanced GNN combining all four high-impact enhancements:
    1. R-GCN for edge-type aware message passing
    2. HGT for heterogeneous node/edge attention
    3. TransE pre-training for rich initialization
    4. Dynamic text-conditioned attention
    """

    def __init__(
        self,
        kg_dir: str,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_transe_pretrain: bool = True,
        transe_epochs: int = 100,
    ):
        super().__init__()

        self.kg_dir = Path(kg_dir)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_transe_pretrain = use_transe_pretrain
        self.transe_epochs = transe_epochs

        # Load KG
        self._load_kg()

        # Node type and edge type counts
        self.num_node_types = len(self.node_types)
        self.num_edge_types = len(self.edge_types)

        # Initial embeddings (will be set by TransE or random)
        self.node_embeddings = nn.Embedding(self.num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.node_embeddings.weight)

        # Node type embeddings
        self.node_type_embedding = nn.Embedding(self.num_node_types, hidden_dim)

        # R-GCN layers
        self.rgcn_layers = nn.ModuleList([
            RGCNConv(
                hidden_dim, hidden_dim,
                num_relations=self.num_edge_types,
                num_bases=min(4, self.num_edge_types),  # Basis decomposition
                dropout=dropout,
            )
            for _ in range(num_layers // 2)
        ])

        # HGT layers
        self.hgt_layers = nn.ModuleList([
            HGTConv(
                hidden_dim,
                num_heads=num_heads,
                num_node_types=self.num_node_types,
                num_edge_types=self.num_edge_types,
                dropout=dropout,
            )
            for _ in range(num_layers // 2)
        ])

        # Dynamic text-conditioned attention (to be applied at inference)
        self.text_conditioned_att = DynamicTextConditionedAttention(
            text_dim=hidden_dim,
            kg_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Cache for edge tensors
        self._edge_index = None
        self._edge_type = None
        self._node_type = None

        # TransE pre-trained embeddings (set during initialization)
        self.register_buffer('transe_embeddings', None)

    def _load_kg(self):
        """Load knowledge graph structure."""
        with open(self.kg_dir / 'graph_nodes.json', 'r') as f:
            self.nodes = json.load(f)

        with open(self.kg_dir / 'graph_edges.json', 'r') as f:
            self.edges = json.load(f)

        with open(self.kg_dir / 'node_to_index.json', 'r') as f:
            self.node_to_idx = json.load(f)

        with open(self.kg_dir / 'technique_list.json', 'r') as f:
            self.technique_list = json.load(f)

        self.idx_to_node = {v: k for k, v in self.node_to_idx.items()}
        self.num_nodes = len(self.nodes)
        self.num_techniques = len(self.technique_list)

        # Node types
        self.node_types = ['tactic', 'technique', 'subtechnique', 'group', 'malware', 'tool', 'mitigation']
        self.type_to_idx = {t: i for i, t in enumerate(self.node_types)}

        # Edge types
        self.edge_types = ['USES', 'MITIGATES', 'SUBTECHNIQUE_OF', 'TACTIC_HAS_TECHNIQUE', 'TECHNIQUE_HAS_SUBTECHNIQUE', 'ATTRIBUTED_TO']
        self.edge_type_to_idx = {t: i for i, t in enumerate(self.edge_types)}

        # Build node type list
        self.node_type_list = []
        for node_id in sorted(self.node_to_idx.keys(), key=lambda x: self.node_to_idx[x]):
            node_type = self.nodes[node_id]['type']
            self.node_type_list.append(self.type_to_idx.get(node_type, 0))

    def pretrain_with_transe(self, device: torch.device):
        """Pre-train TransE embeddings and use for initialization."""
        if not self.use_transe_pretrain:
            return

        pretrainer = TransEPretrainer(
            edges=self.edges,
            node_to_idx=self.node_to_idx,
            edge_type_to_idx=self.edge_type_to_idx,
            embedding_dim=self.hidden_dim,
            device=device,
        )

        embeddings = pretrainer.train(epochs=self.transe_epochs)
        self.transe_embeddings = embeddings

        # Initialize node embeddings with TransE
        with torch.no_grad():
            self.node_embeddings.weight.copy_(embeddings)

        print(f"Initialized node embeddings with TransE pre-training")

    def _get_edge_tensors(self, device: torch.device):
        """Get edge index and type tensors (cached)."""
        if self._edge_index is None or self._edge_index.device != device:
            sources, targets, edge_types = [], [], []

            for edge in self.edges:
                src, tgt = edge['source'], edge['target']
                if src in self.node_to_idx and tgt in self.node_to_idx:
                    sources.append(self.node_to_idx[src])
                    targets.append(self.node_to_idx[tgt])
                    edge_types.append(self.edge_type_to_idx.get(edge['type'], 0))

            # Bidirectional
            self._edge_index = torch.tensor(
                [sources + targets, targets + sources],
                dtype=torch.long, device=device
            )
            self._edge_type = torch.tensor(
                edge_types + edge_types,
                dtype=torch.long, device=device
            )
            self._node_type = torch.tensor(
                self.node_type_list,
                dtype=torch.long, device=device
            )

        return self._edge_index, self._edge_type, self._node_type

    def forward(
        self,
        device: torch.device,
        text_emb: torch.Tensor = None,  # Optional text conditioning
    ) -> torch.Tensor:
        """
        Forward pass through advanced GNN.

        Args:
            device: Device to use
            text_emb: Optional text embeddings for dynamic conditioning [batch, hidden]

        Returns:
            Node embeddings [num_nodes, hidden_dim] or [batch, num_nodes, hidden_dim] if text_emb provided
        """
        # Get initial embeddings
        node_ids = torch.arange(self.num_nodes, device=device)
        x = self.node_embeddings(node_ids)

        # Add node type embeddings
        edge_index, edge_type, node_type = self._get_edge_tensors(device)
        type_emb = self.node_type_embedding(node_type)
        x = x + type_emb

        # R-GCN layers (edge-type aware)
        for rgcn in self.rgcn_layers:
            x = x + rgcn(x, edge_index, edge_type)  # Residual

        # HGT layers (heterogeneous attention)
        for hgt in self.hgt_layers:
            x = hgt(x, edge_index, node_type, edge_type)

        # Dynamic text-conditioned attention (if text provided)
        if text_emb is not None:
            x = self.text_conditioned_att(x, text_emb, edge_index)
            # x is now [batch, num_nodes, hidden]
            x = self.output_projection(x)
        else:
            x = self.output_projection(x)

        return x

    def get_technique_embeddings(
        self,
        node_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Extract technique embeddings from full node embeddings."""
        # Handle both batched and non-batched
        if node_embeddings.dim() == 3:  # [batch, nodes, hidden]
            batch_size = node_embeddings.size(0)
            technique_embs = []
            for tech_id in self.technique_list:
                if tech_id in self.node_to_idx:
                    idx = self.node_to_idx[tech_id]
                    technique_embs.append(node_embeddings[:, idx, :])
                else:
                    technique_embs.append(torch.zeros(batch_size, self.hidden_dim, device=node_embeddings.device))
            return torch.stack(technique_embs, dim=1)  # [batch, num_techniques, hidden]
        else:  # [nodes, hidden]
            technique_embs = []
            for tech_id in self.technique_list:
                if tech_id in self.node_to_idx:
                    idx = self.node_to_idx[tech_id]
                    technique_embs.append(node_embeddings[idx])
                else:
                    technique_embs.append(torch.zeros(self.hidden_dim, device=node_embeddings.device))
            return torch.stack(technique_embs, dim=0)  # [num_techniques, hidden]


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--kg_dir', type=str, default='../data/attack_framework')
    args = parser.parse_args()

    print("Testing Advanced Knowledge Graph GNN...")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    model = AdvancedKnowledgeGraphGNN(
        kg_dir=args.kg_dir,
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        use_transe_pretrain=True,
        transe_epochs=50,  # Reduced for testing
    )
    model = model.to(device)

    print(f"\nModel Statistics:")
    print(f"  - Nodes: {model.num_nodes}")
    print(f"  - Techniques: {model.num_techniques}")
    print(f"  - Node types: {model.node_types}")
    print(f"  - Edge types: {model.edge_types}")

    # Pre-train TransE
    print("\n" + "-" * 40)
    model.pretrain_with_transe(device)

    # Forward pass without text conditioning
    print("\n" + "-" * 40)
    print("Testing forward pass (no text conditioning)...")
    node_embeddings = model(device)
    print(f"  - Node embeddings shape: {node_embeddings.shape}")

    technique_embeddings = model.get_technique_embeddings(node_embeddings)
    print(f"  - Technique embeddings shape: {technique_embeddings.shape}")

    # Forward pass with text conditioning
    print("\nTesting forward pass (with text conditioning)...")
    batch_size = 4
    text_emb = torch.randn(batch_size, 512, device=device)
    node_embeddings_cond = model(device, text_emb=text_emb)
    print(f"  - Conditioned node embeddings shape: {node_embeddings_cond.shape}")

    technique_embeddings_cond = model.get_technique_embeddings(node_embeddings_cond)
    print(f"  - Conditioned technique embeddings shape: {technique_embeddings_cond.shape}")

    # Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameters:")
    print(f"  - Total: {total_params:,}")
    print(f"  - Trainable: {trainable_params:,}")

    print("\n" + "=" * 70)
    print("Advanced KG-GNN ready!")
