# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CyberBERT-GNN is a hybrid BERT + Graph Neural Network architecture for multi-label threat classification using the MITRE ATT&CK framework. It combines text encoding from SecBERT/BERT with knowledge graph embeddings from a GNN to classify security logs/alerts into ~600 ATT&CK techniques.

## Architecture

The model pipeline:
1. **Text Encoder**: SecBERT/BERT encodes security log text → 768-dim embeddings
2. **Text Projection**: Projects to shared dimension (768 → 512)
3. **GNN Encoder**: Encodes MITRE ATT&CK knowledge graph (~5,000 nodes including tactics, techniques, groups, malware)
4. **Cross-Attention**: Text embeddings attend to technique embeddings from GNN
5. **Classification Head**: Multi-label classification to ATT&CK techniques

Key components:
- `AttackKnowledgeGraph`: Loads and manages the MITRE ATT&CK graph (nodes, edges, technique labels)
- `GNNEncoder`: SAGEConv + GATv2Conv layers for graph encoding
- `SecBERTGNNModel`: Main hybrid model combining BERT + GNN

## Build Commands

### 1. Download MITRE ATT&CK Data
```bash
mkdir -p data/attack_framework
curl -o data/attack_framework/enterprise-attack.json \
  https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json
```

### 2. Build Knowledge Graph
```bash
python scripts/build_attack_kg.py \
  --stix_file data/attack_framework/enterprise-attack.json \
  --output_dir data/attack_framework
```

### 3. Process Security Logs
```bash
python scripts/process_security_logs.py \
  --input data/security_logs/raw \
  --output_dir data/security_logs \
  --kg_dir data/attack_framework
```

### 4. Train Model (Multi-GPU with DDP)
```bash
torchrun --nproc_per_node=8 scripts/train_secbert_gnn.py \
  --train_data data/security_logs/train.jsonl \
  --val_data data/security_logs/val.jsonl \
  --kg_dir data/attack_framework \
  --output_dir checkpoints/secbert_gnn_v1 \
  --batch_size 32 \
  --epochs 50 \
  --lr 2e-5
```

### 5. Single-GPU Training
```bash
python scripts/train_secbert_gnn.py \
  --train_data data/security_logs/train.jsonl \
  --val_data data/security_logs/val.jsonl \
  --kg_dir data/attack_framework \
  --output_dir checkpoints/secbert_gnn_v1 \
  --batch_size 32
```

## Data Format

Training data is JSONL with structure:
```json
{"text": "security log text...", "techniques": ["T1059.001", "T1027"], "source": "sysmon"}
```

The log processor supports Sysmon, Zeek/Bro, Suricata, and Mordor dataset formats with automatic format detection and heuristic technique labeling.

## Key Dependencies

- PyTorch with DDP for distributed training
- PyTorch Geometric (torch_geometric) for GNN layers
- Transformers (HuggingFace) for BERT models
- scikit-learn for metrics

## Knowledge Graph Structure

**Node Types**: tactic, technique, subtechnique, group, malware, tool, mitigation

**Edge Types**: TACTIC_HAS_TECHNIQUE, TECHNIQUE_HAS_SUBTECHNIQUE, USES, MITIGATES, DETECTS, ATTRIBUTED_TO

Graph files generated in `data/attack_framework/`:
- `graph_nodes.json`: Node definitions
- `graph_edges.json`: Edge list
- `node_to_index.json`: Node ID to index mapping
- `technique_list.json`: Classification labels (~600 techniques)
