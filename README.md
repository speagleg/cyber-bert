# CyberBERT-GNN: MITRE ATT&CK Threat Classification

A hybrid BERT + Graph Neural Network architecture for multi-label threat classification using the MITRE ATT&CK framework.

## Architecture Overview

```
Security Log/Alert Text
        │
        ▼
┌───────────────────┐
│   SecBERT/BERT    │  ← Text encoder (768-dim embeddings)
│   Text Encoder    │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Text Projection  │  ← Project to shared dim (512)
│     (768→512)     │
└────────┬──────────┘
         │
         ├──────────────────┐
         │                  │
         ▼                  ▼
┌───────────────────┐  ┌───────────────────┐
│   Concatenate     │  │   MITRE ATT&CK    │
│   Text + Graph    │  │   Knowledge Graph │
│   Embeddings      │  │      (GNN)        │
└────────┬──────────┘  └───────────────────┘
         │
         ▼
┌───────────────────┐
│  Multi-Label      │  ← Classify to ATT&CK Techniques
│  Classification   │     (~600 techniques)
└───────────────────┘
```

## MITRE ATT&CK Knowledge Graph

**Nodes (~5,000):**
- 14 Tactics (Initial Access, Execution, Persistence, etc.)
- 200+ Techniques (T1566 Phishing, T1059 Command Line, etc.)
- 400+ Sub-techniques (T1566.001 Spearphishing Attachment)
- 140+ Threat Groups (APT29, Lazarus, etc.)
- 500+ Malware Families (Emotet, Cobalt Strike, etc.)

**Edge Types:**
- `TACTIC_HAS_TECHNIQUE`: Tactic → Technique
- `TECHNIQUE_HAS_SUBTECHNIQUE`: Technique → Sub-technique
- `GROUP_USES_TECHNIQUE`: Threat Group → Technique
- `MALWARE_USES_TECHNIQUE`: Malware → Technique
- `TECHNIQUE_MITIGATED_BY`: Technique → Mitigation
- `TECHNIQUE_DETECTED_BY`: Technique → Data Source

## Directory Structure

```
cyber-bert/
├── scripts/
│   ├── build_attack_kg.py       # Build MITRE ATT&CK knowledge graph
│   ├── secbert_gnn_model.py     # SecBERT + GNN hybrid model
│   ├── train_secbert_gnn.py     # DDP training script
│   └── process_security_logs.py # Dataset processor
├── data/
│   ├── attack_framework/        # MITRE ATT&CK STIX data
│   └── security_logs/           # Training data (logs, reports)
├── checkpoints/                 # Model checkpoints
└── logs/                        # Training logs
```

## Quick Start

### 1. Download MITRE ATT&CK Data

```bash
cd data/attack_framework
curl -o enterprise-attack.json \
  https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json
```

### 2. Build Knowledge Graph

```bash
python scripts/build_attack_kg.py \
  --stix_file data/attack_framework/enterprise-attack.json \
  --output_dir data/attack_framework
```

### 3. Train Model (Multi-GPU)

```bash
torchrun --nproc_per_node=8 scripts/train_secbert_gnn.py \
  --train_data data/security_logs/train.jsonl \
  --kg_dir data/attack_framework \
  --output_dir checkpoints/secbert_gnn_v1 \
  --batch_size 32 \
  --epochs 50 \
  --lr 2e-5
```

## Data Format

Training data should be JSONL with this structure:

```json
{
  "text": "PowerShell.exe -enc JABzAD0AJwAxADcAMgAuADEANgAuADAALgAxACcA...",
  "techniques": ["T1059.001", "T1027", "T1071.001"],
  "source": "sysmon_log"
}
```

## Public Datasets

| Dataset | Description | Use |
|---------|-------------|-----|
| [MITRE ATT&CK](https://attack.mitre.org/) | Official framework (STIX) | Knowledge graph |
| [Mordor](https://github.com/OTRF/mordor) | Labeled attack simulations | Training data |
| [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) | Network intrusion data | Evaluation |
| [SecRepo](https://www.secrepo.com/) | Security log samples | Training data |

## Model Performance Targets

| Metric | Target |
|--------|--------|
| Micro-F1 | > 0.75 |
| Macro-F1 | > 0.60 |
| Top-5 Accuracy | > 0.90 |
| Inference Time | < 50ms |

## Cross-Industry Applications

This architecture can be adapted for:

| Industry | Application | Knowledge Graph | BERT Model |
|----------|-------------|-----------------|------------|
| **Cybersecurity** | ATT&CK Classification | MITRE ATT&CK | SecBERT |
| Legal | Patent Classification | IPC/CPC Codes | Legal-BERT |
| Finance | SEC Risk Classification | GICS + Risk Taxonomy | FinBERT |
| Scientific | Paper Classification | Citation Network | SciBERT |

## License

MIT License
