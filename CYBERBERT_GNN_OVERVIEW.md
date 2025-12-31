# CyberBERT-GNN: Hybrid Transformer-Graph Neural Network for Threat Intelligence

## Executive Summary

CyberBERT-GNN is a state-of-the-art deep learning system that automatically classifies security logs and threat reports into MITRE ATT&CK framework techniques. By combining domain-specific language models with knowledge graph neural networks, the system achieves **86.72% Micro F1** across 596 attack technique categories—enabling security teams to rapidly triage alerts and map threats to standardized frameworks.

---

## Problem Statement

Security Operations Centers (SOCs) face a critical challenge: analysts must manually review thousands of security alerts daily and map them to the MITRE ATT&CK framework for threat intelligence. This process is:

- **Time-consuming**: Manual classification takes 15-30 minutes per incident
- **Inconsistent**: Different analysts may classify the same event differently
- **Incomplete**: The ATT&CK framework has 596+ techniques—no human can master all of them
- **Bottlenecked**: Alert fatigue leads to missed threats

---

## Solution Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CyberBERT-GNN Architecture                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐ │
│  │  Security Log   │    │    SecBERT      │    │   Text Embedding        │ │
│  │  "powershell    │───▶│   Transformer   │───▶│   768-dim → 512-dim     │ │
│  │   -enc base64"  │    │   (12 layers)   │    │                         │ │
│  └─────────────────┘    └─────────────────┘    └───────────┬─────────────┘ │
│                                                            │               │
│                                                            ▼               │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐ │
│  │  MITRE ATT&CK   │    │  Advanced       │    │   Cross-Attention       │ │
│  │  Knowledge      │───▶│  KG-GNN         │───▶│   Fusion Layer          │ │
│  │  Graph (5K+     │    │  (R-GCN + HGT)  │    │   (8 heads)             │ │
│  │  nodes)         │    └─────────────────┘    └───────────┬─────────────┘ │
│  └─────────────────┘                                       │               │
│                                                            ▼               │
│                                              ┌─────────────────────────┐   │
│                                              │  Multi-Label Classifier │   │
│                                              │  596 ATT&CK Techniques  │   │
│                                              │  Per-Class Thresholds   │   │
│                                              └─────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Technical Components

### 1. SecBERT Text Encoder (Transformer)

**Model**: `jackaduma/SecBERT` - A BERT model pre-trained on cybersecurity corpora including:
- Security blogs and threat reports
- CVE descriptions
- Malware analysis documents
- Incident response documentation

**Architecture**:
| Parameter | Value |
|-----------|-------|
| Hidden Size | 768 |
| Attention Heads | 12 |
| Transformer Layers | 12 |
| Max Sequence Length | 512 tokens |
| Parameters | 110M |

**Why SecBERT over BERT?**
- Understands security-specific terminology (IOCs, TTPs, CVEs)
- Better tokenization of technical strings (IP addresses, file hashes, registry keys)
- Pre-trained on domain-relevant data

---

### 2. MITRE ATT&CK Knowledge Graph

The knowledge graph encodes the complete MITRE ATT&CK framework as a heterogeneous graph:

**Node Types** (5,000+ nodes):
| Type | Count | Description |
|------|-------|-------------|
| Technique | 596 | Attack techniques (e.g., T1059.001 PowerShell) |
| Tactic | 14 | Kill chain phases (e.g., Execution, Persistence) |
| Group | 130+ | Threat actor groups (e.g., APT29, Lazarus) |
| Malware | 500+ | Malware families (e.g., Cobalt Strike, Emotet) |
| Tool | 70+ | Adversary tools (e.g., Mimikatz, PsExec) |
| Mitigation | 40+ | Defensive countermeasures |

**Edge Types** (Relational Semantics):
| Relation | Description |
|----------|-------------|
| `USES` | Threat actor/malware uses technique |
| `MITIGATES` | Defense counters technique |
| `SUBTECHNIQUE_OF` | Hierarchical refinement |
| `TACTIC_HAS_TECHNIQUE` | Category membership |
| `ATTRIBUTED_TO` | Attribution relationships |
| `DETECTS` | Detection capability |

---

### 3. Advanced Knowledge Graph GNN

Four key architectural innovations:

#### a) R-GCN (Relational Graph Convolutional Network)
```
Different edge types → Different weight matrices

W_uses ≠ W_mitigates ≠ W_subtechnique

Learns type-specific message passing
```

#### b) HGT (Heterogeneous Graph Transformer)
```
Node-type and edge-type specific attention

Attention(technique, group) ≠ Attention(technique, malware)

Enables fine-grained relationship modeling
```

#### c) TransE Pre-training
```
Knowledge graph embedding objective:
    head + relation ≈ tail

Pre-trains node embeddings before task training
Captures global graph structure
```

#### d) Dynamic Text-Conditioned Attention
```
Input text modulates graph attention weights

"powershell -enc" → Emphasize scripting techniques
"reg add HKLM"   → Emphasize persistence techniques

Enables input-specific knowledge retrieval
```

**GNN Configuration**:
| Parameter | Value |
|-----------|-------|
| Hidden Dimension | 512 |
| GNN Layers | 4 |
| Attention Heads | 8 |
| Dropout | 0.1 |

---

### 4. Cross-Attention Fusion

Fuses text embeddings with technique embeddings:

```
Query: Text embedding (what the log describes)
Keys:  Technique embeddings (what attacks exist)
Values: Technique embeddings

Output: Text-aware technique relevance scores
```

This allows the model to:
- Attend to relevant techniques based on text content
- Ignore irrelevant techniques (most of 596 classes)
- Learn semantic similarity between logs and attack patterns

---

### 5. Multi-Label Classification Head

**Challenge**: 596 techniques with extreme class imbalance (some techniques have 10,000+ examples, others have <10)

**Solutions Implemented**:

| Component | Description |
|-----------|-------------|
| **Focal Loss** | Down-weights easy examples, focuses on hard ones |
| **Hierarchical Loss** | Exploits tactic→technique relationships |
| **Per-Class Thresholds** | Optimized threshold per technique (not fixed 0.5) |
| **Effective Number Weighting** | Class weights based on effective sample count |

---

## Training Pipeline

### Data Sources

| Dataset | Samples | Description |
|---------|---------|-------------|
| Security Logs | 7,000 | Sysmon, Zeek, Suricata logs |
| Generated (LLM) | 29,000 | Synthetic examples via LLM |
| Augmented | 73,000 | Data augmentation (synonyms, noise) |
| SMOTE | 43,000 | Oversampled minority classes |

### Training Configuration

```yaml
Optimizer: AdamW
Learning Rate: 3e-5 (with warmup)
Scheduler: OneCycleLR
Batch Size: 12
Epochs: 30
Mixed Precision: FP16
Hardware: NVIDIA A100 (40GB)
```

### Loss Function

```python
Loss = FocalLoss(γ=2.0→5.0) + 0.3 × HierarchicalLoss
```

- Adaptive focal gamma increases during training
- Hierarchical loss ensures tactic consistency

---

## Performance Metrics

### Best Model Results

| Metric | Score |
|--------|-------|
| **Micro F1** | **86.72%** |
| **Macro F1** | 72.4% |
| **Precision** | 85.1% |
| **Recall** | 88.4% |

### Comparison to Baselines

| Model | Micro F1 | Notes |
|-------|----------|-------|
| BERT + Linear | 71.2% | No graph knowledge |
| SecBERT + Linear | 74.8% | Domain-specific LM |
| SecBERT + Basic GNN | 79.3% | Simple GCN |
| **CyberBERT-GNN** | **86.72%** | Full architecture |

### Per-Tactic Performance

| Tactic | F1 Score |
|--------|----------|
| Execution | 91.2% |
| Persistence | 88.7% |
| Defense Evasion | 84.3% |
| Command & Control | 89.1% |
| Discovery | 82.5% |

---

## Key Innovations

### 1. Knowledge-Grounded Classification
Unlike pure text classifiers, CyberBERT-GNN reasons over the ATT&CK knowledge graph, understanding:
- Which techniques are related
- Which threat actors use which techniques
- How techniques relate to tactics

### 2. Per-Class Threshold Optimization
Instead of a fixed 0.5 threshold:
```
T1059.001 (PowerShell): threshold = 0.32
T1053.005 (Scheduled Task): threshold = 0.41
T1078 (Valid Accounts): threshold = 0.28
```
Optimized per-class based on precision-recall trade-offs.

### 3. Hierarchical Consistency
The hierarchical loss ensures:
- If T1059.001 (PowerShell) is predicted
- Then T1059 (Command Line) should also be predicted
- And TA0002 (Execution) should be the tactic

### 4. Extreme Multi-Label Handling
596 classes with long-tail distribution handled via:
- Focal loss for hard example mining
- Effective number weighting for class balance
- Data augmentation for minority classes

---

## Deployment Options

### API Service
```python
from cyberbert_gnn import ThreatClassifier

classifier = ThreatClassifier.load("model.pt")
predictions = classifier.predict(
    "powershell.exe -enc JABzAD0AJw..."
)
# Returns: [("T1059.001", 0.94), ("T1027", 0.87), ...]
```

### Batch Processing
```bash
python classify_logs.py \
    --input logs.jsonl \
    --output predictions.jsonl \
    --model checkpoints/best_model.pt
```

### SIEM Integration
- Splunk app via HTTP Event Collector
- Elastic integration via Logstash
- Real-time streaming via Kafka

---

## Use Cases

### 1. Alert Triage
Automatically classify incoming alerts → Reduce analyst workload by 60%

### 2. Threat Hunting
Query logs by technique → "Show me all T1059.001 activity this week"

### 3. Detection Engineering
Validate detection rules → Does this rule catch the intended technique?

### 4. Incident Response
Map observed activity to ATT&CK → Generate standardized reports

### 5. Threat Intelligence
Enrich IOCs with technique mappings → Improve CTI feeds

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Deep Learning | PyTorch 2.x |
| Transformers | HuggingFace Transformers |
| Graph Neural Networks | PyTorch Geometric |
| Training | Distributed Data Parallel (DDP) |
| Hardware | NVIDIA A100 / H100 |
| Serving | FastAPI / TorchServe |

---

## Future Roadmap

1. **Multi-lingual Support**: Extend to non-English threat reports
2. **Temporal Modeling**: Sequence-aware classification for kill chain detection
3. **Explainability**: Attention visualization for analyst trust
4. **Continuous Learning**: Online updates with new ATT&CK versions
5. **Edge Deployment**: Optimized models for endpoint agents

---

## Contact

For collaboration, licensing, or integration inquiries:

**Gordon Speagle**
gspeagle@gmail.com

---

*Built with SecBERT, PyTorch Geometric, and the MITRE ATT&CK Framework*
