#!/usr/bin/env python3
"""
Advanced Training Script with Ablation Parameters.

All training enhancements are parameterized for systematic ablation studies:
1. Loss Functions: Focal, ASL, Label Smoothing
2. Sampling: SMOTE, Class-weighted
3. Threshold: Per-class, Dynamic
4. Regularization: Label correlation, Hierarchical consistency
5. Training Strategies: Two-stage, Curriculum learning

Usage:
    # Baseline (all enhancements OFF)
    python train_advanced_ablation.py --experiment baseline

    # Full model (all enhancements ON)
    python train_advanced_ablation.py --experiment full

    # Ablation: Test specific enhancement
    python train_advanced_ablation.py --experiment ablation_focal_loss \
        --use_focal_loss --focal_gamma 2.0

    # Custom configuration
    python train_advanced_ablation.py \
        --use_focal_loss --focal_gamma 2.0 --focal_gamma_max 5.0 \
        --use_label_smoothing --label_smoothing_alpha 0.1 \
        --use_per_class_thresholds \
        --use_hierarchical_loss --hierarchical_weight 0.3
"""

import argparse
import json
import os
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, OneCycleLR
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from secbert_advanced_gnn_model import SecBERTAdvancedGNNModel, create_advanced_model


# =============================================================================
# LOSS FUNCTIONS (Ablation: --use_focal_loss, --use_asl, --use_label_smoothing)
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.

    Ablation params:
        --use_focal_loss: Enable/disable
        --focal_gamma: Focusing parameter (default: 2.0)
        --focal_alpha: Class balancing weight (default: 0.25)
    """
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * labels + (1 - probs) * (1 - labels)
        focal_term = (1 - p_t) ** self.gamma

        # Alpha weighting
        alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)

        # Class weights if provided
        if self.class_weights is not None:
            weight = self.class_weights.unsqueeze(0).to(logits.device)
            focal_term = focal_term * weight

        loss = alpha_t * focal_term * bce
        return loss.mean()


class AdaptiveFocalLoss(nn.Module):
    """
    Adaptive Focal Loss with dynamic gamma.

    Ablation params:
        --focal_gamma: Initial gamma (default: 2.0)
        --focal_gamma_max: Maximum gamma (default: 5.0)
        --focal_warmup_epochs: Warmup before gamma increase (default: 5)
    """
    def __init__(
        self,
        gamma_init: float = 2.0,
        gamma_max: float = 5.0,
        warmup_epochs: int = 5,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.gamma_init = gamma_init
        self.gamma_max = gamma_max
        self.warmup_epochs = warmup_epochs
        self.current_gamma = gamma_init
        self.class_weights = class_weights

    def update_gamma(self, epoch: int, total_epochs: int):
        if epoch < self.warmup_epochs:
            self.current_gamma = self.gamma_init
        else:
            progress = (epoch - self.warmup_epochs) / max(1, total_epochs - self.warmup_epochs)
            self.current_gamma = self.gamma_init + progress * (self.gamma_max - self.gamma_init)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * labels + (1 - probs) * (1 - labels)
        focal_term = (1 - p_t) ** self.current_gamma

        if self.class_weights is not None:
            weight = self.class_weights.unsqueeze(0).to(logits.device)
            loss = weight * focal_term * bce
        else:
            loss = focal_term * bce

        return loss.mean()


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) - different gamma for positives vs negatives.

    Reference: "Asymmetric Loss For Multi-Label Classification" (ICCV 2021)

    Ablation params:
        --use_asl: Enable/disable
        --asl_gamma_neg: Gamma for negatives (default: 4.0)
        --asl_gamma_pos: Gamma for positives (default: 1.0)
        --asl_clip: Probability clipping for negatives (default: 0.05)
    """
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        disable_torch_grad_focal_loss: bool = False,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Probabilities
        probs = torch.sigmoid(logits)
        probs_pos = probs
        probs_neg = 1 - probs

        # Asymmetric clipping (for negatives only)
        if self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1)

        # Basic cross entropy
        los_pos = labels * torch.log(probs_pos.clamp(min=1e-8))
        los_neg = (1 - labels) * torch.log(probs_neg.clamp(min=1e-8))

        # Asymmetric focusing
        if self.disable_torch_grad_focal_loss:
            with torch.no_grad():
                asymmetric_w_pos = (1 - probs_pos) ** self.gamma_pos
                asymmetric_w_neg = probs ** self.gamma_neg
        else:
            asymmetric_w_pos = (1 - probs_pos) ** self.gamma_pos
            asymmetric_w_neg = probs ** self.gamma_neg

        los_pos = los_pos * asymmetric_w_pos
        los_neg = los_neg * asymmetric_w_neg

        loss = -los_pos - los_neg
        return loss.mean()


class LabelSmoothingBCE(nn.Module):
    """
    BCE with label smoothing.

    Ablation params:
        --use_label_smoothing: Enable/disable
        --label_smoothing_alpha: Smoothing factor (default: 0.1)
    """
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Smooth labels: 1 -> 1-smoothing, 0 -> smoothing
        smooth_labels = labels * (1 - self.smoothing) + (1 - labels) * self.smoothing
        return F.binary_cross_entropy_with_logits(logits, smooth_labels)


# =============================================================================
# REGULARIZATION LOSSES (Ablation: --use_hierarchical_loss, --use_label_correlation)
# =============================================================================

class HierarchicalConsistencyLoss(nn.Module):
    """
    Enforces tactic → technique consistency.

    If a technique is predicted, its parent tactic should also be predicted.

    Ablation params:
        --use_hierarchical_loss: Enable/disable
        --hierarchical_weight: Loss weight (default: 0.3)
    """
    def __init__(self, technique_to_tactic: Dict[str, str], technique_list: List[str]):
        super().__init__()
        # Build mapping from technique index to tactic indices
        self.technique_to_tactic_idx = {}
        tactic_set = set(technique_to_tactic.values())
        self.tactic_to_idx = {t: i for i, t in enumerate(sorted(tactic_set))}

        for tech_idx, tech in enumerate(technique_list):
            if tech in technique_to_tactic:
                tactic = technique_to_tactic[tech]
                self.technique_to_tactic_idx[tech_idx] = self.tactic_to_idx[tactic]

    def forward(self, logits: torch.Tensor, tactic_logits: torch.Tensor = None) -> torch.Tensor:
        if tactic_logits is None:
            return torch.tensor(0.0, device=logits.device)

        # For each predicted technique, check if parent tactic is also predicted
        tech_probs = torch.sigmoid(logits)
        tactic_probs = torch.sigmoid(tactic_logits)

        consistency_loss = 0.0
        count = 0

        for tech_idx, tactic_idx in self.technique_to_tactic_idx.items():
            # Technique prob should not exceed tactic prob
            # Loss when P(technique) > P(tactic)
            violation = F.relu(tech_probs[:, tech_idx] - tactic_probs[:, tactic_idx])
            consistency_loss += violation.mean()
            count += 1

        return consistency_loss / max(count, 1)


class LabelCorrelationLoss(nn.Module):
    """
    Enforces label co-occurrence patterns from training data.

    Ablation params:
        --use_label_correlation: Enable/disable
        --label_correlation_weight: Loss weight (default: 0.1)
    """
    def __init__(self, co_occurrence_matrix: torch.Tensor, threshold: float = 0.5):
        super().__init__()
        # co_occurrence_matrix[i,j] = P(label_j | label_i) from training data
        self.register_buffer('co_occurrence', co_occurrence_matrix)
        self.threshold = threshold

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # For each positive label, check if correlated labels are also predicted
        batch_size, num_labels = probs.shape

        # Expected co-occurrence based on predictions
        expected = torch.matmul(probs, self.co_occurrence)  # [B, L]

        # Penalize when expected correlation is high but prediction is low
        loss = F.mse_loss(probs, expected.clamp(0, 1))

        return loss


# =============================================================================
# THRESHOLD OPTIMIZATION (Ablation: --use_per_class_thresholds, --use_dynamic_threshold)
# =============================================================================

def find_optimal_thresholds(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    method: str = 'per_class',  # 'per_class', 'global', 'f1_optimized'
) -> np.ndarray:
    """
    Find optimal classification thresholds.

    Ablation params:
        --use_per_class_thresholds: Enable per-class threshold optimization
        --threshold_method: 'per_class', 'global', 'f1_optimized'
    """
    if method == 'global':
        # Single global threshold
        best_f1 = 0
        best_threshold = 0.5
        for t in np.arange(0.01, 0.99, 0.01):
            pred_binary = (predictions >= t).astype(int)
            f1 = f1_score(labels, pred_binary, average='micro', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
        return np.full(num_classes, best_threshold)

    elif method == 'per_class':
        # Per-class threshold optimization
        thresholds = np.zeros(num_classes)
        for c in range(num_classes):
            best_f1 = 0
            best_t = 0.5
            for t in np.arange(0.01, 0.99, 0.02):
                pred_c = (predictions[:, c] >= t).astype(int)
                label_c = labels[:, c].astype(int)
                if label_c.sum() > 0:  # Only if class has positives
                    f1 = f1_score(label_c, pred_c, zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_t = t
            thresholds[c] = best_t
        return thresholds

    elif method == 'f1_optimized':
        # Optimize for micro F1 using gradient-free search
        from scipy.optimize import minimize

        def neg_micro_f1(thresholds):
            pred_binary = (predictions >= thresholds).astype(int)
            return -f1_score(labels, pred_binary, average='micro', zero_division=0)

        # Start from per-class optimal
        x0 = find_optimal_thresholds(predictions, labels, num_classes, 'per_class')
        result = minimize(neg_micro_f1, x0, method='L-BFGS-B',
                         bounds=[(0.01, 0.99)] * num_classes)
        return result.x

    return np.full(num_classes, 0.5)


# =============================================================================
# SAMPLING STRATEGIES (Ablation: --use_weighted_sampling, --use_smote)
# =============================================================================

def compute_sample_weights(
    dataset: Dataset,
    technique_list: List[str],
    method: str = 'inverse_freq',  # 'inverse_freq', 'sqrt_inverse', 'effective_num'
) -> torch.Tensor:
    """
    Compute per-sample weights for weighted random sampling.

    Ablation params:
        --use_weighted_sampling: Enable/disable
        --sampling_method: 'inverse_freq', 'sqrt_inverse', 'effective_num'
    """
    # Count technique occurrences
    technique_counts = Counter()
    for sample in dataset.samples:
        for tech in sample['techniques']:
            technique_counts[tech] += 1

    # Compute class weights
    total_samples = len(dataset)
    num_classes = len(technique_list)

    if method == 'inverse_freq':
        class_weights = {
            tech: total_samples / max(technique_counts.get(tech, 1), 1)
            for tech in technique_list
        }
    elif method == 'sqrt_inverse':
        class_weights = {
            tech: np.sqrt(total_samples / max(technique_counts.get(tech, 1), 1))
            for tech in technique_list
        }
    elif method == 'effective_num':
        beta = 0.9999
        class_weights = {
            tech: (1 - beta) / (1 - beta ** max(technique_counts.get(tech, 1), 1))
            for tech in technique_list
        }
    else:
        class_weights = {tech: 1.0 for tech in technique_list}

    # Compute per-sample weight (max weight of its labels)
    sample_weights = []
    for sample in dataset.samples:
        weight = max(class_weights.get(tech, 1.0) for tech in sample['techniques'])
        sample_weights.append(weight)

    return torch.tensor(sample_weights, dtype=torch.float32)


def compute_class_weights(
    technique_counts: Dict[str, int],
    technique_list: List[str],
    method: str = 'inverse_freq',
) -> torch.Tensor:
    """
    Compute class weights for loss function.

    Ablation params:
        --class_weight_method: 'inverse_freq', 'sqrt_inverse', 'effective_num'
    """
    total = sum(technique_counts.values())
    weights = []

    for tech in technique_list:
        count = technique_counts.get(tech, 1)
        if method == 'inverse_freq':
            w = total / (len(technique_list) * max(count, 1))
        elif method == 'sqrt_inverse':
            w = np.sqrt(total / (len(technique_list) * max(count, 1)))
        elif method == 'effective_num':
            beta = 0.9999
            w = (1 - beta) / (1 - beta ** max(count, 1))
        else:
            w = 1.0
        weights.append(w)

    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.mean()  # Normalize to mean=1
    return weights


# =============================================================================
# CURRICULUM LEARNING (Ablation: --use_curriculum)
# =============================================================================

class CurriculumSampler:
    """
    Curriculum learning: start with easy samples, gradually add harder ones.

    Ablation params:
        --use_curriculum: Enable/disable
        --curriculum_warmup_epochs: Epochs before full difficulty (default: 10)
    """
    def __init__(
        self,
        difficulties: np.ndarray,  # Per-sample difficulty scores
        warmup_epochs: int = 10,
    ):
        self.difficulties = difficulties
        self.warmup_epochs = warmup_epochs
        self.sorted_indices = np.argsort(difficulties)  # Easy to hard

    def get_indices(self, epoch: int, total_epochs: int) -> np.ndarray:
        if epoch >= self.warmup_epochs:
            return np.arange(len(self.difficulties))  # All samples

        # Gradually increase proportion of hard samples
        progress = epoch / self.warmup_epochs
        num_samples = int(len(self.difficulties) * (0.5 + 0.5 * progress))
        return self.sorted_indices[:num_samples]


def compute_sample_difficulty(
    dataset: Dataset,
    technique_counts: Dict[str, int],
) -> np.ndarray:
    """
    Compute difficulty score for each sample.

    Difficulty = inverse of average frequency of its labels
    (rare labels = harder samples)
    """
    difficulties = []
    for sample in dataset.samples:
        avg_freq = np.mean([
            technique_counts.get(tech, 1)
            for tech in sample['techniques']
        ])
        difficulty = 1.0 / max(avg_freq, 1)
        difficulties.append(difficulty)

    return np.array(difficulties)


# =============================================================================
# DATASET
# =============================================================================

class SecurityLogDataset(Dataset):
    """Dataset for security logs with ATT&CK technique labels."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        technique_list: List[str],
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.technique_list = technique_list
        self.technique_to_idx = {t: i for i, t in enumerate(technique_list)}
        self.max_length = max_length
        self.samples = []

        self._load_data(data_path)

    def _load_data(self, data_path: str):
        with open(data_path, 'r') as f:
            for line in f:
                sample = json.loads(line.strip())
                text = sample.get('text', '')
                techniques = sample.get('techniques', [])

                valid_techniques = [
                    t for t in techniques
                    if t in self.technique_to_idx
                ]

                if valid_techniques and text:
                    self.samples.append({
                        'text': text,
                        'techniques': valid_techniques,
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        encoding = self.tokenizer(
            sample['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        labels = torch.zeros(len(self.technique_list))
        for tech in sample['techniques']:
            if tech in self.technique_to_idx:
                labels[self.technique_to_idx[tech]] = 1.0

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels,
        }


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray = None,
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    if thresholds is None:
        thresholds = np.full(predictions.shape[1], 0.5)

    binary_preds = (predictions >= thresholds).astype(int)

    micro_f1 = f1_score(labels, binary_preds, average='micro', zero_division=0)
    macro_f1 = f1_score(labels, binary_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(labels, binary_preds, average='weighted', zero_division=0)
    micro_precision = precision_score(labels, binary_preds, average='micro', zero_division=0)
    micro_recall = recall_score(labels, binary_preds, average='micro', zero_division=0)

    return {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'precision': micro_precision,
        'recall': micro_recall,
    }


# =============================================================================
# TRAINING
# =============================================================================

def setup_distributed():
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        rank = 0
        local_rank = 0
        world_size = 1

    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,  # Added scheduler parameter
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    rank: int,
    args,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=rank != 0)

    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        with autocast(device_type='cuda', dtype=torch.float16, enabled=args.fp16):
            logits = model(
                input_ids,
                attention_mask,
                use_text_conditioning=args.use_text_conditioning,
            )
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        # Step scheduler after each batch (OneCycleLR requires per-batch stepping)
        scheduler.step()

        total_loss += loss.item()

        with torch.no_grad():
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{current_lr:.2e}'})

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    metrics = compute_metrics(all_preds, all_labels)
    metrics['loss'] = total_loss / len(dataloader)

    return metrics, all_preds, all_labels


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    rank: int,
    args,
    thresholds: np.ndarray = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate the model."""
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc="Evaluating", disable=rank != 0)

    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with autocast(device_type='cuda', dtype=torch.float16, enabled=args.fp16):
            logits = model(
                input_ids,
                attention_mask,
                use_text_conditioning=args.use_text_conditioning,
            )
            loss = criterion(logits, labels)

        total_loss += loss.item()

        preds = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Use provided thresholds or default
    metrics = compute_metrics(all_preds, all_labels, thresholds)
    metrics['loss'] = total_loss / len(dataloader)

    return metrics, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='Advanced Training with Ablation')

    # Data arguments
    parser.add_argument('--train_data', required=True)
    parser.add_argument('--val_data', required=True)
    parser.add_argument('--kg_dir', required=True)
    parser.add_argument('--output_dir', required=True)

    # Model arguments
    parser.add_argument('--bert_model', default='jackaduma/SecBERT')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--gnn_layers', type=int, default=4)
    parser.add_argument('--gnn_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp16', action='store_true', default=False,
                       help='Enable mixed precision training (requires AMP-compatible model)')

    # Experiment name
    parser.add_argument('--experiment', type=str, default='custom',
                       help='Experiment name for logging')

    # ==========================================================================
    # ABLATION PARAMETERS - Loss Functions
    # ==========================================================================
    parser.add_argument('--use_focal_loss', action='store_true',
                       help='Enable Focal Loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Focal loss gamma (focusing parameter)')
    parser.add_argument('--focal_gamma_max', type=float, default=5.0,
                       help='Max gamma for adaptive focal loss')
    parser.add_argument('--focal_warmup_epochs', type=int, default=5,
                       help='Warmup epochs before gamma increases')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                       help='Focal loss alpha (class balance)')

    parser.add_argument('--use_asl', action='store_true',
                       help='Enable Asymmetric Loss')
    parser.add_argument('--asl_gamma_neg', type=float, default=4.0,
                       help='ASL gamma for negatives')
    parser.add_argument('--asl_gamma_pos', type=float, default=1.0,
                       help='ASL gamma for positives')
    parser.add_argument('--asl_clip', type=float, default=0.05,
                       help='ASL probability clipping')

    parser.add_argument('--use_label_smoothing', action='store_true',
                       help='Enable label smoothing')
    parser.add_argument('--label_smoothing_alpha', type=float, default=0.1,
                       help='Label smoothing factor')

    # ==========================================================================
    # ABLATION PARAMETERS - Regularization
    # ==========================================================================
    parser.add_argument('--use_hierarchical_loss', action='store_true',
                       help='Enable hierarchical consistency loss')
    parser.add_argument('--hierarchical_weight', type=float, default=0.3,
                       help='Weight for hierarchical loss')

    parser.add_argument('--use_label_correlation', action='store_true',
                       help='Enable label correlation loss')
    parser.add_argument('--label_correlation_weight', type=float, default=0.1,
                       help='Weight for label correlation loss')

    # ==========================================================================
    # ABLATION PARAMETERS - Thresholds
    # ==========================================================================
    parser.add_argument('--use_per_class_thresholds', action='store_true',
                       help='Enable per-class threshold optimization')
    parser.add_argument('--use_dynamic_threshold', action='store_true',
                       help='Enable dynamic threshold search')
    parser.add_argument('--threshold_method', type=str, default='per_class',
                       choices=['global', 'per_class', 'f1_optimized'])

    # ==========================================================================
    # ABLATION PARAMETERS - Sampling
    # ==========================================================================
    parser.add_argument('--use_weighted_sampling', action='store_true',
                       help='Enable weighted random sampling')
    parser.add_argument('--sampling_method', type=str, default='inverse_freq',
                       choices=['inverse_freq', 'sqrt_inverse', 'effective_num'])
    parser.add_argument('--use_class_weights', action='store_true',
                       help='Enable class weights in loss function')
    parser.add_argument('--class_weight_method', type=str, default='inverse_freq',
                       choices=['inverse_freq', 'sqrt_inverse', 'effective_num'])

    # ==========================================================================
    # ABLATION PARAMETERS - Curriculum Learning
    # ==========================================================================
    parser.add_argument('--use_curriculum', action='store_true',
                       help='Enable curriculum learning')
    parser.add_argument('--curriculum_warmup_epochs', type=int, default=10,
                       help='Epochs before full difficulty')

    # ==========================================================================
    # ABLATION PARAMETERS - GNN Features
    # ==========================================================================
    parser.add_argument('--use_transe_pretrain', action='store_true', default=True,
                       help='Enable TransE pretraining')
    parser.add_argument('--transe_epochs', type=int, default=100,
                       help='TransE pretraining epochs')
    parser.add_argument('--use_text_conditioning', action='store_true', default=True,
                       help='Enable text-conditioned GNN attention')

    args = parser.parse_args()

    # Setup
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save config
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(vars(args), f, indent=2)

    # Load tokenizer
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Experiment: {args.experiment}")
        print(f"{'='*60}")
        print(f"Loading tokenizer: {args.bert_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    # Create model
    if rank == 0:
        print(f"Creating model...")
    model = create_advanced_model(
        kg_dir=args.kg_dir,
        bert_model=args.bert_model,
        hidden_dim=args.hidden_dim,
        gnn_layers=args.gnn_layers,
        gnn_heads=args.gnn_heads,
        dropout=args.dropout,
        use_transe_pretrain=args.use_transe_pretrain,
        transe_epochs=args.transe_epochs,
        device=device,
    )

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        base_model = model.module
    else:
        base_model = model

    # Create datasets
    if rank == 0:
        print(f"Loading datasets...")
    train_dataset = SecurityLogDataset(
        args.train_data, tokenizer, base_model.technique_list, args.max_length
    )
    val_dataset = SecurityLogDataset(
        args.val_data, tokenizer, base_model.technique_list, args.max_length
    )

    if rank == 0:
        print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        print(f"  Techniques: {len(base_model.technique_list)}")

    # Compute technique counts for class weights
    technique_counts = Counter()
    for sample in train_dataset.samples:
        for tech in sample['techniques']:
            technique_counts[tech] += 1

    # Setup class weights
    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(
            technique_counts, base_model.technique_list, args.class_weight_method
        ).to(device)
        if rank == 0:
            print(f"  Using class weights ({args.class_weight_method})")

    # Setup loss function
    if args.use_asl:
        criterion = AsymmetricLoss(
            gamma_neg=args.asl_gamma_neg,
            gamma_pos=args.asl_gamma_pos,
            clip=args.asl_clip,
        )
        if rank == 0:
            print(f"  Loss: ASL (γ_neg={args.asl_gamma_neg}, γ_pos={args.asl_gamma_pos})")
    elif args.use_focal_loss:
        criterion = AdaptiveFocalLoss(
            gamma_init=args.focal_gamma,
            gamma_max=args.focal_gamma_max,
            warmup_epochs=args.focal_warmup_epochs,
            class_weights=class_weights,
        )
        if rank == 0:
            print(f"  Loss: Adaptive Focal (γ={args.focal_gamma}→{args.focal_gamma_max})")
    elif args.use_label_smoothing:
        criterion = LabelSmoothingBCE(smoothing=args.label_smoothing_alpha)
        if rank == 0:
            print(f"  Loss: Label Smoothing (α={args.label_smoothing_alpha})")
    else:
        criterion = nn.BCEWithLogitsLoss()
        if rank == 0:
            print(f"  Loss: BCE")

    # Setup sampler
    if args.use_weighted_sampling and world_size == 1:
        sample_weights = compute_sample_weights(
            train_dataset, base_model.technique_list, args.sampling_method
        )
        sampler = WeightedRandomSampler(sample_weights, len(train_dataset))
        shuffle = False
        if rank == 0:
            print(f"  Sampling: Weighted ({args.sampling_method})")
    elif world_size > 1:
        sampler = DistributedSampler(train_dataset, shuffle=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    # Dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        shuffle=shuffle, num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)
    scheduler = OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=warmup_steps/total_steps, anneal_strategy='cos'
    )

    scaler = GradScaler()

    # Print ablation config
    if rank == 0:
        print(f"\nAblation Configuration:")
        print(f"  Focal Loss: {args.use_focal_loss}")
        print(f"  ASL: {args.use_asl}")
        print(f"  Label Smoothing: {args.use_label_smoothing}")
        print(f"  Class Weights: {args.use_class_weights}")
        print(f"  Weighted Sampling: {args.use_weighted_sampling}")
        print(f"  Per-class Thresholds: {args.use_per_class_thresholds}")
        print(f"  TransE Pretrain: {args.use_transe_pretrain}")
        print(f"  Text Conditioning: {args.use_text_conditioning}")
        print(f"\nStarting training...")

    # Training loop
    best_f1 = 0.0
    optimal_thresholds = None

    for epoch in range(1, args.epochs + 1):
        if world_size > 1:
            sampler.set_epoch(epoch)

        # Update adaptive focal loss gamma
        if hasattr(criterion, 'update_gamma'):
            criterion.update_gamma(epoch, args.epochs)
            if rank == 0:
                print(f"\n  Focal γ: {criterion.current_gamma:.2f}")

        # Train (scheduler is now stepped inside train_epoch per batch)
        train_metrics, train_preds, train_labels = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, scaler,
            device, epoch, rank, args
        )

        # Optimize thresholds on training data
        if args.use_per_class_thresholds or args.use_dynamic_threshold:
            optimal_thresholds = find_optimal_thresholds(
                train_preds, train_labels,
                len(base_model.technique_list),
                method=args.threshold_method
            )

        # Evaluate
        val_metrics, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device, rank, args, optimal_thresholds
        )

        # Log
        if rank == 0:
            print(f"\nEpoch {epoch}/{args.epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Micro F1: {train_metrics['micro_f1']:.4f}, "
                  f"Macro F1: {train_metrics['macro_f1']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Micro F1: {val_metrics['micro_f1']:.4f}, "
                  f"Macro F1: {val_metrics['macro_f1']:.4f}")
            print(f"  Val   - Precision: {val_metrics['precision']:.4f}, "
                  f"Recall: {val_metrics['recall']:.4f}")

            # Log to file
            log_entry = {
                'epoch': epoch,
                'experiment': args.experiment,
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()},
                'lr': optimizer.param_groups[0]['lr'],
            }
            if hasattr(criterion, 'current_gamma'):
                log_entry['focal_gamma'] = criterion.current_gamma

            with open(output_dir / 'metrics.jsonl', 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

            # Save best
            combined_f1 = (val_metrics['micro_f1'] + val_metrics['macro_f1']) / 2
            if combined_f1 > best_f1:
                best_f1 = combined_f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_f1': best_f1,
                    'thresholds': optimal_thresholds,
                    'config': vars(args),
                }, output_dir / 'best_model.pt')
                print(f"  ✓ New best (F1: {best_f1:.4f})")

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Best combined F1: {best_f1:.4f}")
        print(f"{'='*60}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
