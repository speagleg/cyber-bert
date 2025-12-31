#!/usr/bin/env python3
"""
Training Script for SecBERT + Enhanced KG-GNN.

This script trains the enhanced hybrid model with:
1. Semantic node embeddings from ATT&CK descriptions
2. Hierarchical attention (Tactic → Technique → Sub-technique)
3. Ontology-aware message passing
4. Adaptive Focal Loss for class imbalance
5. Hierarchical auxiliary losses

Usage:
    python train_enhanced.py \
        --train_data data/security_logs/train.jsonl \
        --val_data data/security_logs/val.jsonl \
        --kg_dir data/attack_framework \
        --output_dir checkpoints/enhanced_v1 \
        --batch_size 16 \
        --epochs 50 \
        --lr 2e-5 \
        --use_focal_loss
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from secbert_enhanced_gnn import SecBERTEnhancedGNN, create_enhanced_model
from enhanced_kg_gnn import EnhancedKnowledgeGraphGNN

# Import adaptive focal loss
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'fixes'))
try:
    from adaptive_focal_loss import HybridAdaptiveFocalLoss, compute_class_weights
    FOCAL_LOSS_AVAILABLE = True
except ImportError:
    FOCAL_LOSS_AVAILABLE = False
    print("Warning: Adaptive focal loss not available, using BCE")


def load_technique_counts(data_dir: str) -> dict:
    """Load technique counts from stats.json for class weighting."""
    stats_path = Path(data_dir).parent / 'stats.json'
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            stats = json.load(f)
            return stats.get('technique_counts', {})
    return {}


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
        """Load and preprocess data from JSONL file."""
        with open(data_path, 'r') as f:
            for line in f:
                sample = json.loads(line.strip())
                text = sample.get('text', '')
                techniques = sample.get('techniques', [])

                # Filter to valid techniques
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

        # Tokenize text
        encoding = self.tokenizer(
            sample['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        # Convert techniques to multi-hot labels
        labels = torch.zeros(len(self.technique_list))
        for tech in sample['techniques']:
            if tech in self.technique_to_idx:
                labels[self.technique_to_idx[tech]] = 1.0

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels,
        }


def setup_distributed():
    """Initialize distributed training."""
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


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute multi-label classification metrics."""
    label_binary = labels.astype(int)

    # Try multiple thresholds and find optimal (include very low thresholds for extreme imbalance)
    best_f1 = 0
    best_threshold = threshold
    # Extended range for extreme class imbalance - go much lower
    thresholds = [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    for t in thresholds:
        pred_t = (predictions > t).astype(int)
        f1_t = f1_score(label_binary, pred_t, average='micro', zero_division=0)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_threshold = t

    # Use best threshold for metrics
    pred_binary = (predictions > best_threshold).astype(int)

    # Micro metrics (across all labels)
    micro_f1 = f1_score(label_binary, pred_binary, average='micro', zero_division=0)
    micro_precision = precision_score(label_binary, pred_binary, average='micro', zero_division=0)
    micro_recall = recall_score(label_binary, pred_binary, average='micro', zero_division=0)

    # Macro metrics (per label, then average)
    macro_f1 = f1_score(label_binary, pred_binary, average='macro', zero_division=0)

    # Samples metrics (per sample, then average)
    samples_f1 = f1_score(label_binary, pred_binary, average='samples', zero_division=0)

    # Top-k accuracy and F1
    top5_acc = top_k_accuracy(predictions, labels, k=5)
    top10_acc = top_k_accuracy(predictions, labels, k=10)
    topk_f1 = top_k_f1(predictions, labels, k=3)

    return {
        'micro_f1': micro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'macro_f1': macro_f1,
        'samples_f1': samples_f1,
        'top5_acc': top5_acc,
        'top10_acc': top10_acc,
        'topk_f1': topk_f1,
        'best_threshold': best_threshold,
    }


def top_k_accuracy(predictions: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    """Compute top-k accuracy for multi-label classification."""
    correct = 0
    total = 0

    for pred, label in zip(predictions, labels):
        top_k_indices = np.argsort(pred)[-k:]
        true_indices = np.where(label > 0.5)[0]

        if len(true_indices) == 0:
            continue

        if len(set(top_k_indices) & set(true_indices)) > 0:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def top_k_f1(predictions: np.ndarray, labels: np.ndarray, k: int = 3) -> float:
    """Compute F1 using top-k predictions per sample."""
    all_pred_binary = []
    all_label_binary = []

    for pred, label in zip(predictions, labels):
        top_k_indices = set(np.argsort(pred)[-k:])
        pred_binary = np.zeros_like(pred)
        for idx in top_k_indices:
            pred_binary[idx] = 1
        all_pred_binary.append(pred_binary)
        all_label_binary.append(label)

    pred_binary = np.array(all_pred_binary)
    label_binary = np.array(all_label_binary).astype(int)

    return f1_score(label_binary, pred_binary, average='micro', zero_division=0)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: GradScaler,
    device: torch.device,
    rank: int,
    epoch: int,
    args,
    focal_loss_fn: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_main_loss = 0.0
    total_hierarchical_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(
        dataloader,
        desc=f'Epoch {epoch}',
        disable=rank != 0,
    )

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        with autocast(device_type=device.type, enabled=args.fp16 and device.type == 'cuda'):
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            if focal_loss_fn is not None:
                # Use adaptive focal loss for main loss
                main_loss = focal_loss_fn(outputs['logits'], labels)

                # Add hierarchical losses if available
                if 'tactic_loss' in outputs:
                    hierarchical_loss = (
                        0.1 * outputs['tactic_loss'] +
                        0.3 * outputs['technique_loss'] +
                        0.6 * outputs['subtechnique_loss']
                    )
                    loss = main_loss + 0.3 * hierarchical_loss
                else:
                    loss = main_loss
                    hierarchical_loss = torch.tensor(0.0)
            else:
                # Use model's built-in loss
                loss = outputs['loss']
                main_loss = outputs.get('main_loss', loss)
                hierarchical_loss = loss - main_loss if 'main_loss' in outputs else torch.tensor(0.0)

        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_main_loss += main_loss.item()
        if hasattr(hierarchical_loss, 'item'):
            total_hierarchical_loss += hierarchical_loss.item()
        num_batches += 1

        if rank == 0:
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
            })

    avg_loss = total_loss / num_batches
    return {
        'train_loss': avg_loss,
        'train_main_loss': total_main_loss / num_batches,
        'train_hierarchical_loss': total_hierarchical_loss / num_batches,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    rank: int,
    world_size: int,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_predictions = []
    all_labels = []

    progress_bar = tqdm(
        dataloader,
        desc='Evaluating',
        disable=rank != 0,
    )

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        total_loss += outputs['loss'].item()
        num_batches += 1

        probs = torch.sigmoid(outputs['logits'])
        all_predictions.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    if world_size > 1:
        gathered_preds = [None] * world_size
        gathered_labels = [None] * world_size
        dist.all_gather_object(gathered_preds, predictions)
        dist.all_gather_object(gathered_labels, labels)
        if rank == 0:
            predictions = np.concatenate(gathered_preds, axis=0)
            labels = np.concatenate(gathered_labels, axis=0)

    avg_loss = total_loss / num_batches

    if rank == 0:
        metrics = compute_metrics(predictions, labels)
        metrics['val_loss'] = avg_loss
        return metrics, predictions, labels
    else:
        return {'val_loss': avg_loss}, predictions, labels


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    metrics: Dict[str, float],
    output_dir: Path,
    is_best: bool = False,
):
    """Save model checkpoint."""
    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }

    checkpoint_path = output_dir / 'checkpoint_latest.pt'
    torch.save(checkpoint, checkpoint_path)

    epoch_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, epoch_path)

    if is_best:
        best_path = output_dir / 'checkpoint_best.pt'
        torch.save(checkpoint, best_path)


def main():
    parser = argparse.ArgumentParser(description='Train SecBERT + Enhanced KG-GNN')

    # Data arguments
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, required=True)
    parser.add_argument('--kg_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)

    # Model arguments
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--gnn_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--freeze_bert_layers', type=int, default=0)
    parser.add_argument('--use_hierarchical_loss', action='store_true', default=True)

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--max_length', type=int, default=512)

    # Optimization arguments
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--gradient_checkpointing', action='store_true')

    # Other arguments
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--early_stopping_patience', type=int, default=5)

    # Focal loss arguments
    parser.add_argument('--use_focal_loss', action='store_true')
    parser.add_argument('--focal_gamma_init', type=float, default=2.0)
    parser.add_argument('--focal_gamma_max', type=float, default=5.0)
    parser.add_argument('--focal_warmup_epochs', type=int, default=5)

    args = parser.parse_args()

    # Setup
    rank, local_rank, world_size = setup_distributed()

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")

    set_seed(args.seed + rank)

    # Create output directory
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'args.json', 'w') as f:
            json.dump(vars(args), f, indent=2)

    if world_size > 1:
        dist.barrier()

    # Create model
    if rank == 0:
        print("Creating SecBERT + Enhanced KG-GNN model...")

    model = create_enhanced_model(
        kg_dir=args.kg_dir,
        bert_model=args.bert_model,
        hidden_dim=args.hidden_dim,
        gnn_layers=args.gnn_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        freeze_bert_layers=args.freeze_bert_layers,
        use_hierarchical_loss=args.use_hierarchical_loss,
    )

    model = model.to(device)

    # Load tokenizer (needed for both dataset and KG semantic embeddings)
    if rank == 0:
        print(f"Loading tokenizer: {args.bert_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    # Pre-compute KG semantic embeddings using shared BERT
    if rank == 0:
        print("Pre-computing KG semantic embeddings using shared BERT...")
    model.precompute_kg_embeddings(device, tokenizer=tokenizer)

    if rank == 0:
        print(f"  - Num techniques: {model.num_techniques}")
        print(f"  - Num tactics: {model.num_tactics}")
        print(f"  - Num parent techniques: {model.num_parent_techniques}")
        print(f"  - Num sub-techniques: {model.num_subtechniques}")

    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        model.bert.gradient_checkpointing_enable()

    # Wrap in DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Count parameters
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")

    # Get technique list from model (tokenizer already loaded above)
    if isinstance(model, DDP):
        technique_list = model.module.technique_list
    else:
        technique_list = model.technique_list

    # Create datasets
    if rank == 0:
        print(f"Loading training data from {args.train_data}...")
    train_dataset = SecurityLogDataset(
        args.train_data,
        tokenizer,
        technique_list,
        max_length=args.max_length,
    )

    if rank == 0:
        print(f"Loading validation data from {args.val_data}...")
    val_dataset = SecurityLogDataset(
        args.val_data,
        tokenizer,
        technique_list,
        max_length=args.max_length,
    )

    if rank == 0:
        print(f"  - Training samples: {len(train_dataset)}")
        print(f"  - Validation samples: {len(val_dataset)}")

    # Create data loaders
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    ) if world_size > 1 else None

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    ) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Create focal loss if requested
    focal_loss_fn = None
    if args.use_focal_loss and FOCAL_LOSS_AVAILABLE:
        if rank == 0:
            print("Setting up Adaptive Focal Loss...")

        technique_counts = load_technique_counts(args.train_data)
        class_weights = compute_class_weights(
            technique_counts,
            technique_list,
            power=0.5,
            min_weight=0.1,
            max_weight=10.0,
        )
        class_weights = class_weights.to(device)

        focal_loss_fn = HybridAdaptiveFocalLoss(
            class_weights=class_weights,
            gamma_init=args.focal_gamma_init,
            gamma_max=args.focal_gamma_max,
            warmup_epochs=args.focal_warmup_epochs,
        )

        if rank == 0:
            print(f"  - Class weights: min={class_weights.min():.3f}, max={class_weights.max():.3f}")
            print(f"  - Gamma schedule: {args.focal_gamma_init} → {args.focal_gamma_max}")
    elif args.use_focal_loss and not FOCAL_LOSS_AVAILABLE:
        if rank == 0:
            print("Warning: Focal loss requested but not available. Using BCE.")

    # Create optimizer
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters()
                      if not any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': args.weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters()
                      if any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    # Create scheduler
    num_training_steps = len(train_loader) * args.epochs
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=args.warmup_steps,
    )
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps - args.warmup_steps,
        eta_min=args.lr * 0.1,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[args.warmup_steps],
    )

    scaler = GradScaler(enabled=args.fp16 and torch.cuda.is_available())

    # Training loop
    best_f1 = 0.0
    epochs_without_improvement = 0

    if rank == 0:
        print(f"\nStarting training for up to {args.epochs} epochs...")
        print(f"  - Batch size: {args.batch_size * world_size}")
        print(f"  - Steps per epoch: {len(train_loader)}")
        print(f"  - Early stopping patience: {args.early_stopping_patience}")
        print(f"  - Hierarchical loss: {args.use_hierarchical_loss}")
        print(f"  - Focal loss: {args.use_focal_loss}")

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Update focal loss gamma
        if focal_loss_fn is not None:
            focal_loss_fn.update_gamma(epoch, args.epochs)
            if rank == 0:
                print(f"\nEpoch {epoch}: gamma = {focal_loss_fn.get_current_gamma():.2f}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            scaler, device, rank, epoch, args,
            focal_loss_fn=focal_loss_fn,
        )

        # Evaluate
        if epoch % args.eval_interval == 0:
            val_metrics, predictions, labels = evaluate(
                model, val_loader, device, rank, world_size,
            )

            if rank == 0:
                all_metrics = {**train_metrics, **val_metrics}

                print(f"\nEpoch {epoch} Results:")
                print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
                print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
                print(f"  Micro F1: {val_metrics['micro_f1']:.4f}")
                print(f"  Macro F1: {val_metrics['macro_f1']:.4f}")
                print(f"  Top-5 Acc: {val_metrics['top5_acc']:.4f}")
                print(f"  Top-10 Acc: {val_metrics['top10_acc']:.4f}")
                print(f"  Best Threshold: {val_metrics['best_threshold']:.3f}")

                # Save metrics
                metrics_path = output_dir / 'metrics.jsonl'
                with open(metrics_path, 'a') as f:
                    f.write(json.dumps({'epoch': epoch, **all_metrics}) + '\n')

                # Check for best model
                is_best = val_metrics['micro_f1'] > best_f1
                if is_best:
                    best_f1 = val_metrics['micro_f1']
                    epochs_without_improvement = 0
                    print(f"  New best Micro F1: {best_f1:.4f}")
                else:
                    epochs_without_improvement += 1
                    print(f"  No improvement for {epochs_without_improvement} epoch(s)")

                # Save checkpoint
                if epoch % args.save_interval == 0 or is_best:
                    save_checkpoint(
                        model, optimizer, scheduler,
                        epoch, all_metrics, output_dir, is_best,
                    )

                # Early stopping
                if epochs_without_improvement >= args.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs!")
                    break

        if world_size > 1:
            dist.barrier()

    if rank == 0:
        print(f"\nTraining complete!")
        print(f"Best Micro F1: {best_f1:.4f}")
        print(f"Checkpoints saved to: {output_dir}")

    cleanup_distributed()


if __name__ == '__main__':
    main()
