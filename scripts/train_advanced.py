#!/usr/bin/env python3
"""
Training Script for SecBERT + Advanced KG-GNN.

Features:
1. SecBERT text encoder (security-domain pretrained)
2. Advanced GNN with R-GCN, HGT, TransE pre-training, and text conditioning
3. Adaptive Focal Loss for class imbalance
4. Mixed precision training
5. Gradient accumulation
6. Distributed training support (DDP)

Usage:
    # Single GPU
    python train_advanced.py \
        --train_data data/combined_train.jsonl \
        --val_data data/combined_val.jsonl \
        --kg_dir data/attack_framework \
        --output_dir checkpoints/advanced_v1 \
        --batch_size 16

    # Multi-GPU
    torchrun --nproc_per_node=8 train_advanced.py \
        --train_data data/combined_train.jsonl \
        --val_data data/combined_val.jsonl \
        --kg_dir data/attack_framework \
        --output_dir checkpoints/advanced_v1 \
        --batch_size 16
"""

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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

from secbert_advanced_gnn_model import SecBERTAdvancedGNNModel, create_advanced_model

# Try to import focal loss
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'fixes'))
try:
    from adaptive_focal_loss import HybridAdaptiveFocalLoss, compute_class_weights
    FOCAL_LOSS_AVAILABLE = True
except ImportError:
    FOCAL_LOSS_AVAILABLE = False
    print("Warning: Adaptive focal loss not available, using BCE")


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


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    binary_preds = (predictions >= threshold).astype(int)

    # Per-sample metrics
    micro_f1 = f1_score(labels, binary_preds, average='micro', zero_division=0)
    macro_f1 = f1_score(labels, binary_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(labels, binary_preds, average='weighted', zero_division=0)

    # Precision and recall
    micro_precision = precision_score(labels, binary_preds, average='micro', zero_division=0)
    micro_recall = recall_score(labels, binary_preds, average='micro', zero_division=0)

    return {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'precision': micro_precision,
        'recall': micro_recall,
    }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    rank: int,
    accumulation_steps: int = 1,
    use_text_conditioning: bool = True,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=rank != 0)

    optimizer.zero_grad()

    for step, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with autocast(device_type='cuda', dtype=torch.float16):
            logits = model(
                input_ids,
                attention_mask,
                use_text_conditioning=use_text_conditioning,
            )
            loss = criterion(logits, labels)
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

        # Collect predictions
        with torch.no_grad():
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})

    # Compute metrics
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    metrics = compute_metrics(all_preds, all_labels)
    metrics['loss'] = total_loss / len(dataloader)

    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    rank: int,
    use_text_conditioning: bool = True,
) -> Dict[str, float]:
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

        with autocast(device_type='cuda', dtype=torch.float16):
            logits = model(
                input_ids,
                attention_mask,
                use_text_conditioning=use_text_conditioning,
            )
            loss = criterion(logits, labels)

        total_loss += loss.item()

        preds = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    # Compute metrics
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    metrics = compute_metrics(all_preds, all_labels)
    metrics['loss'] = total_loss / len(dataloader)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', required=True, help='Training data JSONL')
    parser.add_argument('--val_data', required=True, help='Validation data JSONL')
    parser.add_argument('--kg_dir', required=True, help='Knowledge graph directory')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--bert_model', default='jackaduma/SecBERT', help='BERT model')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--gnn_layers', type=int, default=4)
    parser.add_argument('--gnn_heads', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_focal_loss', action='store_true')
    parser.add_argument('--use_transe_pretrain', action='store_true', default=True)
    parser.add_argument('--transe_epochs', type=int, default=100)
    parser.add_argument('--use_text_conditioning', action='store_true', default=True)
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    args = parser.parse_args()

    # Setup distributed training
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    # Set seed
    set_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    if rank == 0:
        print(f"Loading tokenizer: {args.bert_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    # Create model
    if rank == 0:
        print(f"Creating SecBERT + Advanced KG-GNN model...")

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

    # Wrap with DDP if distributed
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        base_model = model.module
    else:
        base_model = model

    # Create datasets
    if rank == 0:
        print(f"Loading datasets...")

    train_dataset = SecurityLogDataset(
        args.train_data,
        tokenizer,
        base_model.technique_list,
        max_length=args.max_length,
    )

    val_dataset = SecurityLogDataset(
        args.val_data,
        tokenizer,
        base_model.technique_list,
        max_length=args.max_length,
    )

    if rank == 0:
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")
        print(f"  Techniques: {len(base_model.technique_list)}")

    # Create dataloaders
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Loss function
    if args.use_focal_loss and FOCAL_LOSS_AVAILABLE:
        if rank == 0:
            print("Using Adaptive Focal Loss")
        criterion = HybridAdaptiveFocalLoss(
            gamma=2.0,
            pos_weight_factor=2.0,
        )
    else:
        if rank == 0:
            print("Using BCE with Logits Loss")
        criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler with warmup
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=args.warmup_epochs * len(train_loader),
    )
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(args.epochs - args.warmup_epochs) * len(train_loader),
        eta_min=1e-7,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[args.warmup_epochs * len(train_loader)],
    )

    # Mixed precision
    scaler = GradScaler()

    # Resume from checkpoint
    start_epoch = 0
    best_f1 = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint.get('best_f1', 0.0)
        if rank == 0:
            print(f"Resumed from epoch {start_epoch}, best F1: {best_f1:.4f}")

    # Training loop
    if rank == 0:
        print(f"\nStarting training...")
        print(f"  Epochs: {args.epochs}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Learning rate: {args.lr}")
        print(f"  Text conditioning: {args.use_text_conditioning}")

    metrics_log = []

    for epoch in range(start_epoch, args.epochs):
        if world_size > 1:
            train_sampler.set_epoch(epoch)

        # Train
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
            epoch + 1,
            rank,
            args.accumulation_steps,
            args.use_text_conditioning,
        )

        # Update scheduler
        scheduler.step()

        # Evaluate
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            rank,
            args.use_text_conditioning,
        )

        # Log metrics
        if rank == 0:
            log_entry = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_micro_f1': train_metrics['micro_f1'],
                'train_macro_f1': train_metrics['macro_f1'],
                'val_loss': val_metrics['loss'],
                'val_micro_f1': val_metrics['micro_f1'],
                'val_macro_f1': val_metrics['macro_f1'],
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall'],
                'lr': optimizer.param_groups[0]['lr'],
            }
            metrics_log.append(log_entry)

            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Train Micro F1: {train_metrics['micro_f1']:.4f}")
            print(f"  Train Macro F1: {train_metrics['macro_f1']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val Micro F1: {val_metrics['micro_f1']:.4f}")
            print(f"  Val Macro F1: {val_metrics['macro_f1']:.4f}")
            print(f"  Val Precision: {val_metrics['precision']:.4f}")
            print(f"  Val Recall: {val_metrics['recall']:.4f}")

            # Save metrics
            with open(output_dir / 'metrics.jsonl', 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

            # Save best model
            combined_f1 = (val_metrics['micro_f1'] + val_metrics['macro_f1']) / 2
            if combined_f1 > best_f1:
                best_f1 = combined_f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_f1': best_f1,
                    'metrics': val_metrics,
                    'args': vars(args),
                }, output_dir / 'best_model.pt')
                print(f"  Saved best model (F1: {best_f1:.4f})")

            # Save latest checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
            }, output_dir / 'latest_checkpoint.pt')

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()

    if rank == 0:
        print(f"\nTraining complete!")
        print(f"Best combined F1: {best_f1:.4f}")
        print(f"Model saved to: {output_dir}")


if __name__ == '__main__':
    main()
