#!/usr/bin/env python3
"""
Post-training optimization for SecBERT + Advanced KG-GNN model.

Loads a trained checkpoint and optimizes:
1. Per-class thresholds for best F1
2. Evaluates on validation set
3. Saves optimized thresholds and final metrics
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent))

from secbert_advanced_gnn_model import SecBERTAdvancedGNNModel
from per_class_threshold import optimize_per_class_thresholds, apply_per_class_thresholds


class SecurityLogDataset(torch.utils.data.Dataset):
    """Simple dataset for inference."""

    def __init__(self, data_path: str, tokenizer, technique_list: list, max_length: int = 256):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.technique_to_idx = {t: i for i, t in enumerate(technique_list)}
        self.num_techniques = len(technique_list)

        with open(data_path, 'r') as f:
            for line in f:
                sample = json.loads(line)
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        encoding = self.tokenizer(
            sample['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        # Multi-hot labels
        labels = torch.zeros(self.num_techniques)
        for tech in sample.get('techniques', []):
            if tech in self.technique_to_idx:
                labels[self.technique_to_idx[tech]] = 1

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels,
        }


def run_inference(model, dataloader, device):
    """Run inference and collect predictions and labels."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_preds.append(probs)
            all_labels.append(labels.numpy())

    return np.vstack(all_preds), np.vstack(all_labels)


def main():
    parser = argparse.ArgumentParser(description='Post-training optimization')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--val_data', type=str, required=True, help='Path to validation data')
    parser.add_argument('--kg_dir', type=str, required=True, help='Path to knowledge graph directory')
    parser.add_argument('--bert_model', type=str, default='jackaduma/SecBERT')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (default: checkpoint dir)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Output directory
    if args.output_dir is None:
        args.output_dir = str(Path(args.checkpoint).parent)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.bert_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model = SecBERTAdvancedGNNModel(
        kg_dir=args.kg_dir,
        bert_model=args.bert_model,
        hidden_dim=512,
        gnn_layers=4,
        gnn_heads=8,
        dropout=0.1,
        use_transe_pretrain=False,  # Already trained
    )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model = model.to(device)
    print(f"  Techniques: {model.num_techniques}")

    # Load validation data
    print(f"\nLoading validation data: {args.val_data}")
    val_dataset = SecurityLogDataset(
        args.val_data,
        tokenizer,
        model.technique_list,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"  Samples: {len(val_dataset)}")

    # Run inference
    print("\n" + "="*60)
    print("Running inference on validation set...")
    print("="*60)
    predictions, labels = run_inference(model, val_loader, device)
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Labels shape: {labels.shape}")

    # Baseline metrics (threshold = 0.5)
    print("\n" + "="*60)
    print("Baseline metrics (threshold = 0.5)")
    print("="*60)
    baseline_preds = (predictions >= 0.5).astype(int)
    baseline_micro = f1_score(labels, baseline_preds, average='micro', zero_division=0)
    baseline_macro = f1_score(labels, baseline_preds, average='macro', zero_division=0)
    print(f"  Micro F1: {baseline_micro*100:.2f}%")
    print(f"  Macro F1: {baseline_macro*100:.2f}%")

    # Optimize per-class thresholds
    print("\n" + "="*60)
    print("Optimizing per-class thresholds...")
    print("="*60)

    threshold_range = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]

    optimal_thresholds, per_class_f1 = optimize_per_class_thresholds(
        predictions, labels, model.technique_list,
        threshold_range=threshold_range,
        min_positive_samples=3,
    )

    # Apply optimized thresholds
    optimized_preds = apply_per_class_thresholds(predictions, optimal_thresholds, model.technique_list)

    optimized_micro = f1_score(labels, optimized_preds, average='micro', zero_division=0)
    optimized_macro = f1_score(labels, optimized_preds, average='macro', zero_division=0)
    optimized_precision = precision_score(labels, optimized_preds, average='micro', zero_division=0)
    optimized_recall = recall_score(labels, optimized_preds, average='micro', zero_division=0)

    print(f"\nOptimized metrics:")
    print(f"  Micro F1:    {optimized_micro*100:.2f}%  (was {baseline_micro*100:.2f}%)")
    print(f"  Macro F1:    {optimized_macro*100:.2f}%  (was {baseline_macro*100:.2f}%)")
    print(f"  Precision:   {optimized_precision*100:.2f}%")
    print(f"  Recall:      {optimized_recall*100:.2f}%")

    # Threshold statistics
    thresh_values = list(optimal_thresholds.values())
    print(f"\nThreshold statistics:")
    print(f"  Mean:   {np.mean(thresh_values):.3f}")
    print(f"  Std:    {np.std(thresh_values):.3f}")
    print(f"  Min:    {np.min(thresh_values):.3f}")
    print(f"  Max:    {np.max(thresh_values):.3f}")

    # Distribution
    thresh_dist = {}
    for t in thresh_values:
        thresh_dist[t] = thresh_dist.get(t, 0) + 1
    print(f"\nThreshold distribution:")
    for t in sorted(thresh_dist.keys()):
        print(f"  {t:.2f}: {thresh_dist[t]} classes")

    # Save results
    output_file = Path(args.output_dir) / 'optimized_thresholds.json'
    results = {
        'baseline_micro_f1': baseline_micro,
        'baseline_macro_f1': baseline_macro,
        'optimized_micro_f1': optimized_micro,
        'optimized_macro_f1': optimized_macro,
        'optimized_precision': optimized_precision,
        'optimized_recall': optimized_recall,
        'threshold_mean': float(np.mean(thresh_values)),
        'threshold_std': float(np.std(thresh_values)),
        'thresholds': optimal_thresholds,
        'per_class_f1': per_class_f1,
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved optimized thresholds to: {output_file}")

    # Top techniques by F1
    print("\n" + "="*60)
    print("Top 20 techniques by F1 score:")
    print("="*60)
    sorted_f1 = sorted(per_class_f1.items(), key=lambda x: x[1], reverse=True)
    for tech, f1 in sorted_f1[:20]:
        thresh = optimal_thresholds.get(tech, 0.2)
        print(f"  {tech}: F1={f1:.3f}, threshold={thresh:.2f}")

    # Bottom techniques
    print("\n" + "="*60)
    print("Bottom 20 techniques by F1 score:")
    print("="*60)
    for tech, f1 in sorted_f1[-20:]:
        thresh = optimal_thresholds.get(tech, 0.2)
        n_pos = int(labels[:, model.technique_list.index(tech)].sum()) if tech in model.technique_list else 0
        print(f"  {tech}: F1={f1:.3f}, threshold={thresh:.2f}, n_pos={n_pos}")

    print("\n" + "="*60)
    print("Post-training optimization complete!")
    print("="*60)


if __name__ == '__main__':
    main()
