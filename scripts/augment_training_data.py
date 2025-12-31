#!/usr/bin/env python3
"""
Data Augmentation for CyberBERT-GNN Training Data

Implements multiple augmentation strategies to expand the training dataset:
1. Entity masking and replacement
2. Synonym substitution (security-aware)
3. Field permutation
4. Template-based generation
"""

import json
import random
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import argparse


class SecurityDataAugmenter:
    """Augment security log data for multi-label classification."""

    def __init__(self, seed: int = 42):
        random.seed(seed)

        # Security-specific synonyms
        self.synonyms = {
            'powershell.exe': ['pwsh.exe', 'PowerShell', 'powershell'],
            'cmd.exe': ['cmd', 'command prompt', 'command shell'],
            'process creation': ['process start', 'new process', 'process spawned'],
            'registry': ['registry key', 'reg', 'windows registry'],
            'network connection': ['network activity', 'connection', 'network traffic'],
            'user': ['account', 'username'],
        }

        # Entity patterns for masking/replacement
        self.entity_patterns = {
            'ipv4': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            'process_path': r'[A-Z]:\\\\(?:[^\\:\*\?"<>\|]+\\)*[^\\:\*\?"<>\|]+',
            'hash': r'\b[A-Fa-f0-9]{32,64}\b',
            'username': r'(?:User|Account|Username):\s*([A-Za-z0-9_\-\.]+)',
            'registry_key': r'HK[A-Z_]+\\[A-Za-z0-9_\\]+',
        }

        # Replacement pools
        self.replacements = {
            'ipv4': [
                '192.168.1.100', '10.0.0.50', '172.16.5.10',
                '8.8.8.8', '1.1.1.1', '192.168.0.5'
            ],
            'process_path': [
                'C:\\Windows\\System32\\svchost.exe',
                'C:\\Windows\\System32\\rundll32.exe',
                'C:\\Program Files\\Internet Explorer\\iexplore.exe',
                'C:\\Windows\\explorer.exe',
                'C:\\Windows\\System32\\cmd.exe',
                'C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe',
            ],
            'username': [
                'Administrator', 'SYSTEM', 'admin', 'user',
                'service_account', 'domain_admin', 'guest'
            ],
        }

    def mask_entities(self, text: str, mask_prob: float = 0.3) -> str:
        """
        Randomly mask entities with generic placeholders.

        Args:
            text: Original log text
            mask_prob: Probability of masking each entity type

        Returns:
            Text with some entities masked
        """
        result = text

        for entity_type, pattern in self.entity_patterns.items():
            if random.random() < mask_prob:
                mask = f"[{entity_type.upper()}]"
                result = re.sub(pattern, mask, result)

        return result

    def replace_entities(self, text: str, replace_prob: float = 0.5) -> str:
        """
        Replace specific entities with alternatives from the same category.

        Args:
            text: Original log text
            replace_prob: Probability of replacing each entity instance

        Returns:
            Text with entities replaced
        """
        result = text

        for entity_type, pattern in self.entity_patterns.items():
            if entity_type not in self.replacements:
                continue

            matches = list(re.finditer(pattern, text))
            for match in matches:
                if random.random() < replace_prob:
                    replacement = random.choice(self.replacements[entity_type])
                    result = result.replace(match.group(), replacement, 1)

        return result

    def synonym_substitution(self, text: str, sub_prob: float = 0.3) -> str:
        """
        Substitute words with security domain synonyms.

        Args:
            text: Original log text
            sub_prob: Probability of substituting each matched term

        Returns:
            Text with synonyms substituted
        """
        result = text

        for term, synonyms in self.synonyms.items():
            if term in text.lower() and random.random() < sub_prob:
                synonym = random.choice(synonyms)
                # Case-insensitive replacement
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                result = pattern.sub(synonym, result, count=1)

        return result

    def permute_fields(self, text: str) -> str:
        """
        Permute the order of fields in structured logs.

        Works for logs with field1: value1 field2: value2 format.
        """
        # Find all field:value pairs
        pattern = r'([A-Za-z_]+):\s*([^\s][^:]*?)(?=\s+[A-Za-z_]+:|$)'
        matches = re.findall(pattern, text)

        if len(matches) < 2:
            return text  # Not enough fields to permute

        # Keep first field (usually event type) and shuffle the rest
        first = matches[0]
        rest = matches[1:]
        random.shuffle(rest)

        # Reconstruct
        fields = [first] + rest
        result = ' '.join([f"{field}: {value}" for field, value in fields])

        # Add back any prefix (like [Event 1])
        prefix_match = re.match(r'^(\[[^\]]+\])\s*', text)
        if prefix_match:
            result = prefix_match.group(1) + ' ' + result

        return result

    def augment_sample(
        self,
        sample: Dict,
        strategies: List[str] = ['entity_mask', 'entity_replace', 'synonym', 'permute'],
        num_augmentations: int = 1,
    ) -> List[Dict]:
        """
        Generate augmented versions of a sample.

        Args:
            sample: Original sample dict with 'text' and 'techniques'
            strategies: List of augmentation strategies to apply
            num_augmentations: Number of augmented versions to create

        Returns:
            List of augmented samples
        """
        augmented = []

        for _ in range(num_augmentations):
            text = sample['text']

            # Randomly select subset of strategies to apply
            num_strategies = random.randint(1, len(strategies))
            selected = random.sample(strategies, num_strategies)

            # Apply selected strategies
            for strategy in selected:
                if strategy == 'entity_mask':
                    text = self.mask_entities(text, mask_prob=0.3)
                elif strategy == 'entity_replace':
                    text = self.replace_entities(text, replace_prob=0.5)
                elif strategy == 'synonym':
                    text = self.synonym_substitution(text, sub_prob=0.3)
                elif strategy == 'permute':
                    text = self.permute_fields(text)

            # Create augmented sample
            aug_sample = {
                'text': text,
                'techniques': sample['techniques'],
                'source': sample.get('source', 'unknown') + '_augmented',
                'original_source': sample.get('source', 'unknown'),
                'augmentation': selected,
            }

            # Add hash for deduplication
            aug_sample['hash'] = hashlib.md5(text.encode()).hexdigest()[:12]

            augmented.append(aug_sample)

        return augmented

    def augment_dataset(
        self,
        samples: List[Dict],
        target_multiplier: float = 2.0,
        balance_classes: bool = True,
        min_samples_per_class: int = 100,
    ) -> List[Dict]:
        """
        Augment entire dataset with class balancing.

        Args:
            samples: Original training samples
            target_multiplier: Target dataset size = original * multiplier
            balance_classes: Whether to oversample minority classes
            min_samples_per_class: Minimum samples per technique after augmentation

        Returns:
            Augmented dataset (original + generated)
        """
        print(f"Starting augmentation:")
        print(f"  Original samples: {len(samples)}")
        print(f"  Target multiplier: {target_multiplier}x")

        # Count samples per technique
        technique_counts = defaultdict(int)
        for sample in samples:
            for tech in sample['techniques']:
                technique_counts[tech] += 1

        # Determine augmentation strategy per sample
        augmented = list(samples)  # Start with originals

        if balance_classes:
            # Oversample minority classes more
            max_count = max(technique_counts.values())

            for sample in samples:
                # Get rarest technique in this sample
                min_count = min(technique_counts[tech] for tech in sample['techniques'])

                # Calculate augmentation factor based on rarity
                if min_count < min_samples_per_class:
                    # Rare techniques: augment heavily
                    num_aug = int((min_samples_per_class - min_count) / len(sample['techniques']))
                    num_aug = min(num_aug, 10)  # Cap at 10x
                elif min_count < max_count * 0.5:
                    # Medium rarity: moderate augmentation
                    num_aug = random.randint(1, 3)
                else:
                    # Common techniques: minimal augmentation
                    num_aug = 0 if random.random() > 0.3 else 1

                if num_aug > 0:
                    aug_samples = self.augment_sample(sample, num_augmentations=num_aug)
                    augmented.extend(aug_samples)
        else:
            # Uniform augmentation
            target_size = int(len(samples) * target_multiplier)
            num_to_generate = target_size - len(samples)

            # Randomly select samples to augment
            for _ in range(num_to_generate):
                sample = random.choice(samples)
                aug_samples = self.augment_sample(sample, num_augmentations=1)
                augmented.extend(aug_samples)

        # Deduplicate by hash
        seen_hashes = set()
        unique_augmented = []
        for sample in augmented:
            h = sample.get('hash', hashlib.md5(sample['text'].encode()).hexdigest()[:12])
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_augmented.append(sample)

        print(f"\nAugmentation complete:")
        print(f"  Generated samples: {len(augmented) - len(samples)}")
        print(f"  After deduplication: {len(unique_augmented)}")
        print(f"  Final dataset size: {len(unique_augmented)}")
        print(f"  Actual multiplier: {len(unique_augmented) / len(samples):.2f}x")

        return unique_augmented


def main():
    parser = argparse.ArgumentParser(description='Augment security log training data')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input JSONL file with training data'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output JSONL file for augmented data'
    )
    parser.add_argument(
        '--multiplier',
        type=float,
        default=2.0,
        help='Target dataset size multiplier (default: 2.0)'
    )
    parser.add_argument(
        '--balance',
        action='store_true',
        help='Balance classes by oversampling minority techniques'
    )
    parser.add_argument(
        '--min_samples',
        type=int,
        default=100,
        help='Minimum samples per technique (with --balance)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Load input data
    print(f"Loading data from {args.input}")
    samples = []
    with open(args.input, 'r') as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    # Create augmenter
    augmenter = SecurityDataAugmenter(seed=args.seed)

    # Augment dataset
    augmented = augmenter.augment_dataset(
        samples,
        target_multiplier=args.multiplier,
        balance_classes=args.balance,
        min_samples_per_class=args.min_samples,
    )

    # Save augmented data
    print(f"\nSaving augmented data to {args.output}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for sample in augmented:
            # Remove hash before saving
            sample_clean = {k: v for k, v in sample.items() if k != 'hash'}
            f.write(json.dumps(sample_clean) + '\n')

    print(f"Done! Augmented dataset saved to {args.output}")


if __name__ == '__main__':
    main()
