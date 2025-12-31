#!/usr/bin/env python3
"""
Generate comprehensive ATT&CK technique training dataset using Ollama (Llama 3.1 70B).

This script generates realistic security log samples for each ATT&CK technique
using Ollama's local inference.
"""

import json
import os
import random
import argparse
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class TechniqueInfo:
    """Information about an ATT&CK technique."""
    id: str
    name: str
    description: str
    tactics: List[str]
    platforms: List[str]
    is_subtechnique: bool
    parent_id: Optional[str] = None


# Log types to generate for diversity
LOG_TYPES = [
    "sysmon_process_create",
    "sysmon_network_connection",
    "sysmon_file_create",
    "sysmon_registry",
    "windows_security",
    "linux_audit",
    "edr_alert",
]


GENERATION_PROMPT = """You are a cybersecurity expert generating realistic security log samples for training a threat detection model.

Generate {num_samples} UNIQUE and REALISTIC security log entries that would indicate the following MITRE ATT&CK technique being executed:

**Technique ID**: {technique_id}
**Technique Name**: {technique_name}
**Description**: {description}
**Tactics**: {tactics}
**Platforms**: {platforms}

Requirements:
1. Generate realistic {log_type} format logs
2. Include realistic file paths, IP addresses, usernames, process names
3. Include actual command-line arguments that would be used in real attacks
4. Make each sample UNIQUE with variations in:
   - Different tools/methods to achieve the same technique
   - Different file paths and process names
   - Different encodings/obfuscations where applicable
   - Different user contexts (SYSTEM, domain users, local admin)
5. DO NOT include the technique ID in the log text
6. DO NOT include explanations - only raw log entries

Output ONLY the log entries, separated by "---SAMPLE---". No other text.

Begin:"""


MULTI_LABEL_PROMPT = """You are a cybersecurity expert generating realistic attack chain security logs.

Generate {num_samples} UNIQUE security log entries showing these MITRE ATT&CK techniques used TOGETHER in an attack:

**Techniques**:
{techniques_list}

**Attack Scenario**: {scenario}

Requirements:
1. Each log should show evidence of MULTIPLE techniques being used together
2. Use {log_type} format
3. Include realistic artifacts that demonstrate the technique combination
4. Make each sample unique
5. DO NOT include technique IDs in the log text

Output ONLY the log entries, separated by "---SAMPLE---". No other text.

Begin:"""


class OllamaDatasetGenerator:
    """Generate security log dataset using Ollama."""

    def __init__(
        self,
        model_name: str = "llama3.1:70b",
        kg_dir: str = "../data/attack_framework",
        output_dir: str = "../data/generated",
        ollama_url: str = "http://localhost:11434",
    ):
        self.model_name = model_name
        self.kg_dir = Path(kg_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ollama_url = ollama_url

        # Load technique information
        self.techniques = self._load_techniques()
        self.technique_list = list(self.techniques.keys())

        # Group techniques by tactic
        self.tactic_techniques = self._group_by_tactic()

        # Common attack scenarios for multi-label samples
        self.attack_scenarios = [
            "Initial access via phishing followed by credential harvesting",
            "Lateral movement using stolen credentials and remote services",
            "Privilege escalation through local exploit then persistence establishment",
            "Data staging and exfiltration over encrypted channel",
            "Defense evasion techniques combined with execution",
            "Discovery phase followed by collection activities",
            "Command and control establishment with data encoding",
            "Persistence mechanisms combined with privilege escalation",
        ]

        print(f"Loaded {len(self.techniques)} techniques")
        print(f"Using Ollama model: {model_name}")

    def _load_techniques(self) -> Dict[str, TechniqueInfo]:
        """Load technique information from knowledge graph."""
        techniques = {}

        with open(self.kg_dir / "graph_nodes.json", "r") as f:
            nodes = json.load(f)

        with open(self.kg_dir / "technique_list.json", "r") as f:
            technique_ids = json.load(f)

        for tech_id in technique_ids:
            if tech_id in nodes:
                node = nodes[tech_id]
                parent_id = tech_id.split(".")[0] if "." in tech_id else None

                techniques[tech_id] = TechniqueInfo(
                    id=tech_id,
                    name=node.get("name", tech_id),
                    description=node.get("description", "")[:1200],
                    tactics=node.get("tactics", []),
                    platforms=node.get("platforms", ["Windows", "Linux", "macOS"]),
                    is_subtechnique=node.get("is_subtechnique", "." in tech_id),
                    parent_id=parent_id,
                )

        return techniques

    def _group_by_tactic(self) -> Dict[str, List[str]]:
        """Group techniques by tactic."""
        tactic_techniques = {}
        for tech_id, info in self.techniques.items():
            for tactic in info.tactics:
                if tactic not in tactic_techniques:
                    tactic_techniques[tactic] = []
                tactic_techniques[tactic].append(tech_id)
        return tactic_techniques

    def _generate_ollama(self, prompt: str, temperature: float = 0.8) -> str:
        """Generate text using Ollama API."""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "top_p": 0.9,
                        "num_predict": 2000,
                    }
                },
                timeout=120,
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            print(f"Ollama error: {e}")
            return ""

    def _parse_samples(self, response: str) -> List[str]:
        """Parse generated samples from response."""
        samples = []

        # Split by separator
        if "---SAMPLE---" in response:
            parts = response.split("---SAMPLE---")
        else:
            parts = response.split("\n\n\n")

        for part in parts:
            sample = part.strip()
            # Filter out too short or explanatory text
            if sample and len(sample) > 50 and not sample.startswith("Here"):
                # Remove markdown code blocks if present
                sample = sample.replace("```json", "").replace("```", "").strip()
                samples.append(sample)

        return samples

    def _get_log_type_for_technique(self, info: TechniqueInfo) -> str:
        """Select appropriate log type based on technique."""
        # Map tactics to preferred log types
        tactic_log_map = {
            "EXECUTION": ["sysmon_process_create", "windows_security"],
            "PERSISTENCE": ["sysmon_registry", "sysmon_file_create"],
            "PRIVILEGE_ESCALATION": ["sysmon_process_create", "windows_security"],
            "DEFENSE_EVASION": ["sysmon_process_create", "edr_alert"],
            "CREDENTIAL_ACCESS": ["windows_security", "sysmon_process_create"],
            "DISCOVERY": ["sysmon_process_create", "sysmon_network_connection"],
            "LATERAL_MOVEMENT": ["sysmon_network_connection", "windows_security"],
            "COLLECTION": ["sysmon_file_create", "sysmon_process_create"],
            "COMMAND_AND_CONTROL": ["sysmon_network_connection", "edr_alert"],
            "EXFILTRATION": ["sysmon_network_connection", "sysmon_file_create"],
            "IMPACT": ["sysmon_process_create", "sysmon_file_create"],
        }

        for tactic in info.tactics:
            if tactic in tactic_log_map:
                return random.choice(tactic_log_map[tactic])

        return random.choice(LOG_TYPES)

    def generate_single_technique_samples(
        self,
        technique_id: str,
        num_samples: int = 8,
    ) -> List[Dict]:
        """Generate samples for a single technique."""
        if technique_id not in self.techniques:
            return []

        info = self.techniques[technique_id]
        samples = []

        # Generate in batches of 4 for efficiency
        batch_size = 4
        num_batches = (num_samples + batch_size - 1) // batch_size

        for batch in range(num_batches):
            n = min(batch_size, num_samples - len(samples))
            log_type = self._get_log_type_for_technique(info)

            prompt = GENERATION_PROMPT.format(
                num_samples=n,
                technique_id=technique_id,
                technique_name=info.name,
                description=info.description[:800],
                tactics=", ".join(info.tactics),
                platforms=", ".join(info.platforms),
                log_type=log_type,
            )

            response = self._generate_ollama(prompt)
            parsed = self._parse_samples(response)

            for sample_text in parsed[:n]:
                samples.append({
                    "text": sample_text,
                    "techniques": [technique_id],
                    "source": f"llm_generated_{log_type}",
                    "tactics": info.tactics,
                })

        return samples

    def generate_multi_technique_samples(
        self,
        primary_technique: str,
        num_samples: int = 4,
    ) -> List[Dict]:
        """Generate samples showing multiple techniques together."""
        if primary_technique not in self.techniques:
            return []

        primary_info = self.techniques[primary_technique]

        # Find related techniques from same or adjacent tactics
        related = set()
        for tactic in primary_info.tactics:
            if tactic in self.tactic_techniques:
                for tech in self.tactic_techniques[tactic][:20]:  # Limit search
                    if tech != primary_technique:
                        related.add(tech)

        if not related:
            return []

        # Select 1-2 secondary techniques
        num_secondary = min(2, len(related))
        secondary = random.sample(list(related), num_secondary)

        all_techniques = [primary_technique] + secondary
        techniques_list = "\n".join([
            f"- {t}: {self.techniques[t].name}"
            for t in all_techniques if t in self.techniques
        ])

        log_type = self._get_log_type_for_technique(primary_info)
        scenario = random.choice(self.attack_scenarios)

        prompt = MULTI_LABEL_PROMPT.format(
            num_samples=num_samples,
            techniques_list=techniques_list,
            scenario=scenario,
            log_type=log_type,
        )

        response = self._generate_ollama(prompt)
        parsed = self._parse_samples(response)

        samples = []
        all_tactics = list(set(
            primary_info.tactics +
            sum([self.techniques[t].tactics for t in secondary if t in self.techniques], [])
        ))

        for sample_text in parsed[:num_samples]:
            samples.append({
                "text": sample_text,
                "techniques": all_techniques,
                "source": "llm_generated_multi",
                "tactics": all_tactics,
            })

        return samples

    def generate_full_dataset(
        self,
        target_samples: int = 50000,
        single_samples_per_technique: int = 60,
        multi_samples_per_technique: int = 25,
        checkpoint_every: int = 25,
    ) -> List[Dict]:
        """Generate complete dataset."""
        all_samples = []

        print(f"\nGenerating dataset:")
        print(f"  - Target: ~{target_samples} samples")
        print(f"  - Techniques: {len(self.technique_list)}")
        print(f"  - Single-label per technique: {single_samples_per_technique}")
        print(f"  - Multi-label per technique: {multi_samples_per_technique}")

        pbar = tqdm(self.technique_list, desc="Generating")

        for i, technique_id in enumerate(pbar):
            pbar.set_postfix({
                "tech": technique_id,
                "samples": len(all_samples),
                "rate": f"{len(all_samples)/(i+1):.1f}/tech"
            })

            # Generate single-technique samples
            single = self.generate_single_technique_samples(
                technique_id,
                num_samples=single_samples_per_technique
            )
            all_samples.extend(single)

            # Generate multi-technique samples
            multi = self.generate_multi_technique_samples(
                technique_id,
                num_samples=multi_samples_per_technique
            )
            all_samples.extend(multi)

            # Checkpoint
            if (i + 1) % checkpoint_every == 0:
                self._save_checkpoint(all_samples, i + 1)
                print(f"\n  Checkpoint: {len(all_samples)} samples saved")

        return all_samples

    def _save_checkpoint(self, samples: List[Dict], tech_count: int):
        """Save checkpoint."""
        path = self.output_dir / f"checkpoint_{tech_count}_techniques.jsonl"
        with open(path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

    def save_dataset(
        self,
        samples: List[Dict],
        train_ratio: float = 0.85,
        val_ratio: float = 0.10,
    ):
        """Save dataset with train/val/test splits."""
        random.shuffle(samples)

        n = len(samples)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train = samples[:train_end]
        val = samples[train_end:val_end]
        test = samples[val_end:]

        for name, data in [("train", train), ("val", val), ("test", test)]:
            path = self.output_dir / f"{name}.jsonl"
            with open(path, "w") as f:
                for sample in data:
                    f.write(json.dumps(sample) + "\n")

        # Full dataset
        with open(self.output_dir / "full_dataset.jsonl", "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        # Statistics
        stats = self._compute_stats(samples, train, val, test)
        with open(self.output_dir / "dataset_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        print(f"\nDataset saved to {self.output_dir}")
        print(f"  - Train: {len(train)}")
        print(f"  - Val: {len(val)}")
        print(f"  - Test: {len(test)}")
        print(f"  - Total: {n}")

        return stats

    def _compute_stats(self, all_samples, train, val, test) -> Dict:
        """Compute dataset statistics."""
        from collections import Counter

        tech_counts = Counter()
        for s in all_samples:
            for t in s["techniques"]:
                tech_counts[t] += 1

        multi_label = [s for s in all_samples if len(s["techniques"]) > 1]

        return {
            "total_samples": len(all_samples),
            "train_samples": len(train),
            "val_samples": len(val),
            "test_samples": len(test),
            "unique_techniques": len(tech_counts),
            "techniques_with_100plus": sum(1 for c in tech_counts.values() if c >= 100),
            "techniques_with_50plus": sum(1 for c in tech_counts.values() if c >= 50),
            "multi_label_samples": len(multi_label),
            "multi_label_ratio": len(multi_label) / len(all_samples) if all_samples else 0,
            "avg_samples_per_technique": sum(tech_counts.values()) / len(tech_counts) if tech_counts else 0,
            "min_samples": min(tech_counts.values()) if tech_counts else 0,
            "max_samples": max(tech_counts.values()) if tech_counts else 0,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.1:70b")
    parser.add_argument("--kg_dir", default="../data/attack_framework")
    parser.add_argument("--output_dir", default="../data/generated")
    parser.add_argument("--target_samples", type=int, default=50000)
    parser.add_argument("--single_per_tech", type=int, default=60)
    parser.add_argument("--multi_per_tech", type=int, default=25)
    parser.add_argument("--checkpoint_every", type=int, default=25)
    parser.add_argument("--ollama_url", default="http://localhost:11434")
    args = parser.parse_args()

    generator = OllamaDatasetGenerator(
        model_name=args.model,
        kg_dir=args.kg_dir,
        output_dir=args.output_dir,
        ollama_url=args.ollama_url,
    )

    samples = generator.generate_full_dataset(
        target_samples=args.target_samples,
        single_samples_per_technique=args.single_per_tech,
        multi_samples_per_technique=args.multi_per_tech,
        checkpoint_every=args.checkpoint_every,
    )

    stats = generator.save_dataset(samples)

    print("\n" + "="*50)
    print("Dataset Generation Complete!")
    print("="*50)
    print(f"Total: {stats['total_samples']} samples")
    print(f"Techniques covered: {stats['unique_techniques']}")
    print(f"Multi-label: {stats['multi_label_samples']} ({stats['multi_label_ratio']*100:.1f}%)")


if __name__ == "__main__":
    main()
