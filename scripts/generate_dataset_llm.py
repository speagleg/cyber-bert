#!/usr/bin/env python3
"""
Generate comprehensive ATT&CK technique training dataset using Llama 3.1 70B.

This script generates realistic security log samples for each ATT&CK technique
using an open-source LLM, ensuring comprehensive coverage across all 596 techniques.
"""

import json
import os
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


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
    "sysmon_image_load",
    "windows_security",
    "windows_powershell",
    "linux_audit",
    "linux_syslog",
    "network_firewall",
    "edr_alert",
    "ids_alert",
]

# Templates for different log formats
LOG_FORMAT_TEMPLATES = {
    "sysmon_process_create": """EventID: 1 (Process Create)
UtcTime: {timestamp}
ProcessGuid: {{{guid}}}
ProcessId: {pid}
Image: {image_path}
CommandLine: {command_line}
CurrentDirectory: {current_dir}
User: {user}
ParentImage: {parent_image}
ParentCommandLine: {parent_cmdline}""",

    "sysmon_network_connection": """EventID: 3 (Network Connection)
UtcTime: {timestamp}
ProcessGuid: {{{guid}}}
ProcessId: {pid}
Image: {image_path}
User: {user}
Protocol: {protocol}
SourceIp: {src_ip}
SourcePort: {src_port}
DestinationIp: {dst_ip}
DestinationPort: {dst_port}
DestinationHostname: {dst_hostname}""",

    "sysmon_file_create": """EventID: 11 (File Create)
UtcTime: {timestamp}
ProcessGuid: {{{guid}}}
ProcessId: {pid}
Image: {image_path}
TargetFilename: {target_file}
CreationUtcTime: {creation_time}
User: {user}""",

    "sysmon_registry": """EventID: 13 (Registry Value Set)
UtcTime: {timestamp}
ProcessGuid: {{{guid}}}
ProcessId: {pid}
Image: {image_path}
TargetObject: {reg_key}
Details: {reg_value}
User: {user}""",

    "windows_security": """EventID: {event_id}
TimeCreated: {timestamp}
Computer: {computer}
SubjectUserName: {user}
SubjectDomainName: {domain}
TargetUserName: {target_user}
LogonType: {logon_type}
IpAddress: {ip_address}
ProcessName: {process_name}""",

    "linux_audit": """type=SYSCALL msg=audit({timestamp}): arch=c000003e syscall={syscall_num} success={success} exit={exit_code} a0={a0} a1={a1} a2={a2} a3={a3} items={items} ppid={ppid} pid={pid} auid={auid} uid={uid} gid={gid} euid={euid} suid={suid} fsuid={fsuid} egid={egid} sgid={sgid} fsgid={fsgid} tty={tty} ses={ses} comm="{comm}" exe="{exe}" key="{key}"
type=EXECVE msg=audit({timestamp}): argc={argc} {argv}
type=PATH msg=audit({timestamp}): item=0 name="{path}" inode={inode} dev={dev} mode={mode} ouid={ouid} ogid={ogid} rdev=00:00 nametype=NORMAL""",

    "edr_alert": """Alert Type: {alert_type}
Severity: {severity}
Timestamp: {timestamp}
Hostname: {hostname}
Process: {process_name}
PID: {pid}
User: {user}
Command Line: {command_line}
Parent Process: {parent_process}
File Hash (SHA256): {file_hash}
Detection: {detection_name}
MITRE ATT&CK: {mitre_ref}""",
}


GENERATION_PROMPT_TEMPLATE = """You are a cybersecurity expert generating realistic security log samples for training a threat detection model.

Generate {num_samples} realistic and diverse security log entries that would indicate the following MITRE ATT&CK technique:

**Technique ID**: {technique_id}
**Technique Name**: {technique_name}
**Description**: {description}
**Tactics**: {tactics}
**Platforms**: {platforms}

For each sample, generate a realistic security log that would be observed when this technique is executed. Include:
1. Realistic file paths, IP addresses, usernames, and process names
2. Actual command-line arguments that would be used
3. Realistic timestamps and event sequences
4. Variations in how this technique might be executed (different tools, methods, encodings)

Log Format to use: {log_format}

IMPORTANT:
- Make each sample unique and realistic
- Include both obvious and subtle indicators
- Use realistic Windows/Linux paths and processes
- Include variations (encoded commands, different tools, obfuscation)
- Do NOT include the technique ID in the log text itself

Output exactly {num_samples} log samples, each separated by "---SAMPLE---".
Each sample should be a complete, realistic log entry.

Begin generating:"""


MULTI_LABEL_PROMPT_TEMPLATE = """You are a cybersecurity expert generating realistic security log samples that demonstrate multiple MITRE ATT&CK techniques being used together.

Generate {num_samples} realistic security log entries that demonstrate the following attack chain:

**Primary Technique**: {primary_technique_id} - {primary_technique_name}
**Secondary Technique(s)**: {secondary_techniques}
**Attack Scenario**: {scenario}

The log should realistically show these techniques being used together in a single attack sequence or correlated events.

Log Format: {log_format}

IMPORTANT:
- Show realistic technique combinations that occur in real attacks
- Include command lines and artifacts that demonstrate BOTH/ALL techniques
- Make the logs realistic and varied
- Do NOT mention technique IDs in the log text

Output exactly {num_samples} log samples, each separated by "---SAMPLE---".

Begin generating:"""


class ATTACKDatasetGenerator:
    """Generate security log dataset using LLM."""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-70B-Instruct",
        kg_dir: str = "../data/attack_framework",
        output_dir: str = "../data/generated",
        device: str = "cuda",
        use_4bit: bool = True,
    ):
        self.model_name = model_name
        self.kg_dir = Path(kg_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        # Load technique information
        self.techniques = self._load_techniques()
        self.technique_list = list(self.techniques.keys())

        # Group techniques by tactic for co-occurrence generation
        self.tactic_techniques = self._group_by_tactic()

        # Load model
        print(f"Loading model: {model_name}")
        self._load_model(use_4bit)

    def _load_techniques(self) -> Dict[str, TechniqueInfo]:
        """Load technique information from knowledge graph."""
        techniques = {}

        # Load graph nodes
        with open(self.kg_dir / "graph_nodes.json", "r") as f:
            nodes = json.load(f)

        # Load technique list
        with open(self.kg_dir / "technique_list.json", "r") as f:
            technique_ids = json.load(f)

        for tech_id in technique_ids:
            if tech_id in nodes:
                node = nodes[tech_id]

                # Determine parent for subtechniques
                parent_id = None
                if "." in tech_id:
                    parent_id = tech_id.split(".")[0]

                techniques[tech_id] = TechniqueInfo(
                    id=tech_id,
                    name=node.get("name", tech_id),
                    description=node.get("description", "")[:1500],  # Truncate long descriptions
                    tactics=node.get("tactics", []),
                    platforms=node.get("platforms", ["Windows", "Linux", "macOS"]),
                    is_subtechnique=node.get("is_subtechnique", "." in tech_id),
                    parent_id=parent_id,
                )

        print(f"Loaded {len(techniques)} techniques")
        return techniques

    def _group_by_tactic(self) -> Dict[str, List[str]]:
        """Group techniques by tactic for realistic co-occurrence."""
        tactic_techniques = {}
        for tech_id, info in self.techniques.items():
            for tactic in info.tactics:
                if tactic not in tactic_techniques:
                    tactic_techniques[tactic] = []
                tactic_techniques[tactic].append(tech_id)
        return tactic_techniques

    def _load_model(self, use_4bit: bool = True):
        """Load the LLM with quantization for memory efficiency."""
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Model loaded successfully on {self.device}")

    def _generate_text(self, prompt: str, max_new_tokens: int = 2048) -> str:
        """Generate text from the LLM."""
        messages = [
            {"role": "system", "content": "You are a cybersecurity expert who generates realistic security log samples for threat detection training."},
            {"role": "user", "content": prompt}
        ]

        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response

    def _parse_samples(self, response: str) -> List[str]:
        """Parse generated samples from LLM response."""
        samples = []

        # Split by separator
        if "---SAMPLE---" in response:
            parts = response.split("---SAMPLE---")
        else:
            # Try other common separators
            parts = response.split("\n\n---\n\n")
            if len(parts) == 1:
                parts = response.split("\n---\n")
            if len(parts) == 1:
                # Assume single sample or split by double newlines
                parts = [p.strip() for p in response.split("\n\n\n") if p.strip()]

        for part in parts:
            sample = part.strip()
            if sample and len(sample) > 50:  # Filter out too-short samples
                samples.append(sample)

        return samples

    def generate_single_technique_samples(
        self,
        technique_id: str,
        num_samples: int = 10,
    ) -> List[Dict]:
        """Generate samples for a single technique."""
        if technique_id not in self.techniques:
            print(f"Warning: Unknown technique {technique_id}")
            return []

        info = self.techniques[technique_id]
        samples = []

        # Generate samples using different log formats
        samples_per_format = max(1, num_samples // len(LOG_TYPES))
        remaining = num_samples - (samples_per_format * len(LOG_TYPES))

        for i, log_format in enumerate(LOG_TYPES):
            n = samples_per_format + (1 if i < remaining else 0)
            if n <= 0:
                continue

            # Select platforms compatible with log format
            platforms = info.platforms
            if log_format.startswith("linux") and "Linux" not in platforms:
                continue
            if log_format.startswith("windows") and "Windows" not in platforms:
                continue

            prompt = GENERATION_PROMPT_TEMPLATE.format(
                num_samples=n,
                technique_id=technique_id,
                technique_name=info.name,
                description=info.description[:1000],
                tactics=", ".join(info.tactics),
                platforms=", ".join(platforms),
                log_format=log_format,
            )

            try:
                response = self._generate_text(prompt)
                parsed = self._parse_samples(response)

                for sample_text in parsed[:n]:
                    samples.append({
                        "text": sample_text,
                        "techniques": [technique_id],
                        "source": f"llm_generated_{log_format}",
                        "log_type": log_format,
                        "tactics": info.tactics,
                    })
            except Exception as e:
                print(f"Error generating for {technique_id} ({log_format}): {e}")

        return samples

    def generate_multi_technique_samples(
        self,
        primary_technique: str,
        num_samples: int = 5,
    ) -> List[Dict]:
        """Generate samples demonstrating multiple techniques together."""
        if primary_technique not in self.techniques:
            return []

        primary_info = self.techniques[primary_technique]
        samples = []

        # Find related techniques from same tactics
        related_techniques = set()
        for tactic in primary_info.tactics:
            if tactic in self.tactic_techniques:
                for tech in self.tactic_techniques[tactic]:
                    if tech != primary_technique:
                        related_techniques.add(tech)

        if not related_techniques:
            return []

        # Sample 1-3 secondary techniques
        num_secondary = min(3, len(related_techniques))
        secondary = random.sample(list(related_techniques), num_secondary)

        secondary_info = [
            f"{t} ({self.techniques[t].name})" for t in secondary if t in self.techniques
        ]

        # Create attack scenario
        scenarios = [
            "Initial compromise followed by lateral movement",
            "Credential theft and privilege escalation",
            "Data staging and exfiltration",
            "Persistence establishment after initial access",
            "Defense evasion during command execution",
            "Discovery activities following successful exploitation",
        ]

        prompt = MULTI_LABEL_PROMPT_TEMPLATE.format(
            num_samples=num_samples,
            primary_technique_id=primary_technique,
            primary_technique_name=primary_info.name,
            secondary_techniques=", ".join(secondary_info),
            scenario=random.choice(scenarios),
            log_format=random.choice(LOG_TYPES),
        )

        try:
            response = self._generate_text(prompt)
            parsed = self._parse_samples(response)

            all_techniques = [primary_technique] + secondary

            for sample_text in parsed[:num_samples]:
                samples.append({
                    "text": sample_text,
                    "techniques": all_techniques,
                    "source": "llm_generated_multi",
                    "tactics": list(set(primary_info.tactics + sum(
                        [self.techniques[t].tactics for t in secondary if t in self.techniques], []
                    ))),
                })
        except Exception as e:
            print(f"Error generating multi-technique for {primary_technique}: {e}")

        return samples

    def generate_full_dataset(
        self,
        target_samples: int = 50000,
        samples_per_technique: int = 80,
        multi_label_ratio: float = 0.3,
        checkpoint_every: int = 1000,
    ) -> List[Dict]:
        """Generate complete dataset covering all techniques."""
        all_samples = []

        # Calculate distribution
        num_techniques = len(self.technique_list)
        single_samples_per_tech = int(samples_per_technique * (1 - multi_label_ratio))
        multi_samples_per_tech = int(samples_per_technique * multi_label_ratio)

        print(f"\nGenerating dataset:")
        print(f"  - Target: {target_samples} samples")
        print(f"  - Techniques: {num_techniques}")
        print(f"  - Single-label samples per technique: {single_samples_per_tech}")
        print(f"  - Multi-label samples per technique: {multi_samples_per_tech}")

        # Progress tracking
        pbar = tqdm(self.technique_list, desc="Generating samples")

        for i, technique_id in enumerate(pbar):
            pbar.set_postfix({"technique": technique_id, "samples": len(all_samples)})

            # Generate single-technique samples
            single_samples = self.generate_single_technique_samples(
                technique_id,
                num_samples=single_samples_per_tech,
            )
            all_samples.extend(single_samples)

            # Generate multi-technique samples
            multi_samples = self.generate_multi_technique_samples(
                technique_id,
                num_samples=multi_samples_per_tech,
            )
            all_samples.extend(multi_samples)

            # Checkpoint
            if (i + 1) % checkpoint_every == 0:
                checkpoint_path = self.output_dir / f"checkpoint_{i+1}.jsonl"
                self._save_samples(all_samples, checkpoint_path)
                print(f"\nCheckpoint saved: {len(all_samples)} samples -> {checkpoint_path}")

        return all_samples

    def _save_samples(self, samples: List[Dict], path: Path):
        """Save samples to JSONL file."""
        with open(path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

    def save_dataset(
        self,
        samples: List[Dict],
        train_ratio: float = 0.85,
        val_ratio: float = 0.10,
        test_ratio: float = 0.05,
    ):
        """Save dataset with train/val/test splits."""
        # Shuffle samples
        random.shuffle(samples)

        # Calculate split indices
        n = len(samples)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_samples = samples[:train_end]
        val_samples = samples[train_end:val_end]
        test_samples = samples[val_end:]

        # Save splits
        self._save_samples(train_samples, self.output_dir / "train.jsonl")
        self._save_samples(val_samples, self.output_dir / "val.jsonl")
        self._save_samples(test_samples, self.output_dir / "test.jsonl")

        # Save full dataset
        self._save_samples(samples, self.output_dir / "full_dataset.jsonl")

        # Generate statistics
        stats = self._compute_statistics(samples, train_samples, val_samples, test_samples)
        with open(self.output_dir / "dataset_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        print(f"\nDataset saved to {self.output_dir}")
        print(f"  - Train: {len(train_samples)} samples")
        print(f"  - Val: {len(val_samples)} samples")
        print(f"  - Test: {len(test_samples)} samples")
        print(f"  - Total: {len(samples)} samples")

        return stats

    def _compute_statistics(
        self,
        all_samples: List[Dict],
        train: List[Dict],
        val: List[Dict],
        test: List[Dict],
    ) -> Dict:
        """Compute dataset statistics."""
        from collections import Counter

        def count_techniques(samples):
            counts = Counter()
            for s in samples:
                for t in s["techniques"]:
                    counts[t] += 1
            return counts

        all_counts = count_techniques(all_samples)
        train_counts = count_techniques(train)

        # Multi-label statistics
        multi_label_samples = [s for s in all_samples if len(s["techniques"]) > 1]

        return {
            "total_samples": len(all_samples),
            "train_samples": len(train),
            "val_samples": len(val),
            "test_samples": len(test),
            "unique_techniques": len(all_counts),
            "techniques_with_samples": sum(1 for c in all_counts.values() if c > 0),
            "multi_label_samples": len(multi_label_samples),
            "multi_label_ratio": len(multi_label_samples) / len(all_samples),
            "avg_techniques_per_sample": sum(len(s["techniques"]) for s in all_samples) / len(all_samples),
            "min_samples_per_technique": min(all_counts.values()) if all_counts else 0,
            "max_samples_per_technique": max(all_counts.values()) if all_counts else 0,
            "avg_samples_per_technique": sum(all_counts.values()) / len(all_counts) if all_counts else 0,
            "technique_distribution": dict(all_counts.most_common(50)),
        }


def main():
    parser = argparse.ArgumentParser(description="Generate ATT&CK dataset using LLM")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-70B-Instruct",
                       help="HuggingFace model name")
    parser.add_argument("--kg_dir", type=str, default="../data/attack_framework",
                       help="Path to knowledge graph directory")
    parser.add_argument("--output_dir", type=str, default="../data/generated",
                       help="Output directory for generated dataset")
    parser.add_argument("--target_samples", type=int, default=50000,
                       help="Target number of samples to generate")
    parser.add_argument("--samples_per_technique", type=int, default=85,
                       help="Target samples per technique")
    parser.add_argument("--multi_label_ratio", type=float, default=0.3,
                       help="Ratio of multi-label samples")
    parser.add_argument("--no_4bit", action="store_true",
                       help="Disable 4-bit quantization")
    parser.add_argument("--checkpoint_every", type=int, default=50,
                       help="Save checkpoint every N techniques")
    args = parser.parse_args()

    # Initialize generator
    generator = ATTACKDatasetGenerator(
        model_name=args.model,
        kg_dir=args.kg_dir,
        output_dir=args.output_dir,
        use_4bit=not args.no_4bit,
    )

    # Generate dataset
    samples = generator.generate_full_dataset(
        target_samples=args.target_samples,
        samples_per_technique=args.samples_per_technique,
        multi_label_ratio=args.multi_label_ratio,
        checkpoint_every=args.checkpoint_every,
    )

    # Save dataset
    stats = generator.save_dataset(samples)

    print("\n" + "="*50)
    print("Dataset Generation Complete!")
    print("="*50)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Techniques covered: {stats['techniques_with_samples']}/{stats['unique_techniques']}")
    print(f"Multi-label samples: {stats['multi_label_samples']} ({stats['multi_label_ratio']*100:.1f}%)")
    print(f"Avg samples per technique: {stats['avg_samples_per_technique']:.1f}")


if __name__ == "__main__":
    main()
