#!/usr/bin/env python3
"""
Security Log Dataset Processor for SecBERT-GNN Training.

Processes various security log formats into standardized JSONL training data
with MITRE ATT&CK technique labels.

Supported formats:
- Sysmon logs (Windows Event Logs)
- Zeek/Bro network logs
- Suricata alerts
- Generic JSON/JSONL logs
- Mordor datasets (labeled attack simulations)
"""

import argparse
import json
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
from datetime import datetime
import random


class SecurityLogProcessor:
    """Process security logs into training data for SecBERT-GNN."""

    def __init__(
        self,
        kg_dir: str,
        output_dir: str,
        max_seq_length: int = 512,
        min_techniques: int = 1,
    ):
        self.kg_dir = Path(kg_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_seq_length = max_seq_length
        self.min_techniques = min_techniques

        # Load technique list for validation
        self.valid_techniques: Set[str] = set()
        self._load_valid_techniques()

        # Statistics
        self.stats = defaultdict(int)

    def _load_valid_techniques(self):
        """Load valid MITRE ATT&CK technique IDs."""
        technique_file = self.kg_dir / 'technique_list.json'
        if technique_file.exists():
            with open(technique_file, 'r') as f:
                self.valid_techniques = set(json.load(f))
            print(f"Loaded {len(self.valid_techniques)} valid techniques")
        else:
            print("Warning: technique_list.json not found. Run build_attack_kg.py first.")

    def normalize_technique_id(self, technique: str) -> Optional[str]:
        """Normalize technique ID to standard format (T1234 or T1234.001)."""
        if not technique:
            return None

        # Remove common prefixes
        technique = technique.upper().strip()
        technique = re.sub(r'^(ATTACK[:\-]?|MITRE[:\-]?)', '', technique)

        # Match standard format
        match = re.match(r'(T\d{4})(?:\.(\d{3}))?', technique)
        if match:
            base = match.group(1)
            sub = match.group(2)
            normalized = f"{base}.{sub}" if sub else base

            # Validate against known techniques
            if self.valid_techniques and normalized not in self.valid_techniques:
                # Try without sub-technique
                if base in self.valid_techniques:
                    return base
                return None

            return normalized

        return None

    def extract_techniques_from_text(self, text: str) -> List[str]:
        """Extract technique IDs mentioned in text."""
        techniques = []

        # Pattern for technique IDs in various formats
        patterns = [
            r'T\d{4}(?:\.\d{3})?',  # Standard: T1234 or T1234.001
            r'attack\.mitre\.org/techniques/(T\d{4}(?:/\d{3})?)',  # URLs
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                normalized = self.normalize_technique_id(match)
                if normalized:
                    techniques.append(normalized)

        return list(set(techniques))

    def process_sysmon_log(self, log: Dict) -> Optional[Dict]:
        """Process a Sysmon event log entry."""
        event_id = log.get('EventID', log.get('event_id', ''))

        # Build text representation
        parts = []

        # Event type
        event_types = {
            '1': 'Process Creation',
            '3': 'Network Connection',
            '7': 'Image Load',
            '8': 'Create Remote Thread',
            '10': 'Process Access',
            '11': 'File Create',
            '12': 'Registry Event',
            '13': 'Registry Value Set',
            '22': 'DNS Query',
        }
        event_type = event_types.get(str(event_id), f'Event {event_id}')
        parts.append(f"[{event_type}]")

        # Process info
        if 'Image' in log or 'CommandLine' in log:
            image = log.get('Image', log.get('TargetFilename', ''))
            cmd = log.get('CommandLine', '')
            if image:
                parts.append(f"Process: {image}")
            if cmd:
                parts.append(f"CommandLine: {cmd}")

        # Network info
        if 'DestinationIp' in log:
            parts.append(f"Network: {log.get('SourceIp', '')} -> {log['DestinationIp']}:{log.get('DestinationPort', '')}")

        # Registry info
        if 'TargetObject' in log:
            parts.append(f"Registry: {log['TargetObject']}")

        # User info
        if 'User' in log:
            parts.append(f"User: {log['User']}")

        # Parent process
        if 'ParentImage' in log:
            parts.append(f"Parent: {log['ParentImage']}")

        text = ' '.join(parts)

        # Extract techniques from labels or heuristics
        techniques = []

        # Check for explicit labels
        if 'techniques' in log:
            techniques = [self.normalize_technique_id(t) for t in log['techniques']]
            techniques = [t for t in techniques if t]
        elif 'mitre_attack' in log:
            techniques = [self.normalize_technique_id(t) for t in log['mitre_attack']]
            techniques = [t for t in techniques if t]

        # Apply heuristics if no labels
        if not techniques:
            techniques = self._apply_sysmon_heuristics(log)

        if not techniques or len(techniques) < self.min_techniques:
            return None

        return {
            'text': text[:self.max_seq_length * 4],  # Rough char limit
            'techniques': techniques,
            'source': 'sysmon',
            'event_id': str(event_id),
        }

    def _apply_sysmon_heuristics(self, log: Dict) -> List[str]:
        """Apply heuristics to guess techniques from Sysmon logs."""
        techniques = []
        cmd = log.get('CommandLine', '').lower()
        image = log.get('Image', '').lower()
        event_id = str(log.get('EventID', log.get('event_id', '')))

        # Process execution heuristics
        if event_id == '1':
            if 'powershell' in image:
                techniques.append('T1059.001')  # PowerShell
            if 'cmd.exe' in image:
                techniques.append('T1059.003')  # Windows Command Shell
            if 'wscript' in image or 'cscript' in image:
                techniques.append('T1059.005')  # Visual Basic
            if 'mshta' in image:
                techniques.append('T1218.005')  # Mshta
            if 'rundll32' in image:
                techniques.append('T1218.011')  # Rundll32
            if 'regsvr32' in image:
                techniques.append('T1218.010')  # Regsvr32

            # Encoded commands
            if '-enc' in cmd or '-encodedcommand' in cmd:
                techniques.append('T1027')  # Obfuscated Files

            # Download cradles
            if 'downloadstring' in cmd or 'invoke-webrequest' in cmd:
                techniques.append('T1105')  # Ingress Tool Transfer

        # Network connection heuristics
        if event_id == '3':
            port = log.get('DestinationPort', '')
            if port in ['80', '443', '8080']:
                techniques.append('T1071.001')  # Web Protocols
            if port == '53':
                techniques.append('T1071.004')  # DNS

        # Registry heuristics
        if event_id in ['12', '13']:
            target = log.get('TargetObject', '').lower()
            if 'run' in target or 'runonce' in target:
                techniques.append('T1547.001')  # Registry Run Keys
            if 'services' in target:
                techniques.append('T1543.003')  # Windows Service

        return list(set(techniques))

    def process_zeek_log(self, log: Dict) -> Optional[Dict]:
        """Process a Zeek/Bro network log entry."""
        log_type = log.get('_path', log.get('type', 'unknown'))

        parts = [f"[Zeek {log_type}]"]

        # Connection info
        if 'id.orig_h' in log:
            parts.append(f"Src: {log['id.orig_h']}:{log.get('id.orig_p', '')}")
            parts.append(f"Dst: {log.get('id.resp_h', '')}:{log.get('id.resp_p', '')}")

        # HTTP specific
        if log_type == 'http':
            parts.append(f"Method: {log.get('method', '')}")
            parts.append(f"Host: {log.get('host', '')}")
            parts.append(f"URI: {log.get('uri', '')}")
            parts.append(f"User-Agent: {log.get('user_agent', '')}")

        # DNS specific
        if log_type == 'dns':
            parts.append(f"Query: {log.get('query', '')}")
            parts.append(f"Type: {log.get('qtype_name', '')}")

        # SSL/TLS specific
        if log_type == 'ssl':
            parts.append(f"Server: {log.get('server_name', '')}")
            parts.append(f"Subject: {log.get('subject', '')}")

        text = ' '.join(parts)

        # Extract techniques
        techniques = []
        if 'techniques' in log:
            techniques = [self.normalize_technique_id(t) for t in log['techniques']]
            techniques = [t for t in techniques if t]
        else:
            techniques = self._apply_zeek_heuristics(log, log_type)

        if not techniques or len(techniques) < self.min_techniques:
            return None

        return {
            'text': text[:self.max_seq_length * 4],
            'techniques': techniques,
            'source': f'zeek_{log_type}',
        }

    def _apply_zeek_heuristics(self, log: Dict, log_type: str) -> List[str]:
        """Apply heuristics to guess techniques from Zeek logs."""
        techniques = []

        if log_type == 'http':
            uri = log.get('uri', '').lower()
            user_agent = log.get('user_agent', '').lower()

            # Web protocols
            techniques.append('T1071.001')

            # Suspicious user agents
            if 'python' in user_agent or 'curl' in user_agent or 'wget' in user_agent:
                techniques.append('T1105')  # Ingress Tool Transfer

            # Potential webshell
            if any(x in uri for x in ['.php', '.asp', '.jsp']):
                if any(x in uri for x in ['cmd=', 'exec=', 'shell=']):
                    techniques.append('T1505.003')  # Web Shell

        if log_type == 'dns':
            query = log.get('query', '').lower()

            # DNS as C2
            techniques.append('T1071.004')

            # Potential DNS tunneling (long subdomain)
            if len(query.split('.')[0]) > 50:
                techniques.append('T1572')  # Protocol Tunneling

        return list(set(techniques))

    def process_suricata_alert(self, log: Dict) -> Optional[Dict]:
        """Process a Suricata IDS alert."""
        alert = log.get('alert', {})

        parts = ['[Suricata Alert]']
        parts.append(f"Signature: {alert.get('signature', '')}")
        parts.append(f"Category: {alert.get('category', '')}")
        parts.append(f"Severity: {alert.get('severity', '')}")

        # Network info
        parts.append(f"Src: {log.get('src_ip', '')}:{log.get('src_port', '')}")
        parts.append(f"Dst: {log.get('dest_ip', '')}:{log.get('dest_port', '')}")
        parts.append(f"Proto: {log.get('proto', '')}")

        # HTTP info if available
        if 'http' in log:
            http = log['http']
            parts.append(f"HTTP: {http.get('http_method', '')} {http.get('hostname', '')}{http.get('url', '')}")

        text = ' '.join(parts)

        # Extract techniques from signature or metadata
        techniques = []

        # Check alert metadata for MITRE mappings
        metadata = alert.get('metadata', {})
        if 'mitre_attack' in metadata:
            techniques = metadata['mitre_attack']
        elif 'attack_target' in metadata:
            # Map Suricata attack targets to techniques
            pass

        # Extract from signature text
        sig_techniques = self.extract_techniques_from_text(alert.get('signature', ''))
        techniques.extend(sig_techniques)

        # Apply heuristics based on category
        if not techniques:
            techniques = self._apply_suricata_heuristics(alert)

        techniques = [self.normalize_technique_id(t) for t in techniques if t]
        techniques = [t for t in techniques if t]

        if not techniques or len(techniques) < self.min_techniques:
            return None

        return {
            'text': text[:self.max_seq_length * 4],
            'techniques': techniques,
            'source': 'suricata',
            'severity': alert.get('severity', 0),
        }

    def _apply_suricata_heuristics(self, alert: Dict) -> List[str]:
        """Apply heuristics to guess techniques from Suricata alerts."""
        techniques = []
        category = alert.get('category', '').lower()
        signature = alert.get('signature', '').lower()

        # Category-based mapping
        category_map = {
            'attempted-admin': ['T1068'],  # Exploitation for Privilege Escalation
            'attempted-user': ['T1190'],  # Exploit Public-Facing Application
            'trojan-activity': ['T1105'],  # Ingress Tool Transfer
            'policy-violation': ['T1071'],  # Application Layer Protocol
            'shellcode-detect': ['T1059'],  # Command and Scripting Interpreter
            'web-application-attack': ['T1190'],
            'attempted-recon': ['T1595'],  # Active Scanning
        }

        for cat, techs in category_map.items():
            if cat in category:
                techniques.extend(techs)

        return list(set(techniques))

    def process_mordor_dataset(self, log: Dict) -> Optional[Dict]:
        """Process Mordor labeled attack simulation data."""
        # Mordor logs come pre-labeled with techniques
        techniques = log.get('mitre_attack', [])
        if isinstance(techniques, str):
            techniques = [techniques]

        techniques = [self.normalize_technique_id(t) for t in techniques]
        techniques = [t for t in techniques if t]

        if not techniques or len(techniques) < self.min_techniques:
            return None

        # Build text from log content
        text_parts = []

        # Add log source
        source = log.get('@metadata', {}).get('log_name', 'unknown')
        text_parts.append(f"[{source}]")

        # Add relevant fields based on log type
        for key in ['message', 'CommandLine', 'ProcessName', 'Image',
                    'TargetFilename', 'TargetObject', 'DestinationIp']:
            if key in log:
                text_parts.append(f"{key}: {log[key]}")

        text = ' '.join(text_parts)

        return {
            'text': text[:self.max_seq_length * 4],
            'techniques': techniques,
            'source': 'mordor',
            'simulation': log.get('simulation_name', ''),
        }

    def process_generic_log(self, log: Dict) -> Optional[Dict]:
        """Process a generic JSON log with technique labels."""
        # Try to find text field
        text = None
        for field in ['text', 'message', 'log', 'content', 'raw']:
            if field in log:
                text = log[field]
                break

        if not text:
            # Serialize the whole log as text
            text = json.dumps(log, default=str)

        # Try to find techniques
        techniques = []
        for field in ['techniques', 'mitre_attack', 'attack_techniques', 'labels']:
            if field in log:
                tech_data = log[field]
                if isinstance(tech_data, str):
                    tech_data = [tech_data]
                techniques = [self.normalize_technique_id(t) for t in tech_data]
                techniques = [t for t in techniques if t]
                break

        if not techniques or len(techniques) < self.min_techniques:
            return None

        return {
            'text': str(text)[:self.max_seq_length * 4],
            'techniques': techniques,
            'source': log.get('source', 'generic'),
        }

    def detect_log_format(self, log: Dict) -> str:
        """Detect the format of a log entry."""
        # Sysmon
        if 'EventID' in log or 'event_id' in log:
            if 'Image' in log or 'CommandLine' in log:
                return 'sysmon'

        # Zeek
        if '_path' in log or 'id.orig_h' in log:
            return 'zeek'

        # Suricata
        if 'alert' in log and 'signature' in log.get('alert', {}):
            return 'suricata'

        # Mordor
        if 'mitre_attack' in log and '@metadata' in log:
            return 'mordor'

        return 'generic'

    def process_file(self, input_path: Path) -> List[Dict]:
        """Process a single input file."""
        processed = []

        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    log = json.loads(line)
                except json.JSONDecodeError:
                    self.stats['parse_errors'] += 1
                    continue

                # Detect format and process
                log_format = self.detect_log_format(log)
                self.stats[f'format_{log_format}'] += 1

                result = None
                if log_format == 'sysmon':
                    result = self.process_sysmon_log(log)
                elif log_format == 'zeek':
                    result = self.process_zeek_log(log)
                elif log_format == 'suricata':
                    result = self.process_suricata_alert(log)
                elif log_format == 'mordor':
                    result = self.process_mordor_dataset(log)
                else:
                    result = self.process_generic_log(log)

                if result:
                    # Add hash for deduplication
                    result['hash'] = hashlib.md5(
                        result['text'].encode()
                    ).hexdigest()[:12]
                    processed.append(result)
                    self.stats['processed'] += 1
                else:
                    self.stats['skipped_no_technique'] += 1

        return processed

    def process_directory(
        self,
        input_dir: Path,
        extensions: List[str] = ['.json', '.jsonl', '.log'],
    ) -> List[Dict]:
        """Process all files in a directory."""
        all_processed = []

        for ext in extensions:
            for file_path in input_dir.rglob(f'*{ext}'):
                print(f"Processing: {file_path}")
                processed = self.process_file(file_path)
                all_processed.extend(processed)
                print(f"  -> {len(processed)} samples")

        return all_processed

    def deduplicate(self, samples: List[Dict]) -> List[Dict]:
        """Remove duplicate samples based on text hash."""
        seen = set()
        unique = []

        for sample in samples:
            if sample['hash'] not in seen:
                seen.add(sample['hash'])
                unique.append(sample)
            else:
                self.stats['duplicates'] += 1

        return unique

    def split_dataset(
        self,
        samples: List[Dict],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split dataset into train/val/test sets."""
        random.seed(seed)
        random.shuffle(samples)

        n = len(samples)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train = samples[:train_end]
        val = samples[train_end:val_end]
        test = samples[val_end:]

        return train, val, test

    def save_jsonl(self, samples: List[Dict], output_path: Path):
        """Save samples to JSONL file."""
        with open(output_path, 'w') as f:
            for sample in samples:
                # Remove hash before saving
                sample_clean = {k: v for k, v in sample.items() if k != 'hash'}
                f.write(json.dumps(sample_clean) + '\n')

    def generate_statistics(self, samples: List[Dict]) -> Dict:
        """Generate dataset statistics."""
        stats = {
            'total_samples': len(samples),
            'sources': defaultdict(int),
            'technique_counts': defaultdict(int),
            'samples_per_technique_count': defaultdict(int),
        }

        for sample in samples:
            stats['sources'][sample['source']] += 1
            num_techniques = len(sample['techniques'])
            stats['samples_per_technique_count'][num_techniques] += 1

            for tech in sample['techniques']:
                stats['technique_counts'][tech] += 1

        # Convert defaultdicts
        stats['sources'] = dict(stats['sources'])
        stats['technique_counts'] = dict(sorted(
            stats['technique_counts'].items(),
            key=lambda x: x[1],
            reverse=True
        ))
        stats['samples_per_technique_count'] = dict(
            sorted(stats['samples_per_technique_count'].items())
        )

        return stats

    def process(
        self,
        input_paths: List[str],
        split: bool = True,
    ):
        """Main processing pipeline."""
        all_samples = []

        for input_path in input_paths:
            path = Path(input_path)
            if path.is_file():
                samples = self.process_file(path)
            elif path.is_dir():
                samples = self.process_directory(path)
            else:
                print(f"Warning: {input_path} not found")
                continue

            all_samples.extend(samples)

        print(f"\nTotal raw samples: {len(all_samples)}")

        # Deduplicate
        all_samples = self.deduplicate(all_samples)
        print(f"After deduplication: {len(all_samples)}")

        if not all_samples:
            print("No samples to process!")
            return

        # Generate statistics
        stats = self.generate_statistics(all_samples)

        # Save statistics
        stats_path = self.output_dir / 'dataset_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nSaved statistics to {stats_path}")

        # Print top techniques
        print("\nTop 20 techniques:")
        for tech, count in list(stats['technique_counts'].items())[:20]:
            print(f"  {tech}: {count}")

        # Split and save
        if split:
            train, val, test = self.split_dataset(all_samples)

            self.save_jsonl(train, self.output_dir / 'train.jsonl')
            self.save_jsonl(val, self.output_dir / 'val.jsonl')
            self.save_jsonl(test, self.output_dir / 'test.jsonl')

            print(f"\nSaved splits:")
            print(f"  Train: {len(train)} samples")
            print(f"  Val: {len(val)} samples")
            print(f"  Test: {len(test)} samples")
        else:
            self.save_jsonl(all_samples, self.output_dir / 'all_data.jsonl')
            print(f"\nSaved {len(all_samples)} samples to all_data.jsonl")

        # Print processing stats
        print("\nProcessing statistics:")
        for key, value in sorted(self.stats.items()):
            print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(
        description='Process security logs for SecBERT-GNN training'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        nargs='+',
        required=True,
        help='Input file(s) or directory(ies) containing security logs'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='data/security_logs',
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--kg_dir',
        type=str,
        default='data/attack_framework',
        help='Directory containing MITRE ATT&CK knowledge graph'
    )
    parser.add_argument(
        '--max_seq_length',
        type=int,
        default=512,
        help='Maximum sequence length for text'
    )
    parser.add_argument(
        '--min_techniques',
        type=int,
        default=1,
        help='Minimum number of techniques required per sample'
    )
    parser.add_argument(
        '--no_split',
        action='store_true',
        help='Do not split into train/val/test'
    )

    args = parser.parse_args()

    processor = SecurityLogProcessor(
        kg_dir=args.kg_dir,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        min_techniques=args.min_techniques,
    )

    processor.process(
        input_paths=args.input,
        split=not args.no_split,
    )


if __name__ == '__main__':
    main()
