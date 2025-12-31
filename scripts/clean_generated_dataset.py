#!/usr/bin/env python3
"""
Clean and validate the LLM-generated ATT&CK technique dataset.

Addresses:
1. Remove markdown artifacts (Sample markers, --- separators)
2. Fix mixed OS path inconsistencies
3. Verify technique-to-log accuracy
4. Report statistics and quality metrics
"""

import json
import re
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple


def clean_markdown_artifacts(text: str) -> str:
    """Remove markdown formatting artifacts from generated text."""
    # Remove **Sample N** markers
    text = re.sub(r'\*\*Sample\s*\d+\*\*\s*', '', text)

    # Remove leading/trailing --- separators
    text = re.sub(r'^---+\s*', '', text)
    text = re.sub(r'\s*---+$', '', text)

    # Remove ```json or ``` code blocks
    text = re.sub(r'```(?:json|xml|yaml)?\s*', '', text)
    text = re.sub(r'```\s*', '', text)

    # Clean up extra whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    return text


def detect_os_from_content(text: str) -> str:
    """Detect intended OS from log content."""
    windows_indicators = [
        r'C:\\', r'Windows', r'\.exe', r'\.dll', r'HKEY_',
        r'svchost', r'powershell', r'cmd\.exe', r'S-1-5-',
        r'NT AUTHORITY', r'SYSTEM', r'Administrator'
    ]
    linux_indicators = [
        r'/usr/', r'/bin/', r'/etc/', r'/home/', r'/var/',
        r'/tmp/', r'\.sh', r'bash', r'sudo', r'root:'
    ]
    macos_indicators = [
        r'/Applications/', r'/Library/', r'\.app/', r'macOS',
        r'Darwin', r'/Users/'
    ]

    windows_score = sum(1 for p in windows_indicators if re.search(p, text, re.IGNORECASE))
    linux_score = sum(1 for p in linux_indicators if re.search(p, text))
    macos_score = sum(1 for p in macos_indicators if re.search(p, text))

    if windows_score > linux_score and windows_score > macos_score:
        return 'windows'
    elif linux_score > macos_score:
        return 'linux'
    elif macos_score > 0:
        return 'macos'
    return 'unknown'


def fix_mixed_paths(text: str) -> Tuple[str, bool]:
    """
    Fix mixed OS paths in logs.
    Returns (fixed_text, was_modified).
    """
    detected_os = detect_os_from_content(text)
    was_modified = False

    if detected_os == 'windows':
        # Fix Linux paths that shouldn't be in Windows logs
        # Only fix obvious mismatches in Windows-context logs
        if 'Sysmon' in text or 'Windows' in text:
            # Don't modify paths that are part of command lines (could be WSL/remote)
            # Just flag for review
            if re.search(r'Image[:\s]*=?\s*/usr/', text):
                was_modified = True  # Flag but don't auto-fix

    return text, was_modified


def check_technique_accuracy(text: str, techniques: List[str]) -> Dict:
    """
    Check if the log content is relevant to the labeled techniques.
    Returns accuracy assessment.
    """
    # Technique indicators (simplified - expand as needed)
    technique_keywords = {
        'T1001': ['junk', 'encoded', 'obfuscated', 'steganograph'],
        'T1003': ['lsass', 'credential', 'password', 'hash', 'dump', 'sam', 'ntds', 'secretsdump'],
        'T1021': ['remote', 'rdp', 'ssh', 'smb', 'winrm', 'psexec', 'lateral'],
        'T1027': ['obfuscat', 'encod', 'pack', 'encrypt', 'base64', 'xor'],
        'T1036': ['masquerad', 'rename', 'spoof', 'impersonat'],
        'T1047': ['wmi', 'wmic', 'cimv2'],
        'T1048': ['exfiltrat', 'upload', 'transfer', 'dns tunnel'],
        'T1053': ['schtask', 'cron', 'at ', 'scheduled'],
        'T1055': ['inject', 'hollowing', 'dll', 'process'],
        'T1059': ['powershell', 'cmd', 'bash', 'python', 'script', 'command'],
        'T1071': ['http', 'dns', 'smtp', 'protocol'],
        'T1090': ['proxy', 'tor', 'relay'],
        'T1098': ['account', 'permission', 'group', 'add user'],
    }

    text_lower = text.lower()
    matches = []

    for tech in techniques:
        # Get base technique (T1XXX from T1XXX.YYY)
        base_tech = tech.split('.')[0]

        if base_tech in technique_keywords:
            keywords = technique_keywords[base_tech]
            found = [kw for kw in keywords if kw in text_lower]
            if found:
                matches.append((tech, found))

    return {
        'has_relevant_keywords': len(matches) > 0,
        'matches': matches,
        'techniques_checked': len(techniques)
    }


def clean_dataset(input_path: str, output_path: str) -> Dict:
    """Clean dataset and return statistics."""
    stats = {
        'total_samples': 0,
        'cleaned_samples': 0,
        'markdown_cleaned': 0,
        'path_issues': 0,
        'short_samples': 0,
        'empty_removed': 0,
        'technique_accuracy': {'accurate': 0, 'uncertain': 0},
        'technique_counts': Counter(),
        'source_counts': Counter(),
        'os_distribution': Counter(),
    }

    cleaned_samples = []

    with open(input_path, 'r') as f:
        for line in f:
            stats['total_samples'] += 1
            sample = json.loads(line)

            original_text = sample['text']

            # 1. Clean markdown artifacts
            cleaned_text = clean_markdown_artifacts(original_text)
            if cleaned_text != original_text:
                stats['markdown_cleaned'] += 1

            # 2. Check for mixed paths
            cleaned_text, had_path_issue = fix_mixed_paths(cleaned_text)
            if had_path_issue:
                stats['path_issues'] += 1

            # 3. Skip empty or too short samples
            if len(cleaned_text) < 30:
                if len(cleaned_text) == 0:
                    stats['empty_removed'] += 1
                else:
                    stats['short_samples'] += 1
                continue

            # 4. Check technique accuracy
            accuracy = check_technique_accuracy(cleaned_text, sample['techniques'])
            if accuracy['has_relevant_keywords']:
                stats['technique_accuracy']['accurate'] += 1
            else:
                stats['technique_accuracy']['uncertain'] += 1

            # 5. Detect OS
            detected_os = detect_os_from_content(cleaned_text)
            stats['os_distribution'][detected_os] += 1

            # Update sample
            sample['text'] = cleaned_text
            cleaned_samples.append(sample)

            # Track distributions
            for tech in sample['techniques']:
                stats['technique_counts'][tech] += 1
            stats['source_counts'][sample.get('source', 'unknown')] += 1

    stats['cleaned_samples'] = len(cleaned_samples)

    # Write cleaned dataset
    with open(output_path, 'w') as f:
        for sample in cleaned_samples:
            f.write(json.dumps(sample) + '\n')

    return stats


def print_report(stats: Dict):
    """Print cleaning report."""
    print("\n" + "=" * 60)
    print("DATASET CLEANING REPORT")
    print("=" * 60)

    print(f"\n📊 Sample Counts:")
    print(f"   Original samples:    {stats['total_samples']}")
    print(f"   Cleaned samples:     {stats['cleaned_samples']}")
    print(f"   Removed (empty):     {stats['empty_removed']}")
    print(f"   Removed (short):     {stats['short_samples']}")

    print(f"\n🧹 Cleaning Actions:")
    print(f"   Markdown cleaned:    {stats['markdown_cleaned']}")
    print(f"   Path issues flagged: {stats['path_issues']}")

    print(f"\n✅ Technique Accuracy:")
    total = stats['technique_accuracy']['accurate'] + stats['technique_accuracy']['uncertain']
    acc_pct = stats['technique_accuracy']['accurate'] / total * 100 if total > 0 else 0
    print(f"   Accurate (keywords found): {stats['technique_accuracy']['accurate']} ({acc_pct:.1f}%)")
    print(f"   Uncertain (no keywords):   {stats['technique_accuracy']['uncertain']}")

    print(f"\n💻 OS Distribution:")
    for os_name, count in stats['os_distribution'].most_common():
        print(f"   {os_name}: {count}")

    print(f"\n📁 Source Distribution:")
    for source, count in stats['source_counts'].most_common():
        print(f"   {source}: {count}")

    print(f"\n🎯 Technique Coverage:")
    print(f"   Unique techniques: {len(stats['technique_counts'])}")

    # Distribution stats
    counts = list(stats['technique_counts'].values())
    if counts:
        print(f"   Min samples/technique: {min(counts)}")
        print(f"   Max samples/technique: {max(counts)}")
        print(f"   Avg samples/technique: {sum(counts)/len(counts):.1f}")

        # Show techniques with low coverage
        low_coverage = [(t, c) for t, c in stats['technique_counts'].items() if c < 10]
        if low_coverage:
            print(f"\n   ⚠️  Techniques with <10 samples: {len(low_coverage)}")
            for tech, count in sorted(low_coverage, key=lambda x: x[1])[:10]:
                print(f"      {tech}: {count}")


def main():
    parser = argparse.ArgumentParser(description='Clean generated ATT&CK dataset')
    parser.add_argument('--input', required=True, help='Input JSONL file')
    parser.add_argument('--output', required=True, help='Output cleaned JSONL file')
    args = parser.parse_args()

    print(f"Cleaning dataset: {args.input}")
    stats = clean_dataset(args.input, args.output)
    print_report(stats)
    print(f"\n✅ Cleaned dataset saved to: {args.output}")


if __name__ == '__main__':
    main()
