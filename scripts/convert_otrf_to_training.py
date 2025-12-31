#!/usr/bin/env python3
"""
Convert OTRF Security Datasets to SecBERT-GNN training format.

Maps dataset filenames and content to MITRE ATT&CK techniques based on:
1. Directory/filename (tactic category)
2. Attack tool/technique in filename
3. Sysmon event patterns
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Set
import argparse

# Mapping of attack tools/patterns to ATT&CK techniques
ATTACK_MAPPINGS = {
    # Credential Access
    'mimikatz': ['T1003', 'T1003.001'],  # OS Credential Dumping, LSASS Memory
    'lsass': ['T1003.001'],  # LSASS Memory
    'sam': ['T1003.002'],  # SAM
    'ntds': ['T1003.003'],  # NTDS
    'dcsync': ['T1003.006'],  # DCSync
    'lsa_secrets': ['T1003.004'],  # LSA Secrets
    'kerberos': ['T1558'],  # Steal or Forge Kerberos Tickets
    'rubeus': ['T1558.003'],  # Kerberoasting
    'vault': ['T1555'],  # Credentials from Password Stores
    'logonpasswords': ['T1003.001'],
    'powerdump': ['T1003.002'],
    'pth': ['T1550.002'],  # Pass the Hash
    
    # Defense Evasion
    'mshta': ['T1218.005'],  # Mshta
    'regsvr32': ['T1218.010'],  # Regsvr32
    'rundll32': ['T1218.011'],  # Rundll32
    'installutil': ['T1218.004'],  # InstallUtil
    'cmstp': ['T1218.003'],  # CMSTP
    'msbuild': ['T1127.001'],  # MSBuild
    'wmic': ['T1047'],  # WMI
    'bitsadmin': ['T1197'],  # BITS Jobs
    'process_herpaderping': ['T1055'],  # Process Injection
    'dll_hijack': ['T1574.001'],  # DLL Hijacking
    'injection': ['T1055'],  # Process Injection
    'psinject': ['T1055.001'],  # DLL Injection
    'mavinject': ['T1055.001'],
    'createremotethread': ['T1055.002'],
    'fodhelper': ['T1548.002'],  # Bypass UAC
    'bypassuac': ['T1548.002'],
    'eventlog': ['T1562.002'],  # Disable Event Logging
    'auditpol': ['T1562.002'],
    'netsh': ['T1562.004'],  # Disable Firewall
    'ldap_ntsecuritydescriptor': ['T1222'],  # Modify Permissions
    'monologue': ['T1557.001'],  # LLMNR/NBT-NS Poisoning
    'control_panel': ['T1218.002'],  # Control Panel
    'hh_local': ['T1218.001'],  # Compiled HTML File
    'register_cimprovider': ['T1218'],
    'wuauclt': ['T1218'],
    
    # Discovery
    'net_local': ['T1087.001'],  # Local Account Discovery
    'net_domain': ['T1087.002'],  # Domain Account Discovery
    'domain_admins': ['T1087.002', 'T1069.002'],  # Domain Groups
    'localgroup': ['T1069.001'],  # Local Groups
    'getsession': ['T1049'],  # System Network Connections
    'find_localadmin': ['T1087.002'],
    'seatbelt': ['T1082'],  # System Information Discovery
    'iexplorer_version': ['T1518'],  # Software Discovery
    'samr': ['T1087.002'],
    'sharpview': ['T1087.002', 'T1069.002'],
    
    # Execution
    'powershell': ['T1059.001'],  # PowerShell
    'vbs': ['T1059.005'],  # Visual Basic
    'launcher': ['T1059'],
    'python': ['T1059.006'],  # Python
    'httplistener': ['T1059.001'],
    
    # Lateral Movement
    'psexec': ['T1569.002', 'T1021.002'],  # Service Execution, SMB
    'smbexec': ['T1569.002', 'T1021.002'],
    'wmi_remote': ['T1047'],  # WMI
    'psremoting': ['T1021.006'],  # WinRM
    'dcom': ['T1021.003'],  # DCOM
    'schtask': ['T1053.005'],  # Scheduled Task
    'service': ['T1543.003'],  # Windows Service
    'sharpsc': ['T1569.002'],
    'sharpwmi': ['T1047'],
    'smb': ['T1021.002'],
    'wsman': ['T1021.006'],
    'event_subscription': ['T1546.003'],  # WMI Event Subscription
    'zerologon': ['T1068'],  # CVE-2020-1472
    'CVE-2020-1472': ['T1068'],
    'proxylogon': ['T1190'],  # Exploit Public-Facing Application
    'adfs': ['T1606.002'],  # SAML Tokens
    
    # Persistence
    'registry': ['T1547.001'],  # Registry Run Keys
    'run_keys': ['T1547.001'],
    'userinitmprlogonscript': ['T1037.001'],  # Logon Script
    'schtasks': ['T1053.005'],  # Scheduled Task
    'wmi_local_event': ['T1546.003'],  # WMI Event Subscription
    
    # Privilege Escalation
    'service_mod': ['T1543.003'],  # Windows Service
    'uac': ['T1548.002'],  # Bypass UAC
    
    # Collection
    'record_mic': ['T1123'],  # Audio Capture
}

# Tactic to technique prefix mapping
TACTIC_MAPPINGS = {
    'collection': ['T1119', 'T1005'],
    'credential_access': ['T1003'],
    'defense_evasion': ['T1562', 'T1070'],
    'discovery': ['T1082', 'T1083'],
    'execution': ['T1059'],
    'lateral_movement': ['T1021'],
    'persistence': ['T1547'],
    'privilege_escalation': ['T1548'],
}


def get_techniques_from_filename(filename: str) -> List[str]:
    """Extract techniques based on filename patterns."""
    techniques = []
    filename_lower = filename.lower()
    
    for pattern, techs in ATTACK_MAPPINGS.items():
        if pattern in filename_lower:
            techniques.extend(techs)
    
    return list(set(techniques))


def get_techniques_from_tactic(tactic: str) -> List[str]:
    """Get default techniques for a tactic."""
    return TACTIC_MAPPINGS.get(tactic, [])


def extract_text_from_event(event: Dict) -> str:
    """Extract readable text from a Sysmon/Windows event."""
    parts = []
    
    # Event type
    event_id = event.get('EventID', event.get('event_id', ''))
    if event_id:
        parts.append(f"[Event {event_id}]")
    
    # Process info
    image = event.get('Image', '')
    if image:
        parts.append(f"Process: {image}")
    
    cmd = event.get('CommandLine', '')
    if cmd:
        parts.append(f"CommandLine: {cmd}")
    
    parent = event.get('ParentImage', '')
    if parent:
        parts.append(f"Parent: {parent}")
    
    # Network info
    dest_ip = event.get('DestinationIp', '')
    if dest_ip:
        parts.append(f"Network: {event.get('SourceIp', '')} -> {dest_ip}:{event.get('DestinationPort', '')}")
    
    # Registry
    target_obj = event.get('TargetObject', '')
    if target_obj:
        parts.append(f"Registry: {target_obj}")
    
    # File
    target_file = event.get('TargetFilename', '')
    if target_file:
        parts.append(f"File: {target_file}")
    
    # User
    user = event.get('User', event.get('AccountName', ''))
    if user:
        parts.append(f"User: {user}")
    
    # Message (truncated)
    message = event.get('Message', '')
    if message and len(parts) < 3:
        parts.append(f"Message: {message[:200]}")
    
    return ' '.join(parts)


def process_dataset_file(filepath: Path, techniques: List[str], max_samples: int = 100) -> List[Dict]:
    """Process a single OTRF dataset file."""
    samples = []
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Try to parse as JSON lines or JSON array
        events = []
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                events.append(event)
            except json.JSONDecodeError:
                continue
        
        # Sample events
        if len(events) > max_samples:
            import random
            events = random.sample(events, max_samples)
        
        for event in events:
            text = extract_text_from_event(event)
            if text and len(text) > 50:  # Skip very short texts
                samples.append({
                    'text': text[:2000],  # Truncate
                    'techniques': techniques,
                    'source': 'otrf',
                })
    
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
    
    return samples


def main():
    parser = argparse.ArgumentParser(description='Convert OTRF datasets to training format')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with OTRF JSON files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--kg_dir', type=str, default='data/attack_framework', help='Knowledge graph directory')
    parser.add_argument('--max_samples_per_file', type=int, default=100, help='Max samples per dataset file')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load valid techniques
    valid_techniques = set()
    technique_file = Path(args.kg_dir) / 'technique_list.json'
    if technique_file.exists():
        with open(technique_file, 'r') as f:
            valid_techniques = set(json.load(f))
        print(f"Loaded {len(valid_techniques)} valid techniques")
    
    all_samples = []
    technique_counts = defaultdict(int)
    
    # Process all JSON files
    json_files = list(input_dir.glob('*.json'))
    print(f"Found {len(json_files)} dataset files")
    
    for i, filepath in enumerate(json_files):
        filename = filepath.name
        
        # Extract tactic from filename
        parts = filename.split('_')
        tactic = parts[0] if parts else ''
        
        # Get techniques
        techniques = get_techniques_from_filename(filename)
        if not techniques:
            techniques = get_techniques_from_tactic(tactic)
        
        # Filter to valid techniques
        techniques = [t for t in techniques if t in valid_techniques]
        
        if not techniques:
            print(f"[{i+1}/{len(json_files)}] Skipping {filename} - no valid techniques")
            continue
        
        print(f"[{i+1}/{len(json_files)}] Processing {filename} -> {techniques}")
        
        samples = process_dataset_file(filepath, techniques, args.max_samples_per_file)
        all_samples.extend(samples)
        
        for tech in techniques:
            technique_counts[tech] += len(samples)
    
    print(f"\nTotal samples: {len(all_samples)}")
    print(f"Unique techniques: {len(technique_counts)}")
    
    # Split dataset
    import random
    random.shuffle(all_samples)
    
    n = len(all_samples)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    train = all_samples[:train_end]
    val = all_samples[train_end:val_end]
    test = all_samples[val_end:]
    
    # Save
    for name, data in [('train', train), ('val', val), ('test', test)]:
        outpath = output_dir / f'{name}.jsonl'
        with open(outpath, 'w') as f:
            for sample in data:
                f.write(json.dumps(sample) + '\n')
        print(f"Saved {len(data)} samples to {outpath}")
    
    # Save statistics
    stats = {
        'total_samples': len(all_samples),
        'train_samples': len(train),
        'val_samples': len(val),
        'test_samples': len(test),
        'technique_counts': dict(sorted(technique_counts.items(), key=lambda x: x[1], reverse=True)),
    }
    
    with open(output_dir / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\nTop techniques:")
    for tech, count in list(stats['technique_counts'].items())[:20]:
        print(f"  {tech}: {count}")


if __name__ == '__main__':
    main()
