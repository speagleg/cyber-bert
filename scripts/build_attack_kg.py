#!/usr/bin/env python3
"""
Build MITRE ATT&CK Knowledge Graph from STIX 2.0 data.

Downloads and parses the official MITRE ATT&CK Enterprise dataset,
creating node and edge files for use with GNN training.

Usage:
    python build_attack_kg.py --output_dir data/attack_framework

    # Or with existing STIX file:
    python build_attack_kg.py \
        --stix_file data/attack_framework/enterprise-attack.json \
        --output_dir data/attack_framework
"""

import json
import argparse
import urllib.request
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MITRE ATT&CK STIX URLs
ATTACK_STIX_URL = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json"


class MITREAttackKGBuilder:
    """Build knowledge graph from MITRE ATT&CK STIX data."""

    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []
        self.node_types = defaultdict(int)
        self.edge_types = defaultdict(int)

        # Mappings
        self.stix_id_to_attack_id: Dict[str, str] = {}
        self.attack_id_to_node_idx: Dict[str, int] = {}
        self.technique_to_tactics: Dict[str, List[str]] = defaultdict(list)

    def download_stix_data(self, output_path: Path) -> Path:
        """Download MITRE ATT&CK STIX data."""
        logger.info(f"Downloading MITRE ATT&CK data from {ATTACK_STIX_URL}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(ATTACK_STIX_URL, output_path)

        logger.info(f"Downloaded to {output_path}")
        return output_path

    def load_stix_data(self, stix_path: Path) -> Dict:
        """Load STIX bundle from file."""
        logger.info(f"Loading STIX data from {stix_path}")

        with open(stix_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"Loaded {len(data.get('objects', []))} STIX objects")
        return data

    def parse_tactics(self, stix_objects: List[Dict]) -> None:
        """Parse tactics (x-mitre-tactic objects)."""
        tactics = [obj for obj in stix_objects if obj.get('type') == 'x-mitre-tactic']

        for tactic in tactics:
            tactic_id = tactic.get('x_mitre_shortname', '').upper().replace('-', '_')
            external_refs = tactic.get('external_references', [])
            mitre_ref = next((r for r in external_refs if r.get('source_name') == 'mitre-attack'), {})

            node = {
                'id': tactic_id,
                'type': 'tactic',
                'name': tactic.get('name', ''),
                'description': tactic.get('description', '')[:500] if tactic.get('description') else '',
                'stix_id': tactic.get('id', ''),
                'external_id': mitre_ref.get('external_id', ''),
                'url': mitre_ref.get('url', ''),
            }

            self.nodes[tactic_id] = node
            self.stix_id_to_attack_id[tactic.get('id', '')] = tactic_id
            self.node_types['tactic'] += 1

        logger.info(f"Parsed {len(tactics)} tactics")

    def parse_techniques(self, stix_objects: List[Dict]) -> None:
        """Parse techniques and sub-techniques (attack-pattern objects)."""
        techniques = [obj for obj in stix_objects
                     if obj.get('type') == 'attack-pattern'
                     and not obj.get('revoked', False)
                     and not obj.get('x_mitre_deprecated', False)]

        for tech in techniques:
            external_refs = tech.get('external_references', [])
            mitre_ref = next((r for r in external_refs if r.get('source_name') == 'mitre-attack'), {})
            tech_id = mitre_ref.get('external_id', '')

            if not tech_id:
                continue

            # Determine if sub-technique
            is_subtechnique = '.' in tech_id
            node_type = 'subtechnique' if is_subtechnique else 'technique'

            # Get tactics (kill chain phases)
            kill_chain = tech.get('kill_chain_phases', [])
            tactics = [phase.get('phase_name', '').upper().replace('-', '_')
                      for phase in kill_chain
                      if phase.get('kill_chain_name') == 'mitre-attack']

            node = {
                'id': tech_id,
                'type': node_type,
                'name': tech.get('name', ''),
                'description': tech.get('description', '')[:500] if tech.get('description') else '',
                'stix_id': tech.get('id', ''),
                'external_id': tech_id,
                'url': mitre_ref.get('url', ''),
                'platforms': tech.get('x_mitre_platforms', []),
                'tactics': tactics,
                'detection': tech.get('x_mitre_detection', '')[:300] if tech.get('x_mitre_detection') else '',
                'is_subtechnique': is_subtechnique,
            }

            self.nodes[tech_id] = node
            self.stix_id_to_attack_id[tech.get('id', '')] = tech_id
            self.technique_to_tactics[tech_id] = tactics
            self.node_types[node_type] += 1

        logger.info(f"Parsed {self.node_types['technique']} techniques and {self.node_types['subtechnique']} sub-techniques")

    def parse_groups(self, stix_objects: List[Dict]) -> None:
        """Parse threat groups (intrusion-set objects)."""
        groups = [obj for obj in stix_objects
                 if obj.get('type') == 'intrusion-set'
                 and not obj.get('revoked', False)]

        for group in groups:
            external_refs = group.get('external_references', [])
            mitre_ref = next((r for r in external_refs if r.get('source_name') == 'mitre-attack'), {})
            group_id = mitre_ref.get('external_id', '')

            if not group_id:
                continue

            node = {
                'id': group_id,
                'type': 'group',
                'name': group.get('name', ''),
                'description': group.get('description', '')[:500] if group.get('description') else '',
                'stix_id': group.get('id', ''),
                'external_id': group_id,
                'url': mitre_ref.get('url', ''),
                'aliases': group.get('aliases', []),
            }

            self.nodes[group_id] = node
            self.stix_id_to_attack_id[group.get('id', '')] = group_id
            self.node_types['group'] += 1

        logger.info(f"Parsed {self.node_types['group']} threat groups")

    def parse_malware(self, stix_objects: List[Dict]) -> None:
        """Parse malware families (malware objects)."""
        malware_list = [obj for obj in stix_objects
                       if obj.get('type') == 'malware'
                       and not obj.get('revoked', False)]

        for malware in malware_list:
            external_refs = malware.get('external_references', [])
            mitre_ref = next((r for r in external_refs if r.get('source_name') == 'mitre-attack'), {})
            malware_id = mitre_ref.get('external_id', '')

            if not malware_id:
                continue

            node = {
                'id': malware_id,
                'type': 'malware',
                'name': malware.get('name', ''),
                'description': malware.get('description', '')[:500] if malware.get('description') else '',
                'stix_id': malware.get('id', ''),
                'external_id': malware_id,
                'url': mitre_ref.get('url', ''),
                'platforms': malware.get('x_mitre_platforms', []),
                'aliases': malware.get('x_mitre_aliases', []),
            }

            self.nodes[malware_id] = node
            self.stix_id_to_attack_id[malware.get('id', '')] = malware_id
            self.node_types['malware'] += 1

        logger.info(f"Parsed {self.node_types['malware']} malware families")

    def parse_tools(self, stix_objects: List[Dict]) -> None:
        """Parse tools (tool objects)."""
        tools = [obj for obj in stix_objects
                if obj.get('type') == 'tool'
                and not obj.get('revoked', False)]

        for tool in tools:
            external_refs = tool.get('external_references', [])
            mitre_ref = next((r for r in external_refs if r.get('source_name') == 'mitre-attack'), {})
            tool_id = mitre_ref.get('external_id', '')

            if not tool_id:
                continue

            node = {
                'id': tool_id,
                'type': 'tool',
                'name': tool.get('name', ''),
                'description': tool.get('description', '')[:500] if tool.get('description') else '',
                'stix_id': tool.get('id', ''),
                'external_id': tool_id,
                'url': mitre_ref.get('url', ''),
                'platforms': tool.get('x_mitre_platforms', []),
            }

            self.nodes[tool_id] = node
            self.stix_id_to_attack_id[tool.get('id', '')] = tool_id
            self.node_types['tool'] += 1

        logger.info(f"Parsed {self.node_types['tool']} tools")

    def parse_mitigations(self, stix_objects: List[Dict]) -> None:
        """Parse mitigations (course-of-action objects)."""
        mitigations = [obj for obj in stix_objects
                      if obj.get('type') == 'course-of-action'
                      and not obj.get('revoked', False)]

        for mitigation in mitigations:
            external_refs = mitigation.get('external_references', [])
            mitre_ref = next((r for r in external_refs if r.get('source_name') == 'mitre-attack'), {})
            mit_id = mitre_ref.get('external_id', '')

            if not mit_id:
                continue

            node = {
                'id': mit_id,
                'type': 'mitigation',
                'name': mitigation.get('name', ''),
                'description': mitigation.get('description', '')[:500] if mitigation.get('description') else '',
                'stix_id': mitigation.get('id', ''),
                'external_id': mit_id,
                'url': mitre_ref.get('url', ''),
            }

            self.nodes[mit_id] = node
            self.stix_id_to_attack_id[mitigation.get('id', '')] = mit_id
            self.node_types['mitigation'] += 1

        logger.info(f"Parsed {self.node_types['mitigation']} mitigations")

    def parse_relationships(self, stix_objects: List[Dict]) -> None:
        """Parse relationships between objects."""
        relationships = [obj for obj in stix_objects
                        if obj.get('type') == 'relationship'
                        and not obj.get('revoked', False)]

        for rel in relationships:
            source_stix = rel.get('source_ref', '')
            target_stix = rel.get('target_ref', '')
            rel_type = rel.get('relationship_type', '')

            source_id = self.stix_id_to_attack_id.get(source_stix)
            target_id = self.stix_id_to_attack_id.get(target_stix)

            if not source_id or not target_id:
                continue

            # Map STIX relationship types to our edge types
            edge_type_map = {
                'uses': 'USES',
                'mitigates': 'MITIGATES',
                'subtechnique-of': 'SUBTECHNIQUE_OF',
                'detects': 'DETECTS',
                'attributed-to': 'ATTRIBUTED_TO',
                'targets': 'TARGETS',
                'related-to': 'RELATED_TO',
            }

            edge_type = edge_type_map.get(rel_type, 'RELATED_TO')

            edge = {
                'source': source_id,
                'target': target_id,
                'type': edge_type,
                'description': rel.get('description', '')[:200] if rel.get('description') else '',
            }

            self.edges.append(edge)
            self.edge_types[edge_type] += 1

        logger.info(f"Parsed {len(self.edges)} relationships")

    def add_tactic_technique_edges(self) -> None:
        """Add edges from tactics to their techniques."""
        for tech_id, tactics in self.technique_to_tactics.items():
            for tactic in tactics:
                if tactic in self.nodes:
                    edge = {
                        'source': tactic,
                        'target': tech_id,
                        'type': 'TACTIC_HAS_TECHNIQUE',
                        'description': '',
                    }
                    self.edges.append(edge)
                    self.edge_types['TACTIC_HAS_TECHNIQUE'] += 1

        logger.info(f"Added {self.edge_types['TACTIC_HAS_TECHNIQUE']} tactic-technique edges")

    def add_subtechnique_edges(self) -> None:
        """Add edges from techniques to their sub-techniques."""
        subtechnique_count = 0

        for node_id, node in self.nodes.items():
            if node.get('is_subtechnique'):
                # Parent technique ID is the part before the dot
                parent_id = node_id.split('.')[0]

                if parent_id in self.nodes:
                    edge = {
                        'source': parent_id,
                        'target': node_id,
                        'type': 'TECHNIQUE_HAS_SUBTECHNIQUE',
                        'description': '',
                    }
                    self.edges.append(edge)
                    subtechnique_count += 1

        self.edge_types['TECHNIQUE_HAS_SUBTECHNIQUE'] = subtechnique_count
        logger.info(f"Added {subtechnique_count} sub-technique edges")

    def build_graph(self, stix_data: Dict) -> None:
        """Build the complete knowledge graph."""
        stix_objects = stix_data.get('objects', [])

        logger.info("Building MITRE ATT&CK knowledge graph...")

        # Parse all node types
        self.parse_tactics(stix_objects)
        self.parse_techniques(stix_objects)
        self.parse_groups(stix_objects)
        self.parse_malware(stix_objects)
        self.parse_tools(stix_objects)
        self.parse_mitigations(stix_objects)

        # Parse and add edges
        self.parse_relationships(stix_objects)
        self.add_tactic_technique_edges()
        self.add_subtechnique_edges()

        # Create node index mapping
        for idx, node_id in enumerate(sorted(self.nodes.keys())):
            self.attack_id_to_node_idx[node_id] = idx

        logger.info(f"Built graph with {len(self.nodes)} nodes and {len(self.edges)} edges")

    def get_technique_list(self) -> List[str]:
        """Get list of all technique IDs (for classification labels)."""
        techniques = [
            node_id for node_id, node in self.nodes.items()
            if node['type'] in ['technique', 'subtechnique']
        ]
        return sorted(techniques)

    def save_graph(self, output_dir: Path) -> None:
        """Save knowledge graph to files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save nodes
        nodes_file = output_dir / 'graph_nodes.json'
        with open(nodes_file, 'w', encoding='utf-8') as f:
            json.dump(self.nodes, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(self.nodes)} nodes to {nodes_file}")

        # Save edges
        edges_file = output_dir / 'graph_edges.json'
        with open(edges_file, 'w', encoding='utf-8') as f:
            json.dump(self.edges, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(self.edges)} edges to {edges_file}")

        # Save node index mapping
        index_file = output_dir / 'node_to_index.json'
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(self.attack_id_to_node_idx, f, indent=2)
        logger.info(f"Saved node index mapping to {index_file}")

        # Save technique list (labels for classification)
        techniques = self.get_technique_list()
        techniques_file = output_dir / 'technique_list.json'
        with open(techniques_file, 'w', encoding='utf-8') as f:
            json.dump(techniques, f, indent=2)
        logger.info(f"Saved {len(techniques)} techniques to {techniques_file}")

        # Save statistics
        stats = {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'node_types': dict(self.node_types),
            'edge_types': dict(self.edge_types),
            'num_techniques': len(techniques),
        }
        stats_file = output_dir / 'graph_stats.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved graph statistics to {stats_file}")

        # Print summary
        logger.info("\n" + "="*50)
        logger.info("MITRE ATT&CK Knowledge Graph Summary")
        logger.info("="*50)
        logger.info(f"Total Nodes: {len(self.nodes)}")
        for node_type, count in sorted(self.node_types.items()):
            logger.info(f"  - {node_type}: {count}")
        logger.info(f"\nTotal Edges: {len(self.edges)}")
        for edge_type, count in sorted(self.edge_types.items()):
            logger.info(f"  - {edge_type}: {count}")
        logger.info(f"\nClassification Labels: {len(techniques)} techniques")
        logger.info("="*50)


def main():
    parser = argparse.ArgumentParser(description='Build MITRE ATT&CK Knowledge Graph')
    parser.add_argument('--stix_file', type=str, default=None,
                        help='Path to existing STIX JSON file (will download if not provided)')
    parser.add_argument('--output_dir', type=str, default='data/attack_framework',
                        help='Output directory for graph files')
    parser.add_argument('--download', action='store_true',
                        help='Force download even if STIX file exists')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    builder = MITREAttackKGBuilder()

    # Get STIX data
    if args.stix_file and Path(args.stix_file).exists() and not args.download:
        stix_path = Path(args.stix_file)
    else:
        stix_path = output_dir / 'enterprise-attack.json'
        if not stix_path.exists() or args.download:
            builder.download_stix_data(stix_path)

    # Load and build graph
    stix_data = builder.load_stix_data(stix_path)
    builder.build_graph(stix_data)

    # Save graph files
    builder.save_graph(output_dir)

    logger.info(f"\nKnowledge graph files saved to {output_dir}")
    logger.info("Next steps:")
    logger.info("  1. Process security logs: python scripts/process_security_logs.py")
    logger.info("  2. Train model: python scripts/train_secbert_gnn.py")


if __name__ == '__main__':
    main()
