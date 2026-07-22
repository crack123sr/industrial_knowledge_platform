"""
Production Industrial Knowledge Graph Builder
Builds multi-relational graphs exclusively from provided structured data.
"""

import os
import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import networkx as nx

from config import settings


class KnowledgeGraphBuilder:
    """Universal Industrial Knowledge Graph Engine"""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.entity_index = defaultdict(list)        # Lookup by Entity Type
        self.relationship_index = defaultdict(list)  # Lookup by Edge Type
        self.load_or_create()

    def load_or_create(self):
        """Load existing graph or initialize empty graph"""
        kg_path = Path(settings.KG_STORAGE_PATH)

        if kg_path.exists():
            print(f"📊 Loading existing Knowledge Graph from: {kg_path}")
            try:
                with open(kg_path, 'rb') as f:
                    data = pickle.load(f)
                    self.graph = data['graph']
                    self.entity_index = defaultdict(list, data.get('entity_index', {}))
                    self.relationship_index = defaultdict(list, data.get('relationship_index', {}))
            except Exception as e:
                print(f"⚠️ Error loading stored graph ({e}). Initializing fresh graph.")
                self._initialize_empty()
        else:
            print(f"🆕 Initializing new empty Knowledge Graph")
            self._initialize_empty()

    def _initialize_empty(self):
        self.graph = nx.DiGraph()
        self.entity_index = defaultdict(list)
        self.relationship_index = defaultdict(list)

    # =========================================================================
    # UNIVERSAL INGESTION API
    # =========================================================================

    def add_entity(self, entity_id: str, entity_type: str, properties: Optional[Dict[str, Any]] = None):
        """Add any arbitrary industrial entity to the graph."""
        clean_id = str(entity_id).strip()
        props = properties or {}
        props["node_type"] = entity_type

        self.graph.add_node(clean_id, **props)
        if clean_id not in self.entity_index[entity_type]:
            self.entity_index[entity_type].append(clean_id)

    def add_relation(self, source_id: str, target_id: str, relation_type: str, properties: Optional[Dict[str, Any]] = None):
        """Connect two entities with a typed directed edge."""
        src = str(source_id).strip()
        tgt = str(target_id).strip()
        props = properties or {}
        props["relationship"] = relation_type

        self.graph.add_edge(src, tgt, **props)
        self.relationship_index[relation_type].append((src, tgt))

    def ingest_manual_metadata(self, metadata_list: List[Dict[str, Any]]):
        """
        Generic ingestion method that accepts parsed manual entities:
        - Components, Maintenance, Troubleshooting, Safety, Specs.
        """
        for item in metadata_list:
            item_type = item.get("type", "General")
            item_id = item.get("id") or item.get("name")
            if not item_id:
                continue

            self.add_entity(item_id, item_type, item.get("properties", {}))

            # Add relations if present
            for rel in item.get("relations", []):
                self.add_relation(item_id, rel["target"], rel["relation_type"])

    def build_from_documents_folder(self):
        """Scan documents folder for structured JSON files to build the graph"""
        print("\n🔗 Building Knowledge Graph from structured JSON data...")
        docs_path = Path(settings.DOCUMENTS_PATH)

        found_any = False
        json_files = list(docs_path.glob("*.json"))

        for jf in json_files:
            try:
                with open(jf, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.ingest_manual_metadata(data)
                        found_any = True
            except Exception as e:
                print(f"⚠️ Failed to parse {jf.name}: {e}")

        if not found_any:
            print("⚠️ No structured JSON graph data found. Graph remains empty.")

        self.save()
        self.print_summary()

    # =========================================================================
    # CONTEXT RETRIEVAL FOR LLM RAG AGENT
    # =========================================================================

    def search_entities(self, query: str) -> List[str]:
        """Fuzzy keyword search to match user query against Graph Nodes"""
        query_words = re.findall(r'\w+', query.lower())
        matched_nodes = []

        for node, data in self.graph.nodes(data=True):
            node_str = f"{node} {data.get('name', '')} {data.get('node_type', '')}".lower()
            if any(word in node_str for word in query_words if len(word) > 2):
                matched_nodes.append(node)

        return matched_nodes

    def get_equipment_context(self, entity_id: str) -> Dict[str, Any]:
        """
        Broad graph traversal for any entity (Equipment, Component, Symptom, etc.)
        Retrieves all 1-hop and 2-hop connected operational intelligence.
        """
        target_id = entity_id.strip()

        # Case-insensitive match check
        found_id = None
        for node in self.graph.nodes():
            if str(node).upper() == target_id.upper():
                found_id = node
                break

        if not found_id:
            # Try fuzzy search
            matches = self.search_entities(target_id)
            if matches:
                found_id = matches[0]

        if not found_id:
            return {}

        node_data = dict(self.graph.nodes[found_id])
        context = {
            "entity": found_id,
            "details": node_data,
            "components": [],
            "maintenance_routines": [],
            "troubleshooting": [],
            "safety_and_ppe": [],
            "specifications": [],
            "connected_relations": []
        }

        # Traverse outgoing and incoming edges
        neighbors = set(list(self.graph.successors(found_id)) + list(self.graph.predecessors(found_id)))

        for n_id in neighbors:
            n_data = self.graph.nodes[n_id]
            n_type = n_data.get("node_type", "Unknown")

            # Determine relation type
            edge_data = self.graph.get_edge_data(found_id, n_id) or self.graph.get_edge_data(n_id, found_id)
            rel = edge_data.get("relationship", "connected_to") if edge_data else "connected_to"

            rel_summary = f"{found_id} --[{rel}]--> {n_id} ({n_type})"
            context["connected_relations"].append(rel_summary)

            if n_type == "Component":
                context["components"].append({"id": n_id, "name": n_data.get("name", n_id)})
            elif n_type in ["MaintenanceTask", "Lubrication"]:
                context["maintenance_routines"].append({
                    "task": n_id,
                    "frequency": n_data.get("frequency", "Periodic"),
                    "lubricant": n_data.get("lubricant", "N/A")
                })
            elif n_type in ["Problem", "Solution", "Troubleshooting"]:
                context["troubleshooting"].append({
                    "issue_or_solution": n_id,
                    "cause": n_data.get("cause", "N/A"),
                    "action": n_data.get("action", "N/A")
                })
            elif n_type in ["Safety", "PPE", "Warning"]:
                context["safety_and_ppe"].append({"rule_or_gear": n_id, "detail": n_data.get("detail", "")})
            elif n_type == "Specification":
                context["specifications"].append({"spec": n_id, "value": n_data.get("value", "")})

        return context

    # =========================================================================
    # PERSISTENCE AND SUMMARY
    # =========================================================================

    def save(self):
        """Persist graph to disk file"""
        path = Path(settings.KG_STORAGE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "graph": self.graph,
            "entity_index": dict(self.entity_index),
            "relationship_index": dict(self.relationship_index),
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Knowledge Graph successfully saved to: {path}")

    def print_summary(self):
        """Print graph structural statistics"""
        print(f"\n📊 Industrial Knowledge Graph Summary:")
        print(f"  Total Nodes: {self.graph.number_of_nodes()}")
        print(f"  Total Edges: {self.graph.number_of_edges()}")
        for entity_type, nodes in self.entity_index.items():
            print(f"    - {entity_type}: {len(nodes)} nodes")


# ==========================================
# Script Execution & Direct Testing
# ==========================================
if __name__ == "__main__":
    builder = KnowledgeGraphBuilder()
    builder.build_from_documents_folder()