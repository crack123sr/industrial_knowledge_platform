"""
Knowledge Graph Builder
Constructs relationships between equipment, failures, procedures, and compliance rules.
"""
import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import networkx as nx

from config import settings

class KnowledgeGraphBuilder:
    """Build and query the industrial knowledge graph"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entity_index = defaultdict(list)  # Fast lookup by entity type
        self.relationship_index = defaultdict(list)
        self.load_or_create()
    
    def load_or_create(self):
        """Load existing graph or create a new empty one"""
        kg_path = Path(settings.KG_STORAGE_PATH)
        
        if kg_path.exists():
            print(f"📊 Loading existing knowledge graph from {kg_path}")
            try:
                with open(kg_path, 'rb') as f:
                    data = pickle.load(f)
                    self.graph = data['graph']
                    self.entity_index = data['entity_index']
                    self.relationship_index = data['relationship_index']
            except Exception as e:
                print(f"⚠️ Error loading graph: {e}. Starting fresh.")
                self._initialize_empty()
        else:
            print(f"🆕 Creating new knowledge graph")
            self._initialize_empty()

    def _initialize_empty(self):
        self.graph = nx.DiGraph()
        self.entity_index = defaultdict(list)
        self.relationship_index = defaultdict(list)

    def build_from_documents_folder(self):
        """Look for structured JSON files in the documents folder to build the graph"""
        print("\n🔗 Building knowledge graph from structured data...")
        docs_path = Path(settings.DOCUMENTS_PATH)
        
        # We look for specific files if they exist
        equipments_file = docs_path / "equipments.json"
        if equipments_file.exists():
            with open(equipments_file, 'r') as f:
                self._add_equipments(json.load(f))
                
        failures_file = docs_path / "failures.json"
        if failures_file.exists():
            with open(failures_file, 'r') as f:
                self._add_failures(json.load(f))

        workorders_file = docs_path / "workorders.json"
        if workorders_file.exists():
            with open(workorders_file, 'r') as f:
                self._add_maintenance_records(json.load(f))

        if self.graph.number_of_nodes() == 0:
            print("⚠️ No structured JSON data found. Generating a sample Industrial Graph for the demo...")
            self.create_sample_graph()

        self.save()
        self.print_summary()

    def _add_equipments(self, equipments: List[Dict]):
        for eq in equipments:
            node_id = eq.get("equipment_id", eq.get("id"))
            if not node_id: continue
            
            self.graph.add_node(
                node_id,
                node_type="Equipment",
                name=eq.get("name", "Unknown"),
                type=eq.get("type", "Unknown"),
                location=eq.get("location", "Unknown"),
                criticality=eq.get("criticality", "Medium")
            )
            self.entity_index["Equipment"].append(node_id)

    def _add_failures(self, failures: List[Dict]):
        for failure in failures:
            node_id = failure.get("failure_id")
            eq_id = failure.get("equipment_id")
            if not node_id or not eq_id: continue
            
            self.graph.add_node(
                node_id,
                node_type="Failure",
                failure_mode=failure.get("failure_mode", "Unknown"),
                root_cause=failure.get("root_cause", "Unknown")
            )
            self.entity_index["Failure"].append(node_id)
            
            self.graph.add_edge(eq_id, node_id, relationship="experienced_failure")
            self.graph.add_edge(node_id, eq_id, relationship="occurred_on")

    def _add_maintenance_records(self, records: List[Dict]):
        for record in records:
            node_id = record.get("work_order_id")
            eq_id = record.get("equipment_id")
            if not node_id or not eq_id: continue
            
            self.graph.add_node(
                node_id,
                node_type="WorkOrder",
                maintenance_type=record.get("maintenance_type", "Routine"),
                technician=record.get("technician", "Unknown")
            )
            self.entity_index["WorkOrder"].append(node_id)
            
            self.graph.add_edge(eq_id, node_id, relationship="maintained_by")
            self.graph.add_edge(node_id, eq_id, relationship="maintains")

    def get_equipment_context(self, equipment_id: str) -> Dict[str, Any]:
        """Retrieve all historical context for an equipment (Used by the AI Agent)"""
        # Normalize search (uppercase)
        equipment_id = equipment_id.strip().upper()
        
        # Check if node exists (case-insensitive search)
        found_id = None
        for node in self.graph.nodes():
            if str(node).upper() == equipment_id:
                found_id = node
                break
                
        if not found_id:
            return {}
        
        eq_node = self.graph.nodes[found_id]
        context = {
            "equipment_details": eq_node,
            "failures": [],
            "maintenance_history": [],
            "upstream_downstream_connections": []
        }
        
        # Get all related nodes connected to this equipment
        for neighbor_id in list(self.graph.successors(found_id)) + list(self.graph.predecessors(found_id)):
            node_data = self.graph.nodes[neighbor_id]
            node_type = node_data.get("node_type", "Unknown")
            
            if node_type == "Failure":
                context["failures"].append({
                    "id": neighbor_id,
                    "mode": node_data.get("failure_mode"),
                    "cause": node_data.get("root_cause")
                })
            elif node_type == "WorkOrder":
                context["maintenance_history"].append({
                    "id": neighbor_id,
                    "type": node_data.get("maintenance_type"),
                    "technician": node_data.get("technician")
                })
            elif node_type == "Equipment":
                # Get the relationship type
                edge_data = self.graph.get_edge_data(found_id, neighbor_id) or self.graph.get_edge_data(neighbor_id, found_id)
                rel = edge_data.get("relationship", "connected_to") if edge_data else "connected_to"
                context["upstream_downstream_connections"].append(f"{rel} {neighbor_id}")
                
        return context
    
    def save(self):
        """Persist graph to disk"""
        path = Path(settings.KG_STORAGE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "graph": self.graph,
            "entity_index": dict(self.entity_index),
            "relationship_index": dict(self.relationship_index),
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Knowledge graph saved to {path}")

    def print_summary(self):
        print(f"\n📊 Knowledge Graph Summary:")
        print(f"  Total Nodes: {self.graph.number_of_nodes()}")
        print(f"  Total Edges: {self.graph.number_of_edges()}")
        for entity_type, nodes in self.entity_index.items():
            print(f"    - {entity_type}: {len(nodes)} nodes")

    def create_sample_graph(self):
        """Generates a demo graph so the hackathon prototype works out of the box"""
        # 1. Create Assets
        assets = [
            {"equipment_id": "P-202A", "name": "Main Feed Pump", "type": "Centrifugal Pump", "location": "Hydrocracker Unit 2"},
            {"equipment_id": "V-101", "name": "Control Valve", "type": "Globe Valve", "location": "Hydrocracker Unit 2"},
            {"equipment_id": "C-301", "name": "High Pressure Compressor", "type": "Compressor", "location": "Gas Plant"}
        ]
        self._add_equipments(assets)

        # 2. Link them (Topology)
        self.graph.add_edge("P-202A", "V-101", relationship="feeds_into")
        self.graph.add_edge("V-101", "C-301", relationship="upstream_of")

        # 3. Add Historical Failures
        failures = [
            {"failure_id": "F-9012", "equipment_id": "P-202A", "failure_mode": "Seal Leak", "root_cause": "O-Ring Degradation due to high temp"},
            {"failure_id": "F-8831", "equipment_id": "V-101", "failure_mode": "Stuck open", "root_cause": "Actuator diaphragm failure"}
        ]
        self._add_failures(failures)

        # 4. Add Maintenance Records
        work_orders = [
            {"work_order_id": "WO-4821", "equipment_id": "P-202A", "maintenance_type": "Emergency Repair", "technician": "Ramesh Kumar"},
            {"work_order_id": "WO-5002", "equipment_id": "V-101", "maintenance_type": "Preventive", "technician": "Suresh Singh"}
        ]
        self._add_maintenance_records(work_orders)
        
        # 5. Add Compliance Node
        self.graph.add_node("PESO-Form4", node_type="ComplianceDoc", regulation="PESO Safety Clearance", status="Required for modification")
        self.graph.add_edge("V-101", "PESO-Form4", relationship="subject_to")


# ==========================================
# Run this file directly to test the KG
# ==========================================
if __name__ == "__main__":
    builder = KnowledgeGraphBuilder()
    builder.build_from_documents_folder()
    
    # Test retrieving context for our demo pump
    print("\n-------------------------------------------")
    print("Testing Graph Traversal for Asset 'P-202A':")
    context = builder.get_equipment_context("P-202A")
    print(json.dumps(context, indent=2, default=str))