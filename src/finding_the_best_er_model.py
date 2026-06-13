# import libraries

import warnings
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv", category=RuntimeWarning)
import ssl
from openie import StanfordOpenIE
from datetime import datetime
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from collections import defaultdict
from openie import StanfordOpenIE
from g4f.client import Client
from typing import List
import os
from groq import Groq
import json
import re
import networkx as nx
import math
from pyvis.network import Network
import numpy as np
import math
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
import pulp
import math
import time
from typing import List, Tuple, Dict
import math
import matplotlib.pyplot as plt


# er_ilp_logodds_class
class ER_ILP_LogOdds_Joint:
    """
    ER selection using Pure ILP + Log-Odds with joint probability over
    entities, relationships, and attributes, including penalty for attributes.
    """

    def __init__(
        self,
        entity_probs: Dict[str, float],
        relation_rows: List[Dict[str, float]],
        attribute_probs: Dict[str, List[Tuple[str, float]]]
    ):
        """
        Args:
            entity_probs: {entity: probability}
            relation_rows: [{"e1": str, "e2": str, "p": float}]
            attribute_probs: {entity: [(attr_name, prob), ...]}
        """
        self.entities = entity_probs
        self.relations = [(r["e1"], r["e2"], r["p"]) for r in relation_rows]
        self.attributes = attribute_probs  # keep as is

    @staticmethod
    def log_odds(p: float) -> float:
        eps = 1e-12
        p = max(min(p, 1.0 - eps), eps)
        return math.log(p / (1.0 - p))

    def solve(
        self,
        lambda_E: float = 0.3,
        lambda_R: float = 0.3,
        lambda_A: float = 0.3,
        time_limit: int = 10
    ):
        start_time = time.time()
        prob = pulp.LpProblem("ER_ILP_LogOdds_Joint", pulp.LpMaximize)

        # Binary variables
        xE = pulp.LpVariable.dicts("E", self.entities.keys(), cat="Binary")
        xR = pulp.LpVariable.dicts("R", [(s, t) for s, t, _ in self.relations], cat="Binary")
        xA = {}
        for e, attrs in self.attributes.items():
            for attr_name, _ in attrs:
                xA[(e, attr_name)] = pulp.LpVariable(f"A_{e}_{attr_name}", cat="Binary")

        # Objective: joint log-odds - penalties
        prob += (
            pulp.lpSum(self.log_odds(self.entities[e]) * xE[e] for e in self.entities) +
            pulp.lpSum(self.log_odds(p) * xR[(s, t)] for s, t, p in self.relations) +
            pulp.lpSum(self.log_odds(p_attr) * xA[(e, attr_name)]
                       for e, attrs in self.attributes.items()
                       for attr_name, p_attr in attrs) -
            lambda_E * pulp.lpSum(xE.values()) -
            lambda_R * pulp.lpSum(xR.values()) -
            lambda_A * pulp.lpSum(xA.values())
        )

        # Constraints
        for s, t, p in self.relations:
            prob += xR[(s, t)] <= xE[s]
            prob += xR[(s, t)] <= xE[t]
            if p > min(self.entities.get(s, 0.0), self.entities.get(t, 0.0)):
                prob += xR[(s, t)] == 0

        for e in self.entities:
            incident = [(s, t) for s, t, _ in self.relations if e in (s, t)]
            if incident:
                prob += pulp.lpSum(xR[r] for r in incident) >= xE[e]
            else:
                prob += xE[e] == 0

        for (e, attr_name), var in xA.items():
            prob += var <= xE[e]  # attribute only if entity is selected

        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))
        runtime = time.time() - start_time
        score = pulp.value(prob.objective)

        selected_entities = [e for e in self.entities if xE[e].value() > 0.9]
        selected_relations = [(s, t) for (s, t) in xR if xR[(s, t)].value() > 0.9]
        selected_attributes = [(e, attr_name) for (e, attr_name) in xA if xA[(e, attr_name)].value() > 0.9]

        return score, selected_entities, selected_relations, selected_attributes, runtime

class JointERILPSolverFinal:
    def __init__(
        self,
        entity_probs: Dict[str, float],
        relation_rows: List[Dict[str, float]],
        attribute_probs: Dict[str, List[Tuple[str, float]]]
    ):
        """
        Args:
            entity_probs: {entity: probability}
            relation_rows: [{"e1": str, "e2": str, "p": float}]
            attribute_probs: {entity: [(attr_name, prob), ...]}
        """
        self.entities = entity_probs
        self.relations = [(r["e1"], r["e2"], r["p"]) for r in relation_rows]
        self.attributes = attribute_probs  # keep as is


    @staticmethod
    def log_odds(p, eps=1e-6):
        """
        log( p / (1 - p) )
        """
        p = float(p)
        p = max(min(p, 1 - eps), eps)
        return math.log(p / (1 - p))

    def solve(
        self,
        lambda_E=0.1,
        lambda_A=0.1,
        lambda_R=0.1,
        min_entities=2,
        no_isolated=False,
    ):
        start_time = time.time()

        # -------------------------
        # Problem
        # -------------------------
        prob = pulp.LpProblem("Optimal_ER_Model_Selection", pulp.LpMaximize)

        # -------------------------
        # Decision variables
        # -------------------------
        # Entities
        xE = {
            e: pulp.LpVariable(f"xE_{e}", cat="Binary")
            for e in self.entities
        }

        # Attributes
        zA = {}
        for e, attrs in self.attributes.items():
            for attr, _ in attrs:
                zA[(e, attr)] = pulp.LpVariable(
                    f"zA_{e}_{attr}", cat="Binary"
                )

        # Relationships
        yR = {}
        for s, t, _ in self.relations:
            yR[(s, t)] = pulp.LpVariable(
                f"yR_{s}_{t}", cat="Binary"
            )

        # -------------------------
        # Objective function (EXACTLY as in paper)
        # -------------------------
        prob += (
            # Entities
            pulp.lpSum(
                xE[e] * self.log_odds(self.entities[e])
                for e in self.entities
            )
            # Attributes
            + pulp.lpSum(
                zA[(e, attr)] * self.log_odds(p)
                for e, attrs in self.attributes.items()
                for attr, p in attrs
            )
            # Relationships
            + pulp.lpSum(
                yR[(s, t)] * self.log_odds(p)
                for s, t, p in self.relations
            )
            # Penalties
            - lambda_E * pulp.lpSum(xE.values())
            - lambda_A * pulp.lpSum(zA.values())
            - lambda_R * pulp.lpSum(yR.values())
        )

        # -------------------------
        # Constraints
        # -------------------------
        # Minimum number of entities
        prob += pulp.lpSum(xE.values()) >= min_entities

        # Relationship ⇒ entities must exist
        for s, t, _ in self.relations:
            prob += yR[(s, t)] <= xE[s]
            prob += yR[(s, t)] <= xE[t]

        # Entity ⇒ must participate in at least one relationship (no isolated entities)
        if no_isolated:
            incident = {e: [] for e in self.entities}
            for s, t, _ in self.relations:
                incident[s].append(yR[(s, t)])
                incident[t].append(yR[(s, t)])
            for e in self.entities:
                if incident[e]:
                    prob += xE[e] <= pulp.lpSum(incident[e])

        # Attribute ⇒ entity must exist
        for e, attrs in self.attributes.items():
            for attr, _ in attrs:
                prob += zA[(e, attr)] <= xE[e]

        # -------------------------
        # Solve
        # -------------------------
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        runtime = time.time() - start_time

        # -------------------------
        # Extract solution
        # -------------------------
        selected_entities = [
            e for e in xE if pulp.value(xE[e]) > 0.5
        ]

        selected_attributes = {
            e: [
                attr for attr, _ in self.attributes[e]
                if pulp.value(zA[(e, attr)]) > 0.5
            ]
            for e in selected_entities
            if e in self.attributes
        }

        selected_relations = [
            (s, t)
            for s, t, _ in self.relations
            if pulp.value(yR[(s, t)]) > 0.5
        ]

        score = pulp.value(prob.objective)

        return (
            score,
            selected_entities,
            selected_relations,
            selected_attributes,
            runtime
        )
def normalize_attribute_name(attr_name: str) -> str:
    """Normalize attribute name for comparison"""
    # Convert to lowercase, remove underscores, keep only alphanumeric
    attr = str(attr_name).lower().replace('_', '').strip()
    # Remove common prefixes/suffixes
    for prefix in ['id', 'fk', 'pk', 'ref']:
        if attr.startswith(prefix):
            attr = attr[len(prefix):]
    return attr

def postprocess_associative_entities(entities, attributes, relations):
    new_entities = list(entities)
    new_attributes = {e: list(attrs) for e, attrs in attributes.items()}
    new_relations = []

    for rel in relations:
        card = rel.get("cardinality")
        assoc_obj = rel.get("associative_entity")
        e1 = rel["entity_1"]
        e2 = rel["entity_2"]

        # ---- CASE: N:M with associative entity ----
        if card == "N:M" and assoc_obj:
            assoc_name = assoc_obj["name"]
            assoc_attrs = assoc_obj.get("attributes", [])

            # add associative entity
            if assoc_name not in new_entities:
                new_entities.append(assoc_name)

            # initialize attributes
            if assoc_name not in new_attributes:
                new_attributes[assoc_name] = []

            # add attributes from LLM
            for a in assoc_attrs:
                if isinstance(a, str):
                    # Check if attribute already exists (normalized)
                    normalized_new = normalize_attribute_name(a)
                    exists = any(normalize_attribute_name(existing) == normalized_new 
                                for existing in new_attributes[assoc_name])
                    if not exists:
                        new_attributes[assoc_name].append(a)
                else:
                    # If a is a tuple (attr_name, score)
                    attr_name = a[0] if isinstance(a, tuple) else str(a)
                    normalized_new = normalize_attribute_name(attr_name)
                    exists = any(normalize_attribute_name(existing[0] if isinstance(existing, tuple) else existing) == normalized_new 
                                for existing in new_attributes[assoc_name])
                    if not exists:
                        new_attributes[assoc_name].append(a)

            # Generate standard PK/FK names (prefer entity format)
            # Determine the common ID format of the entity
            def get_entity_id_format(entity_name):
                """Determine the ID format of an entity from its existing attributes"""
                entity_lower = entity_name.lower()
                
                # Common formats
                formats = [
                    f"{entity_lower}_id",      # rental_id
                    f"id_{entity_lower}",      # id_rental  
                    f"{entity_lower}id",       # rentalid
                    f"{entity_lower}_code",    # rental_code
                    f"{entity_lower}key",      # rentalkey
                ]
                
                # Check existing attributes of the entity
                if entity_name in new_attributes:
                    existing_attrs = [attr[0] if isinstance(attr, tuple) else attr 
                                     for attr in new_attributes[entity_name]]
                    
                    # Find which format already exists
                    for fmt in formats:
                        for attr in existing_attrs:
                            if normalize_attribute_name(attr) == normalize_attribute_name(fmt):
                                return fmt
                
                # Default: entity_id
                return f"{entity_lower}_id"

            # Get format for each entity
            pk_format = get_entity_id_format(assoc_name)
            fk1_format = get_entity_id_format(e1)
            fk2_format = get_entity_id_format(e2)
            
            # Ensure PK is unique for the associative entity
            pk_name = pk_format
            
            # Ensure FK references correct format
            # If e1/e2 have no attributes yet, add them in standard format
            for entity, fk_format in [(e1, fk1_format), (e2, fk2_format)]:
                if entity not in new_attributes:
                    new_attributes[entity] = []
                
                # Check if an attribute with this format already exists
                normalized_fk = normalize_attribute_name(fk_format)
                has_fk = any(normalize_attribute_name(attr[0] if isinstance(attr, tuple) else attr) == normalized_fk 
                           for attr in new_attributes[entity])
                if not has_fk:
                    new_attributes[entity].append(fk_format)

            # Add PK/FK to associative entity if not yet present
            for attr_name in [pk_name, fk1_format, fk2_format]:
                normalized_attr = normalize_attribute_name(attr_name)
                exists = any(normalize_attribute_name(existing[0] if isinstance(existing, tuple) else existing) == normalized_attr 
                           for existing in new_attributes[assoc_name])
                if not exists:
                    new_attributes[assoc_name].append(attr_name)

            # replace N:M with two 1:N relationships
            new_relations.append({
                "entity_1": e1,
                "entity_2": assoc_name,
                "cardinality": "1:N",
                "associative_entity": None
            })
            new_relations.append({
                "entity_1": e2,
                "entity_2": assoc_name,
                "cardinality": "1:N",
                "associative_entity": None
            })

        else:
            new_relations.append(rel)

    # Clean up attributes: remove exact duplicates
    for entity in new_attributes:
        unique_attrs = []
        seen_normalized = set()
        
        for attr in new_attributes[entity]:
            if isinstance(attr, tuple):
                attr_name, score = attr
                normalized = normalize_attribute_name(attr_name)
            else:
                attr_name = str(attr)
                normalized = normalize_attribute_name(attr_name)
            
            if normalized not in seen_normalized:
                seen_normalized.add(normalized)
                unique_attrs.append(attr)
        
        new_attributes[entity] = unique_attrs

    return new_entities, new_attributes, new_relations

class JointERILPSolverFinal_older:
    def __init__(
        self,
        entity_probs: Dict[str, float],
        relation_rows: List[Dict[str, float]],
        attribute_probs: Dict[str, List[Tuple[str, float]]]
    ):
        """
        Args:
            entity_probs: {entity: probability}
            relation_rows: [
              {"e1": str, "e2": str, "p": float, "cardinality": str, ...},
              ...
            ]
            attribute_probs: {entity: [(attr_name, prob), ...]}
        """
        self.entities = entity_probs

        # Save with cardinality
        self.relations = [
            (
                r["e1"],
                r["e2"],
                r["p"],
                r.get("cardinality", ""), 
                r.get("associative_entity", None)
            )
            for r in relation_rows
        ]

        self.attributes = attribute_probs  # keep as is

    @staticmethod
    def log_odds(p, eps=1e-6):
        p = float(p)
        p = max(min(p, 1 - eps), eps)
        return math.log(p / (1 - p))

    def solve(
        self,
        lambda_E: float,
        lambda_A: float,
        lambda_R: float,
        min_entities=3
    ):
        start_time = time.time()
 
        prob = pulp.LpProblem("Optimal_ER_Model_Selection", pulp.LpMaximize)

        # -------- Decision variables --------
        xE = {
            e: pulp.LpVariable(f"xE_{e}", cat="Binary")
            for e in self.entities
        }

        zA = {}
        for e, attrs in self.attributes.items():
            for attr, _ in attrs:
                zA[(e, attr)] = pulp.LpVariable(
                    f"zA_{e}_{attr}", cat="Binary"
                )

        yR = {}
        for s, t, _, _, assoc in self.relations:
            yR[(s, t)] = pulp.LpVariable(
                f"yR_{s}_{t}", cat="Binary"
            )

        # -------- Objective --------
        prob += (
            pulp.lpSum(
                xE[e] * self.log_odds(self.entities[e])
                for e in self.entities
            )
            + pulp.lpSum(
                zA[(e, attr)] * self.log_odds(p)
                for e, attrs in self.attributes.items()
                for attr, p in attrs
            )
            + pulp.lpSum(
                yR[(s, t)] * self.log_odds(p)
                for s, t, p, _, assoc in self.relations
            )
            - lambda_E * pulp.lpSum(xE.values())
            - lambda_A * pulp.lpSum(zA.values())
            - lambda_R * pulp.lpSum(yR.values())
        )

        # -------- Constraints --------
        prob += pulp.lpSum(xE.values()) >= min_entities

        for s, t, _, _, assoc in self.relations:
            prob += yR[(s, t)] <= xE[s]
            prob += yR[(s, t)] <= xE[t]

        for e, attrs in self.attributes.items():
            for attr, _ in attrs:
                prob += zA[(e, attr)] <= xE[e]

        # -------- Solve --------
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        runtime = time.time() - start_time

        # -------- Extract solution --------
        selected_entities = [
            e for e in xE if pulp.value(xE[e]) > 0.5
        ]

        selected_attributes = {
            e: [
                attr for attr, _ in self.attributes[e]
                if pulp.value(zA[(e, attr)]) > 0.5
            ]
            for e in selected_entities
            if e in self.attributes
        }

        # Return including cardinality
        selected_relations = []
        for s, t, _, card, assoc in self.relations:
            if pulp.value(yR[(s, t)]) > 0.5:
                selected_relations.append({
                    "entity_1": s,
                    "entity_2": t,
                    "cardinality": card, 
                    "associative_entity": assoc
                })

        score = pulp.value(prob.objective)

        new_entities, new_attributes, new_relations = postprocess_associative_entities(selected_entities, selected_attributes, selected_relations)

        return (
            score,
            new_entities,
            new_relations,
            new_attributes,
            runtime
        )



class JointERILPSolverFinal_1:
    def __init__(
        self,
        entity_probs: Dict[str, float],
        relation_rows: List[Dict[str, Any]],
        attribute_probs: Dict[str, Any]
    ):
        """
        Args:
            entity_probs: {entity: probability}
            relation_rows: [
              {"entity_1": str, "entity_2": str, "probability": float, "cardinality": str, "associative_entity": dict|None},
              ...
            ]
            attribute_probs: {entity: [(attr_name, prob), ...]}
        """
        self.entities = entity_probs

        # Support both "e1"/"e2"/"p" and "entity_1"/"entity_2"/"probability" keys
        # Deduplicate (s, t) pairs and skip relations with non-existing entities
        valid_entities = set(entity_probs.keys())
        _seen: Dict[tuple, tuple] = {}
        for r in relation_rows:
            s = r.get("e1", r.get("entity_1", "")).upper()
            t = r.get("e2", r.get("entity_2", "")).upper()
            if s not in valid_entities or t not in valid_entities:
                continue
            p = float(r.get("p", r.get("probability", 0.0)))
            card = r.get("cardinality", "1:N")
            assoc = r.get("associative_entity", None)
            key = tuple(sorted([s, t]))
            if key not in _seen or p > _seen[key][2]:
                _seen[key] = (s, t, p, card, assoc)
        self.relations = list(_seen.values())

        # Ensure attributes are in List[Tuple[str, float]] format, deduplicate names
        # Normalize key: lowercase + underscore→space so "battle_id" == "battle id"
        def _norm_attr(name: str) -> str:
            return name.replace('_', ' ').strip().lower()

        self.attributes = {}
        for e, attrs in attribute_probs.items():
            if e not in valid_entities:
                continue  # skip attributes for entities not in entity_probs
            if isinstance(attrs, dict):
                pairs = list(attrs.items())
            else:
                pairs = attrs
            seen_attr: Dict[str, tuple] = {}  # norm_key → (original_name, prob)
            for attr, p in pairs:
                key = _norm_attr(attr)
                if key not in seen_attr or float(p) > seen_attr[key][1]:
                    seen_attr[key] = (attr, float(p))
            self.attributes[e] = [(name, prob) for name, prob in seen_attr.values()]

        print(">>> ILP START")
        print("Entities:", self.entities.keys())
        print("Relations:", [(s,t) for s,t,_,_,_ in self.relations])
        print("Attributes keys:", self.attributes.keys())

    @staticmethod
    def log_odds(p, eps=1e-6):
        p = float(p)
        p = max(min(p, 1 - eps), eps)
        return math.log(p / (1 - p))

    def solve(
        self,
        lambda_E: float,
        lambda_A: float,
        lambda_R: float,
        min_entities=3,
        no_isolated=False,
    ):
        start_time = time.time()

        prob = pulp.LpProblem("Optimal_ER_Model_Selection", pulp.LpMaximize)

        def safe_var(name: str) -> str:
            """Replace non-alphanumeric characters to make CBC-safe variable names."""
            return re.sub(r'[^a-zA-Z0-9_]', '_', name)

        # -------- Decision variables --------
        xE = {e: pulp.LpVariable(f"xE_{safe_var(e)}", cat="Binary") for e in self.entities}

        zA = {}
        for e, attrs in self.attributes.items():
            for attr, _ in attrs:
                zA[(e, attr)] = pulp.LpVariable(f"zA_{safe_var(e)}_{safe_var(attr)}", cat="Binary")

        yR = {}
        for s, t, p, _, assoc in self.relations:
            yR[(s, t)] = pulp.LpVariable(f"yR_{safe_var(s)}_{safe_var(t)}", cat="Binary")

        # -------- Objective --------
        prob += (
            pulp.lpSum(xE[e] * self.log_odds(self.entities[e]) for e in self.entities)
            + pulp.lpSum(zA[(e, attr)] * self.log_odds(p) for e, attrs in self.attributes.items() for attr, p in attrs)
            + pulp.lpSum(yR[(s, t)] * self.log_odds(p) for s, t, p, _, _ in self.relations)
            - lambda_E * pulp.lpSum(xE.values())
            - lambda_A * pulp.lpSum(zA.values())
            - lambda_R * pulp.lpSum(yR.values())
        )

        # -------- Constraints --------
        prob += pulp.lpSum(xE.values()) >= min_entities

        for s, t, _, _, _ in self.relations:
            prob += yR[(s, t)] <= xE[s]
            prob += yR[(s, t)] <= xE[t]

        # Entity ⇒ must participate in at least one relationship (no isolated entities)
        # Also covers entities with zero incident relationships: lpSum([]) == 0 → xE[e] <= 0
        if no_isolated:
            incident = {e: [] for e in self.entities}
            for s, t, _, _, _ in self.relations:
                incident[s].append(yR[(s, t)])
                incident[t].append(yR[(s, t)])
            for e in self.entities:
                prob += xE[e] <= pulp.lpSum(incident[e])

        for e, attrs in self.attributes.items():
            for attr, _ in attrs:
                prob += zA[(e, attr)] <= xE[e]

        # -------- Solve --------
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        runtime = time.time() - start_time

        # -------- Extract solution --------
        selected_entities = [e for e in xE if pulp.value(xE[e]) > 0.5]

        selected_attributes = {
            e: [attr for attr, _ in self.attributes[e] if pulp.value(zA[(e, attr)]) > 0.5]
            for e in selected_entities if e in self.attributes
        }

        # Build selected relations
        selected_relations = []
        for s, t, _, card, assoc in self.relations:
            if pulp.value(yR[(s, t)]) > 0.5:
                selected_relations.append({
                    "entity_1": s,
                    "entity_2": t,
                    "cardinality": card,
                    "associative_entity": assoc
                })

        score = pulp.value(prob.objective)

        # Skip post-processing since N:M was already resolved at the probability generation step
        return score, selected_entities, selected_relations, selected_attributes, runtime


# Max-Flow Min Cut
import time
import math
from collections import defaultdict, deque
from typing import Dict, List, Tuple

class MaxFlowER:
    def __init__(self, entity_probs: Dict[str,float], relation_rows: List[Dict], attribute_probs: Dict[str,List[Tuple[str,float]]]):
        # ==================== DATA ====================
        self.entity_probs = entity_probs
        self.relation_rows = relation_rows
        self.attribute_probs = attribute_probs
        
        # ==================== NODES ====================
        self.entities = list(entity_probs.keys())
        self.relations = [f"R_{i}({r['e1']},{r['e2']})" for i,r in enumerate(relation_rows)]
        self.attributes = []
        self.attr_to_parent = {}
        for ent, attrs in attribute_probs.items():
            for attr_name, p in attrs:
                node_name = f"A_{attr_name}@{ent}"
                self.attributes.append(node_name)
                self.attr_to_parent[node_name] = ent
        
        self.all_nodes = self.entities + self.relations + self.attributes
        self.n = len(self.all_nodes)
        self.node_to_id = {node:i for i,node in enumerate(self.all_nodes)}
        self.source = self.n
        self.sink = self.n + 1
        self.total_nodes = self.n + 2
        
        # ==================== WEIGHTS ====================
        self.weights = {}
        for ent,p in self.entity_probs.items():
            self.weights[ent] = self.log_odds(p)
        for i,row in enumerate(self.relation_rows):
            self.weights[self.relations[i]] = self.log_odds(row["p"])
        for ent, attrs in self.attribute_probs.items():
            for attr_name,p in attrs:
                node_name = f"A_{attr_name}@{ent}"
                self.weights[node_name] = self.log_odds(p)
        
        self.INF = sum(max(0,w) for w in self.weights.values()) + 1
    
    def log_odds(self, p: float) -> float:
        p = max(min(p, 0.999999), 0.000001)
        return math.log(p / (1 - p))
    
    # ==================== BUILD GRAPH ====================
    def build_graph(self):
        graph = [defaultdict(float) for _ in range(self.total_nodes)]
        # source/sink edges
        for node_name, w in self.weights.items():
            u = self.node_to_id[node_name]
            if w > 0:
                graph[self.source][u] = w
            elif w < 0:
                graph[u][self.sink] = -w
        # attribute -> parent entity
        for attr, parent in self.attr_to_parent.items():
            if parent in self.node_to_id:
                graph[self.node_to_id[attr]][self.node_to_id[parent]] = self.INF
        # relation -> e1,e2
        for i,row in enumerate(self.relation_rows):
            rel = self.relations[i]
            e1,e2 = row["e1"], row["e2"]
            if e1 in self.node_to_id and e2 in self.node_to_id:
                u = self.node_to_id[rel]
                graph[u][self.node_to_id[e1]] = self.INF
                graph[u][self.node_to_id[e2]] = self.INF
        return graph
    
    # ==================== MAX-FLOW ALGORITHMS ====================
    def edmonds_karp(self):
        graph = self.build_graph()
        parent = [-1]*self.total_nodes
        max_flow = 0.0
        def bfs():
            visited = [False]*self.total_nodes
            q = deque([self.source])
            visited[self.source]=True
            while q:
                u = q.popleft()
                for v in graph[u]:
                    if not visited[v] and graph[u][v] > 1e-9:
                        visited[v]=True
                        parent[v]=u
                        q.append(v)
                        if v==self.sink: return True
            return False
        while bfs():
            path_flow=float('inf')
            v=self.sink
            while v!=self.source:
                u=parent[v]
                path_flow=min(path_flow, graph[u][v])
                v=u
            max_flow+=path_flow
            v=self.sink
            while v!=self.source:
                u=parent[v]
                graph[u][v]-=path_flow
                graph[v][u]+=path_flow
                v=u
        return graph, max_flow
    
    def dinic(self):
        graph = self.build_graph()
        level = [0]*self.total_nodes
        iter = [0]*self.total_nodes
        def bfs():
            level[:] = [-1]*self.total_nodes
            level[self.source]=0
            q = deque([self.source])
            while q:
                u=q.popleft()
                for v in graph[u]:
                    if graph[u][v]>1e-9 and level[v]<0:
                        level[v]=level[u]+1
                        q.append(v)
        def dfs(u,flow):
            if u==self.sink: return flow
            for v in list(graph[u].keys())[iter[u]:]:
                iter[u]+=1
                if level[v]==level[u]+1 and graph[u][v]>1e-9:
                    d=dfs(v,min(flow,graph[u][v]))
                    if d>0:
                        graph[u][v]-=d
                        graph[v][u]+=d
                        return d
            return 0
        flow=0.0
        while True:
            bfs()
            if level[self.sink]<0: break
            iter[:] = [0]*self.total_nodes
            while (f:=dfs(self.source,float('inf')))>0:
                flow+=f
        return graph, flow
    
    def push_relabel(self):
        graph = self.build_graph()
        height = [0]*self.total_nodes
        height[self.source]=self.total_nodes
        excess = [0.0]*self.total_nodes
        for v in list(graph[self.source].keys()):
            cap = graph[self.source][v]
            if cap>0:
                excess[v]+=cap
                graph[v][self.source]+=cap
                graph[self.source][v]=0.0
        queue = deque(i for i in range(self.total_nodes) if excess[i]>1e-9 and i not in (self.source,self.sink))
        def push(u):
            for v in list(graph[u].keys()):
                if graph[u][v]>1e-9 and height[u]==height[v]+1:
                    amt=min(excess[u],graph[u][v])
                    graph[u][v]-=amt
                    graph[v][u]+=amt
                    excess[u]-=amt
                    excess[v]+=amt
                    if excess[v]>1e-9 and v not in (self.source,self.sink):
                        queue.append(v)
                    return True
            return False
        def relabel(u):
            min_h=float('inf')
            for v in graph[u]:
                if graph[u][v]>1e-9:
                    min_h=min(min_h,height[v])
            if min_h<float('inf'):
                height[u]=min_h+1
        while queue:
            u=queue.popleft()
            if excess[u]<1e-9: continue
            if not push(u):
                relabel(u)
                queue.append(u)
        flow=sum(graph[i][self.sink] for i in range(self.total_nodes) if self.sink in graph[i])
        return graph, flow
    
    # ==================== SELECTED NODES ====================
    def get_selected(self, res_graph):
        visited = [False]*self.total_nodes
        q=deque([self.source])
        visited[self.source]=True
        while q:
            u=q.popleft()
            for v in res_graph[u]:
                if not visited[v] and res_graph[u][v]>1e-9:
                    visited[v]=True
                    q.append(v)
        return [node for node,idx in self.node_to_id.items() if visited[idx]]
    
    # ==================== RUN ALL ====================
    def run_all(self):
        results = {}
        for name, func in [("Edmonds-Karp", self.edmonds_karp), ("Dinic", self.dinic), ("Push-Relabel", self.push_relabel)]:
            start = time.time()
            res_graph, flow = func()
            elapsed = (time.time()-start)*1000
            selected = self.get_selected(res_graph)
            score = sum(self.weights.get(x,0) for x in selected)
            ents = [e for e in selected if e in self.entities]
            rels = [r for r in selected if r.startswith('R_')]
            attrs = [a for a in selected if a.startswith('A_')]
            results[name] = {
                "time_ms": elapsed,
                "score": score,
                "flow": flow,
                "entities": ents,
                "relations": rels,
                "attributes": attrs
            }
        return results
    

    # Max-Flow Min Cut with Penalties
import time
import math
from collections import defaultdict, deque
from typing import Dict, List, Tuple

class MaxFlowER_with_panelty:
    def __init__(self, 
                 entity_probs: Dict[str,float], 
                 relation_rows: List[Dict], 
                 attribute_probs: Dict[str,List[Tuple[str,float]]],
                 entity_penalty: float = 0.0,
                 relation_penalty: float = 0.0,
                 attribute_penalty: float = 0.0):
        # ==================== DATA ====================
        self.entity_probs = entity_probs
        self.relation_rows = relation_rows
        self.attribute_probs = attribute_probs
        
        # ==================== PENALTIES ====================
        self.entity_penalty = entity_penalty
        self.relation_penalty = relation_penalty
        self.attribute_penalty = attribute_penalty
        
        # ==================== NODES ====================
        self.entities = list(entity_probs.keys())
        self.relations = [f"R_{i}({r['e1']},{r['e2']})" for i,r in enumerate(relation_rows)]
        self.attributes = []
        self.attr_to_parent = {}
        for ent, attrs in attribute_probs.items():
            for attr_name, p in attrs:
                node_name = f"A_{attr_name}@{ent}"
                self.attributes.append(node_name)
                self.attr_to_parent[node_name] = ent
        
        self.all_nodes = self.entities + self.relations + self.attributes
        self.n = len(self.all_nodes)
        self.node_to_id = {node:i for i,node in enumerate(self.all_nodes)}
        self.source = self.n
        self.sink = self.n + 1
        self.total_nodes = self.n + 2
        
        # ==================== WEIGHTS ====================
        self.weights = {}
        for ent,p in self.entity_probs.items():
            self.weights[ent] = self.log_odds(p) - self.entity_penalty
        for i,row in enumerate(self.relation_rows):
            self.weights[self.relations[i]] = self.log_odds(row["p"]) - self.relation_penalty
        for ent, attrs in self.attribute_probs.items():
            for attr_name,p in attrs:
                node_name = f"A_{attr_name}@{ent}"
                self.weights[node_name] = self.log_odds(p) - self.attribute_penalty
        
        self.INF = sum(max(0,w) for w in self.weights.values()) + 1
    
    # ==================== HELPER ====================
    def log_odds(self, p: float) -> float:
        p = max(min(p, 0.999999), 0.000001)
        return math.log(p / (1 - p))
    
    # ==================== BUILD GRAPH ====================
    def build_graph(self):
        graph = [defaultdict(float) for _ in range(self.total_nodes)]
        # source/sink edges
        for node_name, w in self.weights.items():
            u = self.node_to_id[node_name]
            if w > 0:
                graph[self.source][u] = w
            elif w < 0:
                graph[u][self.sink] = -w
        # attribute -> parent entity
        for attr, parent in self.attr_to_parent.items():
            if parent in self.node_to_id:
                graph[self.node_to_id[attr]][self.node_to_id[parent]] = self.INF
        # relation -> e1,e2
        for i,row in enumerate(self.relation_rows):
            rel = self.relations[i]
            e1,e2 = row["e1"], row["e2"]
            if e1 in self.node_to_id and e2 in self.node_to_id:
                u = self.node_to_id[rel]
                graph[u][self.node_to_id[e1]] = self.INF
                graph[u][self.node_to_id[e2]] = self.INF
        return graph
    
    # ==================== MAX-FLOW ALGORITHMS ====================
    def edmonds_karp(self):
        graph = self.build_graph()
        parent = [-1]*self.total_nodes
        max_flow = 0.0
        def bfs():
            visited = [False]*self.total_nodes
            q = deque([self.source])
            visited[self.source]=True
            while q:
                u = q.popleft()
                for v in graph[u]:
                    if not visited[v] and graph[u][v] > 1e-9:
                        visited[v]=True
                        parent[v]=u
                        q.append(v)
                        if v==self.sink: return True
            return False
        while bfs():
            path_flow=float('inf')
            v=self.sink
            while v!=self.source:
                u=parent[v]
                path_flow=min(path_flow, graph[u][v])
                v=u
            max_flow+=path_flow
            v=self.sink
            while v!=self.source:
                u=parent[v]
                graph[u][v]-=path_flow
                graph[v][u]+=path_flow
                v=u
        return graph, max_flow
    
    def dinic(self):
        graph = self.build_graph()
        level = [0]*self.total_nodes
        iter = [0]*self.total_nodes
        def bfs():
            level[:] = [-1]*self.total_nodes
            level[self.source]=0
            q = deque([self.source])
            while q:
                u=q.popleft()
                for v in graph[u]:
                    if graph[u][v]>1e-9 and level[v]<0:
                        level[v]=level[u]+1
                        q.append(v)
        def dfs(u,flow):
            if u==self.sink: return flow
            for v in list(graph[u].keys())[iter[u]:]:
                iter[u]+=1
                if level[v]==level[u]+1 and graph[u][v]>1e-9:
                    d=dfs(v,min(flow,graph[u][v]))
                    if d>0:
                        graph[u][v]-=d
                        graph[v][u]+=d
                        return d
            return 0
        flow=0.0
        while True:
            bfs()
            if level[self.sink]<0: break
            iter[:] = [0]*self.total_nodes
            while (f:=dfs(self.source,float('inf')))>0:
                flow+=f
        return graph, flow
    
    def push_relabel(self):
        graph = self.build_graph()
        height = [0]*self.total_nodes
        height[self.source]=self.total_nodes
        excess = [0.0]*self.total_nodes
        for v in list(graph[self.source].keys()):
            cap = graph[self.source][v]
            if cap>0:
                excess[v]+=cap
                graph[v][self.source]+=cap
                graph[self.source][v]=0.0
        queue = deque(i for i in range(self.total_nodes) if excess[i]>1e-9 and i not in (self.source,self.sink))
        def push(u):
            for v in list(graph[u].keys()):
                if graph[u][v]>1e-9 and height[u]==height[v]+1:
                    amt=min(excess[u],graph[u][v])
                    graph[u][v]-=amt
                    graph[v][u]+=amt
                    excess[u]-=amt
                    excess[v]+=amt
                    if excess[v]>1e-9 and v not in (self.source,self.sink):
                        queue.append(v)
                    return True
            return False
        def relabel(u):
            min_h=float('inf')
            for v in graph[u]:
                if graph[u][v]>1e-9:
                    min_h=min(min_h,height[v])
            if min_h<float('inf'):
                height[u]=min_h+1
        while queue:
            u=queue.popleft()
            if excess[u]<1e-9: continue
            if not push(u):
                relabel(u)
                queue.append(u)
        flow=sum(graph[i][self.sink] for i in range(self.total_nodes) if self.sink in graph[i])
        return graph, flow
    
    # ==================== SELECTED NODES ====================
    def get_selected(self, res_graph):
        visited = [False]*self.total_nodes
        q=deque([self.source])
        visited[self.source]=True
        while q:
            u=q.popleft()
            for v in res_graph[u]:
                if not visited[v] and res_graph[u][v]>1e-9:
                    visited[v]=True
                    q.append(v)
        return [node for node,idx in self.node_to_id.items() if visited[idx]]
    
    # ==================== RUN ALL ====================
    def run_all(self):
        results = {}
        for name, func in [("Edmonds-Karp", self.edmonds_karp), 
                           ("Dinic", self.dinic), 
                           ("Push-Relabel", self.push_relabel)]:
            start = time.time()
            res_graph, flow = func()
            elapsed = (time.time()-start)*1000
            selected = self.get_selected(res_graph)
            score = sum(self.weights.get(x,0) for x in selected)
            ents = [e for e in selected if e in self.entities]
            rels = [r for r in selected if r.startswith('R_')]
            attrs = [a for a in selected if a.startswith('A_')]
            results[name] = {
                "time_ms": elapsed,
                "score": score,
                "flow": flow,
                "entities": ents,
                "relations": rels,
                "attributes": attrs
            }
        return results


"""

# ==================== RUN EXAMPLE ====================
if __name__ == "__main__":
    entity_probs = {
        "PATIENT": 0.880797, "DOCTOR": 0.880797, "APPOINTMENT": 0.880797,
        "TREATMENT": 0.880797, "BILLING": 0.880797, "HOSPITAL": 0.688110, "INSURANCE_PROVIDER": 0.550131
    }

    relation_rows = [
        {"e1": "PATIENT", "e2": "APPOINTMENT", "p": 0.877466},
        {"e1": "DOCTOR", "e2": "APPOINTMENT", "p": 0.832598},
        {"e1": "APPOINTMENT", "e2": "TREATMENT", "p": 0.876322},
        {"e1": "TREATMENT", "e2": "BILLING", "p": 0.855760},
        {"e1": "PATIENT", "e2": "BILLING", "p": 0.867064},
        {"e1": "PATIENT", "e2": "INSURANCE_PROVIDER", "p": 0.818981},
        {"e1": "DOCTOR", "e2": "HOSPITAL", "p": 0.865542},
        {"e1": "PATIENT", "e2": "HOSPITAL", "p": 0.865542},
        {"e1": "TREATMENT", "e2": "INSURANCE_PROVIDER", "p": 0.814619}
    ]

    attribute_probs = {
        'PATIENT': [('patient id', 0.731059), ('first name', 0.731059), ('last name', 0.731059), ('gender', 0.731059),
                    ('date of birth', 0.731059), ('contact number', 0.731059), ('address', 0.731032),
                    ('registration date', 0.731059), ('hospital', 0.731058), ('insurance provider', 0.730999),
                    ('insurance number', 0.731059), ('email', 0.720472)],
        'DOCTOR': [('doctor id', 0.713846), ('first name', 0.713846), ('last name', 0.713846),
                   ('specialization', 0.65966), ('phone number', 0.713846), ('years experience', 0.713846),
                   ('hospital branch', 0.713846), ('email', 0.713846)],
        'APPOINTMENT': [('appointment id', 0.701981), ('patient id', 0.701981), ('doctor id', 0.701981),
                        ('appointment date', 0.701981), ('appointment time', 0.701981), ('reason for visit', 0.701981),
                        ('status', 0.701981)],
        'TREATMENT': [('treatment id', 0.728635), ('appointment id', 0.728635), ('treatment type', 0.728635),
                      ('description', 0.728635), ('cost', 0.728635), ('treatment date', 0.728635)],
        'BILLING': [('bill id', 0.731059), ('patient id', 0.731059), ('treatment id', 0.731059),
                    ('bill date', 0.731059), ('billing amount', 0.697127), ('payment method', 0.731059),
                    ('payment status', 0.731059)],
        'HOSPITAL': [('hospital id', 0.730456), ('hospital name', 0.730456), ('hospital address', 0.730456)],
        'INSURANCE_PROVIDER': [('insurance provider id', 0.731059), ('insurance provider name', 0.731059),
                               ('insurance provider address', 0.731059)]
    }

    mf = MaxFlowER_with_panelty(entity_probs, relation_rows, attribute_probs,
               entity_penalty=0.5,
               relation_penalty=0.2,
               attribute_penalty=0.1)

    results = mf.run_all()
    print(results["Edmonds-Karp"]["entities"])


    hgws = HGWS_ILP_ER(entity_probs, relation_rows, attribute_probs)
    score, E_sel, R_sel, A_sel, timing = hgws.solve_ilp(lam_E=0.3, lam_R=0.3, lam_A=0.2, timeLimit=10)

    print("HGWS-ILP ER Selection (Entities + Relations + Attributes)")
    print("="*80)
    print(f"Score            : {score:.6f}")
    print(f"Entities         : {len(E_sel)} → {sorted(E_sel)}")
    print(f"Relations        : {len(R_sel)} → {sorted(R_sel)}")
    print(f"Attributes       : {len(A_sel)} → {sorted(A_sel)}")
    print(f"Runtime (ms)     : {timing['total_ms']:.3f}")


    er = MaxFlowER(entity_probs, relation_rows, attribute_probs)
    
    print("="*80)
    print("MAX-FLOW ER MODEL SELECTION (Entities + Relations + Attributes)")
    print("="*80)
    
    results = er.run_all()
    
    for algo, res in results.items():
        E_sel = res['entities']
        R_sel = res['relations']
        A_sel = res['attributes']
        print(f"\nAlgorithm: {algo}")
        print("-"*80)
        print(f"Score            : {res['score']:.6f}")
        print(f"Entities         : {len(E_sel)} → {sorted(E_sel)}")
        print(f"Relations        : {len(R_sel)} → {sorted(R_sel)}")
        print(f"Attributes       : {len(A_sel)} → {sorted(A_sel)}")
        print(f"Runtime (ms)     : {res['time_ms']:.3f}")
        print("="*80)

"""


"""
# ==================== RUN EXAMPLE ====================
if __name__ == "__main__":
    entity_probs = {
        "PATIENT": 0.880797, "DOCTOR": 0.880797, "APPOINTMENT": 0.880797,
        "TREATMENT": 0.880797, "BILLING": 0.880797, "HOSPITAL": 0.688110, "INSURANCE_PROVIDER": 0.550131
    }

    relation_rows = [
        {"e1": "PATIENT", "e2": "APPOINTMENT", "p": 0.877466},
        {"e1": "DOCTOR", "e2": "APPOINTMENT", "p": 0.832598},
        {"e1": "APPOINTMENT", "e2": "TREATMENT", "p": 0.876322},
        {"e1": "TREATMENT", "e2": "BILLING", "p": 0.855760},
        {"e1": "PATIENT", "e2": "BILLING", "p": 0.867064},
        {"e1": "PATIENT", "e2": "INSURANCE_PROVIDER", "p": 0.818981},
        {"e1": "DOCTOR", "e2": "HOSPITAL", "p": 0.865542},
        {"e1": "PATIENT", "e2": "HOSPITAL", "p": 0.865542},
        {"e1": "TREATMENT", "e2": "INSURANCE_PROVIDER", "p": 0.814619}
    ]

    attribute_probs = {
        'PATIENT': [('patient id', 0.731059), ('first name', 0.731059), ('last name', 0.731059), ('gender', 0.731059),
                    ('date of birth', 0.731059), ('contact number', 0.731059), ('address', 0.731032),
                    ('registration date', 0.731059), ('hospital', 0.731058), ('insurance provider', 0.730999),
                    ('insurance number', 0.731059), ('email', 0.720472)],
        'DOCTOR': [('doctor id', 0.713846), ('first name', 0.713846), ('last name', 0.713846),
                   ('specialization', 0.65966), ('phone number', 0.713846), ('years experience', 0.713846),
                   ('hospital branch', 0.713846), ('email', 0.713846)],
        'APPOINTMENT': [('appointment id', 0.701981), ('patient id', 0.701981), ('doctor id', 0.701981),
                        ('appointment date', 0.701981), ('appointment time', 0.701981), ('reason for visit', 0.701981),
                        ('status', 0.701981)],
        'TREATMENT': [('treatment id', 0.728635), ('appointment id', 0.728635), ('treatment type', 0.728635),
                      ('description', 0.728635), ('cost', 0.728635), ('treatment date', 0.728635)],
        'BILLING': [('bill id', 0.731059), ('patient id', 0.731059), ('treatment id', 0.731059),
                    ('bill date', 0.731059), ('billing amount', 0.697127), ('payment method', 0.731059),
                    ('payment status', 0.731059)],
        'HOSPITAL': [('hospital id', 0.730456), ('hospital name', 0.730456), ('hospital address', 0.730456)],
        'INSURANCE_PROVIDER': [('insurance provider id', 0.731059), ('insurance provider name', 0.731059),
                               ('insurance provider address', 0.731059)]
    }

    greedy_model = GreedyER_LogOdds(entity_probs, relation_rows, attribute_probs)
    E_S, R_S, A_S, score_hist, steps_log, runtime = greedy_model.run(lambda_E=0.3, lambda_R=0.3, lambda_A=0.2)

    print("Greedy ER Selection (Entities + Relations + Attributes)")
    print("="*80)
    print(f"Final Score       : {score_hist[-1]:.6f}")
    print(f"Entities (clean)  : {len(E_S)} → {sorted(E_S)}")
    print(f"Relations         : {len(R_S)} → {sorted(R_S)}")
    print(f"Attributes        : {len(A_S)} → {sorted(A_S)}")
    print(f"Runtime           : {runtime*1000:.3f} ms")

"""

# ==================== RUN EXAMPLE ====================
if __name__ == "__main__":
    entity_probs = {
         "PATIENT": 0.880797, "DOCTOR": 0.880797, "APPOINTMENT": 0.880797,
        "TREATMENT": 0.880797, "BILLING": 0.880797, "HOSPITAL": 0.688110, "INSURANCE_PROVIDER": 0.550131
    }

    relation_rows = [
        {"e1": "PATIENT", "e2": "APPOINTMENT", "p": 0.877466},
        {"e1": "DOCTOR", "e2": "APPOINTMENT", "p": 0.832598},
        {"e1": "APPOINTMENT", "e2": "TREATMENT", "p": 0.876322},
        {"e1": "TREATMENT", "e2": "BILLING", "p": 0.855760},
        {"e1": "PATIENT", "e2": "BILLING", "p": 0.867064},
        {"e1": "PATIENT", "e2": "INSURANCE_PROVIDER", "p": 0.818981},
        {"e1": "DOCTOR", "e2": "HOSPITAL", "p": 0.865542},
        {"e1": "PATIENT", "e2": "HOSPITAL", "p": 0.865542},
        {"e1": "TREATMENT", "e2": "INSURANCE_PROVIDER", "p": 0.814619}
    ]

    attribute_probs = {
        'PATIENT': [('patient id', 0.731059), ('first name', 0.731059), ('last name', 0.731059), ('gender', 0.731059),
                    ('date of birth', 0.731059), ('contact number', 0.731059), ('address', 0.731032),
                    ('registration date', 0.731059), ('hospital', 0.731058), ('insurance provider', 0.730999),
                    ('insurance number', 0.731059), ('email', 0.720472)],
        'DOCTOR': [('doctor id', 0.713846), ('first name', 0.713846), ('last name', 0.713846),
                   ('specialization', 0.65966), ('phone number', 0.713846), ('years experience', 0.713846),
                   ('hospital branch', 0.713846), ('email', 0.713846)],
        'APPOINTMENT': [('appointment id', 0.701981), ('patient id', 0.701981), ('doctor id', 0.701981),
                        ('appointment date', 0.701981), ('appointment time', 0.701981), ('reason for visit', 0.701981),
                        ('status', 0.701981)],
        'TREATMENT': [('treatment id', 0.728635), ('appointment id', 0.728635), ('treatment type', 0.728635),
                      ('description', 0.728635), ('cost', 0.728635), ('treatment date', 0.728635)],
        'BILLING': [('bill id', 0.731059), ('patient id', 0.731059), ('treatment id', 0.731059),
                    ('bill date', 0.731059), ('billing amount', 0.697127), ('payment method', 0.731059),
                    ('payment status', 0.731059)],
        'HOSPITAL': [('hospital id', 0.730456), ('hospital name', 0.730456), ('hospital address', 0.730456)],
        'INSURANCE_PROVIDER': [('insurance provider id', 0.731059), ('insurance provider name', 0.731059),
                               ('insurance provider address', 0.731059)]
    }

    er_model = ER_ILP_LogOdds_Joint(entity_probs, relation_rows, attribute_probs)
    score, sel_entities, sel_relations, sel_attributes, runtime = er_model.solve(lambda_E=0.3, lambda_R=0.3, lambda_A=0.2)

    print("ILP + Log-Odds Joint ER Selection (Entities + Relations + Attributes)")
    print("="*80)
    print(f"Score           : {score:.6f}")
    print(f"Entities (clean): {len(sel_entities)} → {sel_entities}")
    print(f"Relations       : {len(sel_relations)} → {sel_relations}")
    print(f"Attributes      : {len(sel_attributes)} → {sel_attributes}")
    print(f"Runtime         : {runtime*1000:.3f} ms")
