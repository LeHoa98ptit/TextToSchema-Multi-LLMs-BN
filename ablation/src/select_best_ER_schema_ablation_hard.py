"""
Ablation Study — Hard-Constraint ILP for ER Schema Selection
=============================================================
Same as JointERILPComplexity but replaces the soft lambda_noattr and
lambda_isolated penalty terms with hard constraints:

    Hard constraint 1 (no isolated entity):
        Σ_{incident r} yR[r]  ≥  xE[e]   for every entity e

    Hard constraint 2 (entity must have at least one attribute):
        Σ_a zA[(e,a)]  ≥  xE[e]           for every entity e that has candidates

    Special case: if an entity has NO candidate relationships (or no candidate
    attributes), it is forced out: xE[e] = 0.

Objective (maximise) — no lambda_noattr / lambda_isolated:
    Σ log_odds(p_e) * xE[e]
  + Σ log_odds(p_a) * zA[(e,a)]
  + Σ log_odds(p_r) * yR[(s,t)]
  - lambda_E  * Σ xE[e]
  - lambda_A  * Σ zA[(e,a)]
  - lambda_R  * Σ yR[(s,t)]
  - lambda_NM * Σ_{N:M} yR[(s,t)]
"""

import math
import re
import time
import pulp
from collections import defaultdict
from typing import Any, Dict, List, Tuple


class HardConstraintERILP:

    def __init__(
        self,
        entity_probs: Dict[str, float],
        relation_rows: List[Dict[str, Any]],
        attribute_probs: Dict[str, Any],
    ):
        self.entities: Dict[str, float] = {
            e.upper(): v for e, v in entity_probs.items()
        }

        _seen: Dict[Tuple[str, str], Tuple] = {}
        for r in relation_rows:
            e1   = r.get("e1",   r.get("entity_1",   "")).upper()
            e2   = r.get("e2",   r.get("entity_2",   "")).upper()
            if e1 not in self.entities or e2 not in self.entities:
                continue
            prob = float(r.get("p", r.get("probability", 0.0)))
            card = r.get("cardinality", "1:N")
            assoc = r.get("associative_entity", None)
            key = (e1, e2)
            if key not in _seen or prob > _seen[key][2]:
                _seen[key] = (e1, e2, prob, card, assoc)
        self.relations: List[Tuple[str, str, float, str, Any]] = list(_seen.values())

        self.attributes: Dict[str, List[Tuple[str, float]]] = {}
        for e, attrs in attribute_probs.items():
            eu = e.upper()
            if eu not in self.entities:
                continue
            if isinstance(attrs, dict):
                self.attributes[eu] = list(attrs.items())
            else:
                self.attributes[eu] = list(attrs)

    @staticmethod
    def _pulp_name(s: str) -> str:
        return re.sub(r'[^a-zA-Z0-9]', '_', str(s))

    @staticmethod
    def log_odds(p: float, eps: float = 1e-6) -> float:
        p = max(min(float(p), 1.0 - eps), eps)
        return math.log(p / (1.0 - p))

    def solve(
        self,
        lambda_E: float  = 0.8,
        lambda_A: float  = 0.6,
        lambda_R: float  = 0.85,
        lambda_NM: float = 0.72,
        min_entities: int = 3,
    ) -> Tuple[float, list, list, dict, float]:
        t0   = time.time()
        prob = pulp.LpProblem("ER_ILP_Hard", pulp.LpMaximize)

        entity_list = list(self.entities.keys())

        xE = {
            e: pulp.LpVariable(f"xE_{i}_{self._pulp_name(e)}", cat="Binary")
            for i, e in enumerate(entity_list)
        }

        zA: Dict[Tuple[str, str], pulp.LpVariable] = {}
        for ei, (e, attrs) in enumerate(self.attributes.items()):
            for ai, (attr, _) in enumerate(attrs):
                zA[(e, attr)] = pulp.LpVariable(
                    f"zA_{ei}_{ai}_{self._pulp_name(e)}_{self._pulp_name(attr)}",
                    cat="Binary",
                )

        yR: Dict[Tuple[str, str], pulp.LpVariable] = {}
        for ri, (s, t, _, _, _) in enumerate(self.relations):
            yR[(s, t)] = pulp.LpVariable(
                f"yR_{ri}_{self._pulp_name(s)}_{self._pulp_name(t)}", cat="Binary"
            )

        # Pre-compute incident relationships and attribute keys per entity
        attrs_by_entity: Dict[str, list] = defaultdict(list)
        for (e, attr) in zA:
            attrs_by_entity[e].append((e, attr))

        rels_by_entity: Dict[str, list] = defaultdict(list)
        for s, t, *_ in self.relations:
            rels_by_entity[s].append((s, t))
            rels_by_entity[t].append((s, t))

        # ── Objective ─────────────────────────────────────────────────────────
        prob += (
            pulp.lpSum(xE[e] * self.log_odds(self.entities[e]) for e in self.entities)
            + pulp.lpSum(
                zA[(e, attr)] * self.log_odds(p)
                for e, attrs in self.attributes.items()
                for attr, p in attrs
            )
            + pulp.lpSum(
                yR[(s, t)] * self.log_odds(p)
                for s, t, p, _, _ in self.relations
            )
            - lambda_E  * pulp.lpSum(xE.values())
            - lambda_A  * pulp.lpSum(zA.values())
            - lambda_R  * pulp.lpSum(yR.values())
            - lambda_NM * pulp.lpSum(
                yR[(s, t)]
                for s, t, _, card, _ in self.relations
                if card == "N:M"
            )
        )

        # ── Constraints ───────────────────────────────────────────────────────

        # 1. Minimum entity count
        prob += pulp.lpSum(xE.values()) >= min_entities, "min_entities"

        # 2. Relationship requires both endpoints
        for i, (s, t, _, _, _) in enumerate(self.relations):
            prob += yR[(s, t)] <= xE[s], f"rel_e1_{i}"
            prob += yR[(s, t)] <= xE[t], f"rel_e2_{i}"

        # 3. Attribute requires its entity
        for ei, (e, attrs) in enumerate(self.attributes.items()):
            for ai, (attr, _) in enumerate(attrs):
                prob += zA[(e, attr)] <= xE[e], f"attr_own_{ei}_{ai}"

        # 4. HARD: selected entity must have ≥1 selected relationship (no isolated)
        for ei, e in enumerate(entity_list):
            incident = rels_by_entity[e]
            if incident:
                prob += pulp.lpSum(yR[k] for k in incident) >= xE[e], f"hard_not_isolated_{ei}"
            else:
                # No candidate relationships → cannot be selected
                prob += xE[e] == 0, f"hard_no_rels_{ei}"

        # 5. HARD: selected entity must have ≥1 selected attribute
        for ei, e in enumerate(entity_list):
            attr_keys = attrs_by_entity[e]
            if attr_keys:
                prob += pulp.lpSum(zA[k] for k in attr_keys) >= xE[e], f"hard_has_attr_{ei}"
            else:
                # No candidate attributes → cannot be selected
                prob += xE[e] == 0, f"hard_no_attrs_{ei}"

        # ── Solve ─────────────────────────────────────────────────────────────
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        runtime = time.time() - t0

        selected_entities = [
            e for e in xE if (pulp.value(xE[e]) or 0) > 0.5
        ]

        selected_attributes = {
            e: [
                attr
                for attr, _ in self.attributes.get(e, [])
                if (pulp.value(zA.get((e, attr), 0)) or 0) > 0.5
            ]
            for e in selected_entities
            if e in self.attributes
        }

        selected_relations = [
            {
                "entity_1":           s,
                "entity_2":           t,
                "cardinality":        card,
                "associative_entity": assoc,
            }
            for s, t, _, card, assoc in self.relations
            if (pulp.value(yR.get((s, t), 0)) or 0) > 0.5
        ]

        score = pulp.value(prob.objective)
        return score, selected_entities, selected_relations, selected_attributes, runtime
