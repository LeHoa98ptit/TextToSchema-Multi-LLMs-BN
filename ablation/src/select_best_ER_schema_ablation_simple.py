"""
Ablation Study — Simple ILP for ER Schema Selection
=====================================================
Baseline ILP with only two structural constraints:

    1. A relationship can only be selected if BOTH endpoint entities are selected.
    2. An attribute can only be selected if its entity is selected.

No complexity penalties beyond the basic lambda_E / lambda_A / lambda_R weights
(no lambda_noattr, no lambda_NM, no lambda_isolated).

This is used as a comparison baseline against JointERILPComplexity
(select_best_ER_schema_ablation.py) to measure the contribution of the
entity-without-attributes and isolated-entity penalty terms.

Objective (maximise):
    Σ log_odds(p_e) * xE[e]       — entity scores
  + Σ log_odds(p_a) * zA[(e,a)]  — attribute scores
  + Σ log_odds(p_r) * yR[(s,t)]  — relationship scores
  - lambda_E * Σ xE[e]           — entity count penalty
  - lambda_A * Σ zA[(e,a)]       — attribute count penalty
  - lambda_R * Σ yR[(s,t)]       — relationship count penalty

Binary decision variables:
    xE[e]      1 iff entity e is selected
    zA[(e,a)]  1 iff attribute a of entity e is selected
    yR[(s,t)]  1 iff relationship (s,t) is selected

Constraints:
    • yR[(s,t)] ≤ xE[s]  and  yR[(s,t)] ≤ xE[t]
    • zA[(e,a)] ≤ xE[e]
    • Σ xE[e] ≥ min_entities
"""

import math
import re
import time
import pulp
from collections import defaultdict
from typing import Any, Dict, List, Tuple


class SimpleERILP:
    """
    Simple ILP for ER schema selection with only structural constraints.

    Compared to JointERILPComplexity, this class omits:
        • lambda_noattr   (entity-without-attribute penalty)
        • lambda_NM       (N:M relationship extra penalty)
        • lambda_isolated (isolated-entity penalty)
        • wE / iE indicator variables

    All entity/relation/attribute normalisation is identical so that
    probability inputs are treated consistently.
    """

    def __init__(
        self,
        entity_probs: Dict[str, float],
        relation_rows: List[Dict[str, Any]],
        attribute_probs: Dict[str, Any],
    ):
        """
        Args:
            entity_probs   : {entity_name: P(entity)}
            relation_rows  : list of dicts with entity_1/e1, entity_2/e2,
                             probability/p, cardinality, associative_entity
            attribute_probs: {entity_name: [(attr_name, prob), ...]}
                             or {entity_name: {attr_name: prob}}
        """
        # Normalise entity keys to uppercase.
        self.entities: Dict[str, float] = {
            e.upper(): v for e, v in entity_probs.items()
        }

        # Normalise and deduplicate relation rows.
        # Keep the highest-probability row for each (e1, e2) pair.
        # Skip rows whose endpoint is not in entity_probs (orphan edges).
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

        # Normalise attribute structure; skip orphan entity keys.
        self.attributes: Dict[str, List[Tuple[str, float]]] = {}
        for e, attrs in attribute_probs.items():
            eu = e.upper()
            if eu not in self.entities:
                continue
            if isinstance(attrs, dict):
                self.attributes[eu] = list(attrs.items())
            else:
                self.attributes[eu] = list(attrs)

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _pulp_name(s: str) -> str:
        """Sanitise a string for use as a PuLP variable/constraint name."""
        return re.sub(r'[^a-zA-Z0-9]', '_', str(s))

    @staticmethod
    def log_odds(p: float, eps: float = 1e-6) -> float:
        """Clip p to (eps, 1-eps) then return log(p / (1-p))."""
        p = max(min(float(p), 1.0 - eps), eps)
        return math.log(p / (1.0 - p))

    # ── Solver ────────────────────────────────────────────────────────────────

    def solve(
        self,
        lambda_E: float = 0.53,
        lambda_A: float = 0.72,
        lambda_R: float = 0.84,
        min_entities: int = 3,
    ) -> Tuple[float, list, list, dict, float]:
        """
        Solve the simple ILP.

        Args:
            lambda_E      : penalty per selected entity
            lambda_A      : penalty per selected attribute
            lambda_R      : penalty per selected relationship
            min_entities  : hard minimum number of selected entities

        Returns:
            (score, selected_entities, selected_relations,
             selected_attributes, runtime_s)
        """
        t0   = time.time()
        prob = pulp.LpProblem("ER_ILP_Simple", pulp.LpMaximize)

        entity_list = list(self.entities.keys())

        # ── Decision variables ────────────────────────────────────────────────

        # xE[e] — entity selection (indexed to prevent name collisions)
        xE = {
            e: pulp.LpVariable(f"xE_{i}_{self._pulp_name(e)}", cat="Binary")
            for i, e in enumerate(entity_list)
        }

        # zA[(e,a)] — attribute selection (doubly indexed)
        zA: Dict[Tuple[str, str], pulp.LpVariable] = {}
        for ei, (e, attrs) in enumerate(self.attributes.items()):
            for ai, (attr, _) in enumerate(attrs):
                zA[(e, attr)] = pulp.LpVariable(
                    f"zA_{ei}_{ai}_{self._pulp_name(e)}_{self._pulp_name(attr)}",
                    cat="Binary",
                )

        # yR[(s,t)] — relationship selection (indexed)
        yR: Dict[Tuple[str, str], pulp.LpVariable] = {}
        for ri, (s, t, _, _, _) in enumerate(self.relations):
            yR[(s, t)] = pulp.LpVariable(
                f"yR_{ri}_{self._pulp_name(s)}_{self._pulp_name(t)}", cat="Binary"
            )

        # ── Objective ─────────────────────────────────────────────────────────

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
                for s, t, p, _, _ in self.relations
            )
            - lambda_E * pulp.lpSum(xE.values())
            - lambda_A * pulp.lpSum(zA.values())
            - lambda_R * pulp.lpSum(yR.values())
        )

        # ── Constraints ───────────────────────────────────────────────────────

        # 1. Minimum entity count
        prob += pulp.lpSum(xE.values()) >= min_entities, "min_entities"

        # 2. Relationship requires both endpoint entities
        for i, (s, t, _, _, _) in enumerate(self.relations):
            prob += yR[(s, t)] <= xE[s], f"rel_e1_{i}"
            prob += yR[(s, t)] <= xE[t], f"rel_e2_{i}"

        # 3. Attribute requires its entity
        for ei, (e, attrs) in enumerate(self.attributes.items()):
            for ai, (attr, _) in enumerate(attrs):
                prob += zA[(e, attr)] <= xE[e], f"attr_own_{ei}_{ai}"

        # ── Solve ─────────────────────────────────────────────────────────────

        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        runtime = time.time() - t0

        # ── Extract solution ──────────────────────────────────────────────────

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
