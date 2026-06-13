"""
JointERILPComplexity
====================
ILP that jointly maximizes log-odds probability and penalizes ER schema
complexity.  Four complexity metrics are encoded as penalty terms:

  1. Relationships per entity   → lambda_R  * total_selected_relationships
  2. Attributes per entity      → lambda_A  * total_selected_attributes
  3. Entities without attributes→ lambda_noattr * count(selected entities with 0 attrs)
  4. N:M relationships          → lambda_NM       * count(selected N:M relationships)
  5. Isolated entities          → lambda_isolated * count(selected entities with 0 rels)

Objective:
  max   Σ log_odds(p_e)*xE[e]
      + Σ log_odds(p_a)*zA[(e,a)]
      + Σ log_odds(p_r)*yR[(s,t)]
      - lambda_E        * Σ xE[e]
      - lambda_A        * Σ zA[(e,a)]
      - lambda_R        * Σ yR[(s,t)]
      - lambda_noattr   * Σ wE[e]
      - lambda_NM       * Σ_{card=N:M} yR[(s,t)]
      - lambda_isolated * Σ iE[e]

where wE[e] = 1 iff entity e is selected AND has no selected attributes.
      iE[e] = 1 iff entity e is selected AND has no selected relationships.
"""

import math
import time
import pulp
from collections import defaultdict
from typing import Dict, List, Any, Tuple


class JointERILPComplexity:

    def __init__(
        self,
        entity_probs: Dict[str, float],
        relation_rows: List[Dict[str, Any]],
        attribute_probs: Dict[str, Any],
    ):
        """
        Args:
            entity_probs   : {entity_name: probability}
            relation_rows  : list of dicts with keys entity_1/e1, entity_2/e2,
                             probability/p, cardinality, associative_entity
            attribute_probs: {entity_name: [(attr_name, prob), ...]}
                             or {entity_name: {attr_name: prob}}
        """
        self.entities = entity_probs

        self.relations: List[Tuple[str, str, float, str, Any]] = [
            (
                r.get("e1", r.get("entity_1", "")).upper(),
                r.get("e2", r.get("entity_2", "")).upper(),
                float(r.get("p", r.get("probability", 0.0))),
                r.get("cardinality", "1:N"),
                r.get("associative_entity", None),
            )
            for r in relation_rows
        ]

        self.attributes: Dict[str, List[Tuple[str, float]]] = {}
        for e, attrs in attribute_probs.items():
            if isinstance(attrs, dict):
                self.attributes[e] = list(attrs.items())
            else:
                self.attributes[e] = list(attrs)

    # ------------------------------------------------------------------
    @staticmethod
    def log_odds(p: float, eps: float = 1e-6) -> float:
        p = float(p)
        p = max(min(p, 1.0 - eps), eps)
        return math.log(p / (1.0 - p))

    # ------------------------------------------------------------------
    def solve(
        self,
        lambda_E: float = 1.0,
        lambda_A: float = 1.0,
        lambda_R: float = 1.0,
        lambda_noattr: float = 2.0,
        lambda_NM: float = 1.5,
        lambda_isolated: float = 2.0,
        min_entities: int = 3,
    ) -> Tuple[float, list, list, dict, float]:
        """
        Solve the complexity-aware ILP.

        Parameters
        ----------
        lambda_E      : penalty per selected entity
        lambda_A      : penalty per selected attribute  → penalises high attr/entity
        lambda_R      : penalty per selected relationship → penalises high rel/entity
        lambda_noattr   : penalty per entity that has no selected attributes
        lambda_NM       : extra penalty per N:M relationship (on top of lambda_R)
        lambda_isolated : penalty per entity that has no selected relationships
        min_entities    : hard lower bound on entity count

        Returns
        -------
        (score, selected_entities, selected_relations, selected_attributes, runtime)
          selected_entities  : list[str]
          selected_relations : list[{entity_1, entity_2, cardinality, associative_entity}]
          selected_attributes: {entity: [attr_name, ...]}
          runtime            : wall-clock seconds
        """
        start = time.time()
        prob = pulp.LpProblem("ER_ILP_Complexity", pulp.LpMaximize)

        # ── Decision variables ────────────────────────────────────────
        xE = {e: pulp.LpVariable(f"xE_{e}", cat="Binary") for e in self.entities}

        zA: Dict[Tuple[str, str], pulp.LpVariable] = {}
        for e, attrs in self.attributes.items():
            for attr, _ in attrs:
                zA[(e, attr)] = pulp.LpVariable(f"zA_{e}_{attr}", cat="Binary")

        yR: Dict[Tuple[str, str], pulp.LpVariable] = {}
        for s, t, _, _, _ in self.relations:
            yR[(s, t)] = pulp.LpVariable(f"yR_{s}_{t}", cat="Binary")

        # wE[e] = 1  iff  entity selected  AND  no attribute selected for it
        wE = {e: pulp.LpVariable(f"wE_{e}", cat="Binary") for e in self.entities}

        # iE[e] = 1  iff  entity selected  AND  no relationship selected for it
        iE = {e: pulp.LpVariable(f"iE_{e}", cat="Binary") for e in self.entities}

        # index: entity → its (e, attr) keys in zA
        attrs_by_entity: Dict[str, list] = defaultdict(list)
        for key in zA:
            attrs_by_entity[key[0]].append(key)

        # index: entity → relationship keys incident to it
        rels_by_entity: Dict[str, list] = defaultdict(list)
        for s, t, *_ in self.relations:
            rels_by_entity[s].append((s, t))
            rels_by_entity[t].append((s, t))

        # ── Objective ─────────────────────────────────────────────────
        prob += (
            # probability score
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
            # complexity penalties
            - lambda_E * pulp.lpSum(xE.values())
            - lambda_A * pulp.lpSum(zA.values())
            - lambda_R * pulp.lpSum(yR.values())
            - lambda_noattr * pulp.lpSum(wE.values())
            - lambda_NM * pulp.lpSum(
                yR[(s, t)]
                for s, t, _, card, _ in self.relations
                if card == "N:M"
            )
            - lambda_isolated * pulp.lpSum(iE.values())
        )

        # ── Constraints ───────────────────────────────────────────────
        # Minimum entity count
        prob += pulp.lpSum(xE.values()) >= min_entities

        # Relationship ⇒ both endpoint entities must be selected
        for s, t, _, _, _ in self.relations:
            prob += yR[(s, t)] <= xE[s]
            prob += yR[(s, t)] <= xE[t]

        # Attribute ⇒ its entity must be selected
        for e, attrs in self.attributes.items():
            for attr, _ in attrs:
                prob += zA[(e, attr)] <= xE[e]

        # "Entity without attributes" indicator
        #   wE[e] <= xE[e]                           only active when entity selected
        #   wE[e] >= xE[e] - Σ_a zA[(e,a)]          forced to 1 if no attr selected
        for e in self.entities:
            prob += wE[e] <= xE[e]
            keys = attrs_by_entity[e]
            if keys:
                prob += wE[e] >= xE[e] - pulp.lpSum(zA[k] for k in keys)
            else:
                prob += wE[e] >= xE[e]  # no candidate attributes → always bare

        # "Isolated entity" indicator
        #   iE[e] <= xE[e]                               only active when entity selected
        #   iE[e] >= xE[e] - Σ_{rel incident e} yR[rel]  forced to 1 if no rel selected
        for e in self.entities:
            prob += iE[e] <= xE[e]
            incident = rels_by_entity[e]
            if incident:
                prob += iE[e] >= xE[e] - pulp.lpSum(yR[k] for k in incident)
            else:
                prob += iE[e] >= xE[e]  # no candidate relationships → always isolated

        # ── Solve ─────────────────────────────────────────────────────
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        runtime = time.time() - start

        # ── Extract solution ──────────────────────────────────────────
        selected_entities = [e for e in xE if pulp.value(xE[e]) > 0.5]

        selected_attributes = {
            e: [
                attr
                for attr, _ in self.attributes[e]
                if pulp.value(zA[(e, attr)]) > 0.5
            ]
            for e in selected_entities
            if e in self.attributes
        }

        selected_relations = []
        for s, t, _, card, assoc in self.relations:
            if pulp.value(yR[(s, t)]) > 0.5:
                selected_relations.append(
                    {
                        "entity_1": s,
                        "entity_2": t,
                        "cardinality": card,
                        "associative_entity": assoc,
                    }
                )

        score = pulp.value(prob.objective)
        return score, selected_entities, selected_relations, selected_attributes, runtime
