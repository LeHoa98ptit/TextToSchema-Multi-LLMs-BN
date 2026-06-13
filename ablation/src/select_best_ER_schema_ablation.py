"""
Ablation Study — ILP-Based ER Schema Selection
================================================
Selects the best ER schema from a set of candidate entities, attributes, and
relationships by solving an Integer Linear Program (ILP) that jointly maximises
the log-odds probability of the schema while penalising complexity.

This is the canonical ILP used throughout the ablation pipeline.  It is
identical in structure to src/ilp_complexity.py (JointERILPComplexity) and is
placed here so the ablation module is fully self-contained.

Objective (maximise):
    Σ log_odds(p_e)  * xE[e]          — entity scores
  + Σ log_odds(p_a)  * zA[(e,a)]      — attribute scores
  + Σ log_odds(p_r)  * yR[(s,t)]      — relationship scores
  - lambda_E        * Σ xE[e]         — entity count penalty
  - lambda_A        * Σ zA[(e,a)]     — attribute count penalty
  - lambda_R        * Σ yR[(s,t)]     — relationship count penalty
  - lambda_noattr   * Σ wE[e]         — penalty: selected entity with 0 attributes
  - lambda_NM       * Σ_{N:M} yR[…]  — extra penalty for N:M relationships
  - lambda_isolated * Σ iE[e]         — penalty: selected entity with 0 relationships

Binary decision variables:
    xE[e]        1 iff entity e is selected
    zA[(e,a)]    1 iff attribute a of entity e is selected
    yR[(s,t)]    1 iff relationship (s,t) is selected
    wE[e]        1 iff entity e selected AND has no selected attributes
    iE[e]        1 iff entity e selected AND has no selected relationships

Constraints:
    • yR[(s,t)] ≤ xE[s]  and  yR[(s,t)] ≤ xE[t]   (relationship → both endpoints)
    • zA[(e,a)] ≤ xE[e]                              (attribute → its entity)
    • wE[e] ≤ xE[e]  and  wE[e] ≥ xE[e] − Σ_a zA   (entity-without-attr indicator)
    • iE[e] ≤ xE[e]  and  iE[e] ≥ xE[e] − Σ_r yR   (isolated-entity indicator)
    • Σ xE[e] ≥ min_entities                         (minimum entity count)
"""

import math
import re
import time
import pulp
from collections import defaultdict
from typing import Any, Dict, List, Tuple


class JointERILPComplexity:
    """
    Complexity-aware ILP for joint ER schema selection.

    Parameters passed at construction time define the candidate pool.
    Parameters passed to solve() control the penalty trade-offs.
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
            relation_rows  : list of dicts; each must contain at least
                             entity_1/e1, entity_2/e2, probability/p,
                             cardinality, associative_entity
            attribute_probs: {entity_name: [(attr_name, prob), ...]}
                             or {entity_name: {attr_name: prob}}
        """
        # Normalise entity keys to uppercase so they match uppercased relation endpoints.
        self.entities = {e.upper(): v for e, v in entity_probs.items()}

        # Normalise relation rows into a uniform tuple list.
        # Deduplicate by (e1, e2) key — keep the row with the highest probability
        # to avoid overlapping PuLP constraint names for duplicate pairs.
        # Also drop rows where either endpoint is not in entity_probs (orphan edges).
        _seen: Dict[Tuple[str, str], Tuple] = {}
        for r in relation_rows:
            e1   = r.get("e1",   r.get("entity_1",   "")).upper()
            e2   = r.get("e2",   r.get("entity_2",   "")).upper()
            if e1 not in self.entities or e2 not in self.entities:
                continue  # skip orphan relationships
            prob = float(r.get("p", r.get("probability", 0.0)))
            card = r.get("cardinality", "1:N")
            assoc = r.get("associative_entity", None)
            key = (e1, e2)
            if key not in _seen or prob > _seen[key][2]:
                _seen[key] = (e1, e2, prob, card, assoc)
        self.relations: List[Tuple[str, str, float, str, Any]] = list(_seen.values())

        # Normalise attribute structure to {entity: [(attr, prob), ...]}
        # Uppercase entity keys; skip entities not in self.entities (orphan attrs).
        self.attributes: Dict[str, List[Tuple[str, float]]] = {}
        for e, attrs in attribute_probs.items():
            eu = e.upper()
            if eu not in self.entities:
                continue  # skip attributes for entities absent from entity_probs
            if isinstance(attrs, dict):
                self.attributes[eu] = list(attrs.items())
            else:
                self.attributes[eu] = list(attrs)

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _pulp_name(s: str) -> str:
        """Replace any non-alphanumeric character with underscore for PuLP names."""
        return re.sub(r'[^a-zA-Z0-9]', '_', str(s))

    @staticmethod
    def log_odds(p: float, eps: float = 1e-6) -> float:
        """Clip p to (eps, 1−eps) then compute log(p / (1−p))."""
        p = max(min(float(p), 1.0 - eps), eps)
        return math.log(p / (1.0 - p))

    # ── Solver ────────────────────────────────────────────────────────────────

    def solve(
        self,
        lambda_E: float        = 0.8,
        lambda_A: float        = 0.6,
        lambda_R: float        = 0.85,
        lambda_noattr: float   = 1.3,
        lambda_NM: float       = 0.72,
        lambda_isolated: float = 1.31,
        min_entities: int      = 3,
    ) -> Tuple[float, list, list, dict, float]:
        """
        Solve the complexity-aware ILP.

        Args:
            lambda_E        : penalty weight per selected entity
            lambda_A        : penalty weight per selected attribute
            lambda_R        : penalty weight per selected relationship
            lambda_noattr   : penalty for each entity that has no selected attributes
            lambda_NM       : extra penalty for each N:M relationship (stacked on lambda_R)
            lambda_isolated : penalty for each entity that has no selected relationships
            min_entities    : hard lower bound on number of selected entities

        Returns:
            (score, selected_entities, selected_relations, selected_attributes, runtime_s)
              score               : ILP objective value
              selected_entities   : list[str]
              selected_relations  : list[{entity_1, entity_2, cardinality, associative_entity}]
              selected_attributes : {entity: [attr_name, ...]}
              runtime_s           : wall-clock seconds
        """
        t0   = time.time()
        prob = pulp.LpProblem("ER_ILP_Ablation", pulp.LpMaximize)

        # ── Decision variables ────────────────────────────────────────────────
        # PuLP variable/constraint names must be alphanumeric+underscore.
        # We use _pulp_name() to sanitize entity/attr strings and add numeric
        # indices where needed to guarantee uniqueness even when sanitized names
        # collide (e.g. "user id" and "user_id" both map to "user_id").

        # xE[e]     — entity selection (indexed to avoid any case-collision)
        entity_list = list(self.entities.keys())
        xE = {
            e: pulp.LpVariable(f"xE_{i}_{self._pulp_name(e)}", cat="Binary")
            for i, e in enumerate(entity_list)
        }

        # zA[(e,a)] — attribute selection (doubly-indexed: entity idx, attr idx)
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

        # wE[e] — entity-without-attribute indicator (indexed)
        wE = {
            e: pulp.LpVariable(f"wE_{i}_{self._pulp_name(e)}", cat="Binary")
            for i, e in enumerate(entity_list)
        }

        # iE[e] — isolated-entity indicator (indexed)
        iE = {
            e: pulp.LpVariable(f"iE_{i}_{self._pulp_name(e)}", cat="Binary")
            for i, e in enumerate(entity_list)
        }

        # ── Pre-computed indices ──────────────────────────────────────────────

        # entity → list of (e, attr) attribute keys
        attrs_by_entity: Dict[str, list] = defaultdict(list)
        for (e, attr) in zA:
            attrs_by_entity[e].append((e, attr))

        # entity → list of (s, t) relationship keys incident to it
        rels_by_entity: Dict[str, list] = defaultdict(list)
        for s, t, *_ in self.relations:
            rels_by_entity[s].append((s, t))
            rels_by_entity[t].append((s, t))

        # ── Objective ─────────────────────────────────────────────────────────

        prob += (
            # Probability terms (log-odds ensures numerically stable scoring)
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
            # Complexity penalties
            - lambda_E        * pulp.lpSum(xE.values())
            - lambda_A        * pulp.lpSum(zA.values())
            - lambda_R        * pulp.lpSum(yR.values())
            - lambda_noattr   * pulp.lpSum(wE.values())
            - lambda_NM       * pulp.lpSum(
                yR[(s, t)]
                for s, t, _, card, _ in self.relations
                if card == "N:M"
            )
            - lambda_isolated * pulp.lpSum(iE.values())
        )

        # ── Constraints ───────────────────────────────────────────────────────

        # 1. Minimum entity count
        prob += pulp.lpSum(xE.values()) >= min_entities, "min_entities"

        # 2. Relationship endpoints: both entities must be selected
        # Use index i to guarantee unique constraint names even if entity names collide
        for i, (s, t, _, _, _) in enumerate(self.relations):
            prob += yR[(s, t)] <= xE[s], f"rel_e1_{i}_{s}_{t}"
            prob += yR[(s, t)] <= xE[t], f"rel_e2_{i}_{s}_{t}"

        # 3. Attribute ownership: attribute requires its entity to be selected
        # Use (entity_idx, attr_idx) in the constraint name to avoid collisions
        # when sanitized attribute names are identical across entities.
        for ei, (e, attrs) in enumerate(self.attributes.items()):
            for ai, (attr, _) in enumerate(attrs):
                prob += zA[(e, attr)] <= xE[e], f"attr_own_{ei}_{ai}"

        # 4. Entity-without-attributes indicator (wE)
        #    wE[e] ≤ xE[e]                     — only active when entity selected
        #    wE[e] ≥ xE[e] − Σ_a zA[(e,a)]    — forced to 1 when no attr selected
        for ei, e in enumerate(entity_list):
            prob += wE[e] <= xE[e], f"noattr_ub_{ei}"
            attr_keys = attrs_by_entity[e]
            if attr_keys:
                prob += (
                    wE[e] >= xE[e] - pulp.lpSum(zA[k] for k in attr_keys),
                    f"noattr_lb_{ei}",
                )
            else:
                # No candidate attributes → entity will always be "bare"
                prob += wE[e] >= xE[e], f"noattr_lb_nokeys_{ei}"

        # 5. Isolated-entity indicator (iE)
        #    iE[e] ≤ xE[e]                          — only active when entity selected
        #    iE[e] ≥ xE[e] − Σ_{incident r} yR[r]  — forced to 1 when no rel selected
        for ei, e in enumerate(entity_list):
            prob += iE[e] <= xE[e], f"isolated_ub_{ei}"
            incident = rels_by_entity[e]
            if incident:
                prob += (
                    iE[e] >= xE[e] - pulp.lpSum(yR[k] for k in incident),
                    f"isolated_lb_{ei}",
                )
            else:
                # Entity has no candidate relationships → always isolated
                prob += iE[e] >= xE[e], f"isolated_lb_norels_{ei}"

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
                "entity_1":         s,
                "entity_2":         t,
                "cardinality":      card,
                "associative_entity": assoc,
            }
            for s, t, _, card, assoc in self.relations
            if (pulp.value(yR.get((s, t), 0)) or 0) > 0.5
        ]

        score = pulp.value(prob.objective)
        return score, selected_entities, selected_relations, selected_attributes, runtime
