# RETRIEVAL -> COMPREHENSION -> CAUSAL -> TRANSFER
# pedagogy/question_planner.py
"""
Elaborative interrogation taxonomy.
Tracks which question level this student can handle.
Slowly pushes toward causal level as procedural confidence grows.

RETRIEVAL:      "What is the formula for splitting?"
COMPREHENSION:  "How do you decide which numbers to split into?"
CAUSAL:         "Why does the product have to equal ac, not just any pair?"
TRANSFER:       "What would change if the leading coefficient were negative?"
WIN_CREATING:   "You said -3 and -4 — what do those multiply to?"
"""
from dataclasses import dataclass
from typing import Optional
from cognition.zpd_estimator  import ZPDEstimate
from cognition.affect_tracker import AffectState
from cognition.belief_model   import BeliefModel


@dataclass
class QuestionPlan:
    question_type:     str
    rationale:         str
    causal_target:     Optional[str]   # root belief to target if CAUSAL
    instruction:       str             # for prompt assembler


class QuestionPlanner:

    def __init__(self):
        self._turn_history: list[str] = []   # history of question types asked

    def plan(
            self,
            zpd:             ZPDEstimate,
            affect:          AffectState,
            belief_model:    BeliefModel,
            active_concept:  str,
            turn_num:        int,
            gate_allows:     list[str],   # from situational_gate
    ) -> QuestionPlan:
        """
        Decide what type of question SYRA should ask next.
        Respects situational gate restrictions.
        """
        concept_data  = belief_model._graph.get(
            "concepts", {}
        ).get(active_concept, {})
        proc_conf     = concept_data.get("procedural_confidence", 0.5)
        conc_conf     = concept_data.get("conceptual_confidence", 0.2)
        unresolved_rb = [
            rb for rb in concept_data.get("root_beliefs", [])
            if not rb.get("resolved", False)
        ]

        # If WIN_CREATING is forced by gate, return immediately
        if gate_allows == ["WIN_CREATING"]:
            return QuestionPlan(
                question_type="WIN_CREATING",
                rationale="Gate forced: create immediate success",
                causal_target=None,
                instruction=(
                    "Ask a question the student can answer correctly "
                    "with what they already know. "
                    "Reference something they said earlier that was right. "
                    "'You mentioned X — what does that give us?'"
                ),
            )

        # If SOCRATIC forced by dependency alarm
        if gate_allows == ["SOCRATIC"]:
            return QuestionPlan(
                question_type="SOCRATIC",
                rationale="Dependency alarm: force independent production",
                causal_target=None,
                instruction=(
                    "Ask one open question that requires the student to "
                    "produce an answer without any hints in the question itself. "
                    "Wait. Do not add hints or rephrase if they pause."
                ),
            )

        # Determine natural level based on belief state + ZPD
        if zpd.position == "ABOVE" or affect.frustration_type == "STUCK":
            natural_level = "COMPREHENSION"

        elif proc_conf < 0.50:
            natural_level = "RETRIEVAL"

        elif proc_conf >= 0.50 and conc_conf < 0.30:
            # Good procedure, weak concept — push toward causal
            natural_level = "COMPREHENSION" if turn_num < 5 else "CAUSAL"

        elif proc_conf >= 0.70 and conc_conf >= 0.40:
            natural_level = "CAUSAL" if not unresolved_rb else "CAUSAL"

        elif proc_conf >= 0.80 and conc_conf >= 0.60:
            natural_level = "TRANSFER"

        else:
            natural_level = "COMPREHENSION"

        # Respect gate restrictions
        if natural_level not in gate_allows and gate_allows:
            natural_level = gate_allows[0]

        # Causal target — aim at unresolved root belief
        causal_target = None
        if natural_level == "CAUSAL" and unresolved_rb:
            causal_target = unresolved_rb[0]["belief"]

        self._turn_history.append(natural_level)

        instruction = self._build_instruction(
            natural_level, active_concept, causal_target, proc_conf, conc_conf
        )

        return QuestionPlan(
            question_type=natural_level,
            rationale=f"proc={proc_conf:.0%} conc={conc_conf:.0%} zpd={zpd.position}",
            causal_target=causal_target,
            instruction=instruction,
        )

    def _build_instruction(
            self,
            q_type:        str,
            concept:       str,
            causal_target: Optional[str],
            proc_conf:     float,
            conc_conf:     float,
    ) -> str:
        if q_type == "RETRIEVAL":
            return (
                f"Ask a retrieval question about {concept}. "
                "Test recall of a specific step or definition. "
                "Keep it simple and answerable."
            )
        if q_type == "COMPREHENSION":
            return (
                f"Ask how the student applies a step in {concept}. "
                "Not what it is — how and when to use it."
            )
        if q_type == "CAUSAL":
            if causal_target:
                return (
                    f"Target this false belief: '{causal_target}'. "
                    "Do NOT directly correct it. "
                    "Ask a question that creates cognitive conflict — "
                    "where their false belief produces a prediction "
                    "they can test and find wrong."
                )
            return (
                f"Ask why a step in {concept} works, not just how. "
                "Push toward conceptual understanding."
            )
        if q_type == "TRANSFER":
            return (
                f"Pose a novel variation of {concept}. "
                "Change one parameter and ask what happens. "
                "Do not solve it — let them figure it out."
            )
        return "Ask an appropriate question to continue the lesson."