# procedural/conceptual/metacognitive per concept
# cognition/belief_model.py
"""
Live model of what the student currently believes.
Updated after every exchange via background Gemini call.
This is the file that does what you did manually in your notebook.

Structure per concept:
  procedural_confidence:  can execute the steps
  conceptual_confidence:  understands WHY
  metacognitive_awareness: knows what they don't know
  root_beliefs:           false beliefs that generate surface errors
  error_history:          type and count of errors
  stability:              spaced repetition parameter
"""
import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timedelta

from google import genai
from google.genai import types
from config import EXTRACT_MODEL, SR_INITIAL_STABILITY, SR_STABILITY_GROWTH

logger  = logging.getLogger("SYRA.BeliefModel")
_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


@dataclass
class RootBelief:
    belief:     str
    confidence: float   # how strongly held — 0.0 to 1.0
    resolved:   bool    = False
    first_seen: str     = field(default_factory=lambda: datetime.now().isoformat())
    evidence:   list    = field(default_factory=list)


@dataclass
class ConceptNode:
    concept:                str
    procedural_confidence:  float = 0.50
    conceptual_confidence:  float = 0.20
    metacognitive_awareness: float = 0.30
    root_beliefs:           list[RootBelief] = field(default_factory=list)
    error_history:          dict = field(default_factory=dict)
    stability:              float = SR_INITIAL_STABILITY
    last_reviewed:          str = field(default_factory=lambda: datetime.now().isoformat())
    next_review_due:        str = field(default_factory=lambda: datetime.now().isoformat())
    total_sessions:         int = 0


@dataclass
class BeliefSnapshot:
    """What gets passed to the prompt assembler each turn."""
    concepts:       dict[str, ConceptNode] = field(default_factory=dict)
    active_concept: Optional[str] = None
    snapshot_time:  str = field(default_factory=lambda: datetime.now().isoformat())


UPDATE_PROMPT = """A tutoring exchange just occurred. Update the belief model.

Concept being studied: {concept}

Current belief state:
{current_state}

Exchange:
Student said: "{student_text}"
SYRA responded: "{ai_text}"

Return ONLY valid JSON:
{{
  "procedural_confidence_delta": 0.0,
  "conceptual_confidence_delta": 0.0,
  "metacognitive_awareness_delta": 0.0,
  "new_root_belief": null,
  "root_belief_resolved": null,
  "explanation": "one sentence summary of what changed"
}}

Rules:
- Deltas range -0.15 to +0.15 per exchange
- new_root_belief: string if a false underlying belief was revealed, null otherwise
- root_belief_resolved: string matching an existing belief if it was corrected
- Be conservative — small confident updates are better than large uncertain ones
- OVERLOAD errors (student clearly knew but made execution mistake) should give 0.0 delta
- Procedural delta positive only if student demonstrated correct execution independently"""


SEED_PROMPT = """Extract initial belief state from this onboarding response.

Question: {question}
Student answered: {answer}

Concept domain: {subject}

Return ONLY valid JSON:
{{
  "concept": "main concept identified",
  "procedural_confidence": 0.0,
  "conceptual_confidence": 0.0,
  "metacognitive_awareness": 0.0,
  "root_beliefs": [],
  "explanation": "one sentence"
}}

Scores 0.0–1.0. Be conservative — first session data is uncertain.
root_beliefs: list of strings, false beliefs revealed in the answer."""


class BeliefModel:
    """
    Manages the live belief state for the current session.
    Reads from and writes to belief_graph.json via belief_graph.py.
    """

    def __init__(self, belief_graph: dict):
        """
        belief_graph: loaded from memory/belief_graph.py
        """
        self._graph      = belief_graph   # reference — mutations are visible
        self._snapshot   = BeliefSnapshot()
        self._pending_updates: list[dict] = []

    def get_snapshot(self, active_concept: Optional[str] = None) -> str:
        """
        Returns a compact belief snapshot string for the prompt assembler.
        Describes what the student believes RIGHT NOW.
        """
        if not self._graph:
            return "No prior belief data — first session."

        lines = []
        for concept, node_dict in self._graph.get("concepts", {}).items():
            proc  = node_dict.get("procedural_confidence", 0.5)
            conc  = node_dict.get("conceptual_confidence", 0.2)
            meta  = node_dict.get("metacognitive_awareness", 0.3)
            roots = node_dict.get("root_beliefs", [])
            active_marker = " [ACTIVE]" if concept == active_concept else ""

            lines.append(
                f"{concept}{active_marker}: "
                f"procedure={proc:.0%} concept={conc:.0%} meta={meta:.0%}"
            )
            for rb in roots:
                if not rb.get("resolved", False):
                    lines.append(
                        f"  ROOT BELIEF: \"{rb['belief']}\" "
                        f"(confidence {rb.get('confidence', 0.7):.0%})"
                    )

        return "\n".join(lines) if lines else "Building belief model..."

    def update_from_exchange(
            self,
            student_text: str,
            ai_text:      str,
            concept:      str,
            error_type:   str = "NONE",
    ):
        """
        Background-safe belief update from one exchange.
        Called from a background thread — never blocks conversation.
        error_type from error_classifier: NONE|CARELESS|PROCEDURAL|CONCEPTUAL|OVERLOAD_INDUCED
        """
        if error_type == "CARELESS":
            # Careless errors carry no belief information
            # They indicate execution slip, not knowledge gap
            return
        if error_type == "OVERLOAD_INDUCED":
            # Student knows the material — don't penalise belief scores
            # Just log the occurrence
            self._log_error(concept, error_type)
            return

        current_node = self._graph.get("concepts", {}).get(concept, {})
        current_state_str = json.dumps(current_node, indent=2) if current_node else "{}"

        try:
            resp = _client.models.generate_content(
                model=EXTRACT_MODEL,
                contents=UPDATE_PROMPT.format(
                    concept=concept,
                    current_state=current_state_str,
                    student_text=student_text,
                    ai_text=ai_text,
                ),
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                    thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                    max_output_tokens=300,
                ),
            )
            update = json.loads(resp.text)
            self._apply_update(concept, update, error_type)

        except Exception as e:
            logger.debug(f"Belief update failed for {concept}: {e}")

    def _apply_update(self, concept: str, update: dict, error_type: str):
        """Apply validated deltas to the graph."""
        concepts = self._graph.setdefault("concepts", {})
        node     = concepts.setdefault(concept, {
            "procedural_confidence":   0.50,
            "conceptual_confidence":   0.20,
            "metacognitive_awareness": 0.30,
            "root_beliefs":            [],
            "error_history":           {},
            "stability":               SR_INITIAL_STABILITY,
            "last_reviewed":           datetime.now().isoformat(),
            "next_review_due":         datetime.now().isoformat(),
        })

        # Apply deltas with bounds
        def apply_delta(key: str, delta_key: str):
            delta = float(update.get(delta_key, 0.0))
            delta = max(-0.15, min(0.15, delta))   # hard clamp
            node[key] = float(
                max(0.0, min(1.0, node.get(key, 0.5) + delta))
            )

        apply_delta("procedural_confidence",   "procedural_confidence_delta")
        apply_delta("conceptual_confidence",   "conceptual_confidence_delta")
        apply_delta("metacognitive_awareness", "metacognitive_awareness_delta")

        # New root belief found
        new_rb = update.get("new_root_belief")
        if new_rb:
            existing = [
                rb["belief"] for rb in node.get("root_beliefs", [])
            ]
            if new_rb not in existing:
                node.setdefault("root_beliefs", []).append({
                    "belief":     new_rb,
                    "confidence": 0.70,
                    "resolved":   False,
                    "first_seen": datetime.now().isoformat(),
                    "evidence":   [f"turn: {student_text[:60]}"]
                    if "student_text" in dir() else [],
                })
                logger.info(f"Root belief found in {concept}: {new_rb}")

        # Root belief resolved
        resolved = update.get("root_belief_resolved")
        if resolved:
            for rb in node.get("root_beliefs", []):
                if rb.get("belief") == resolved:
                    rb["resolved"] = True
                    logger.info(f"Root belief resolved in {concept}: {resolved}")

        # Error history
        if error_type and error_type != "NONE":
            node.setdefault("error_history", {})[error_type] = (
                node["error_history"].get(error_type, 0) + 1
            )

        # Update review timestamp
        node["last_reviewed"] = datetime.now().isoformat()

    def _log_error(self, concept: str, error_type: str):
        """Log error type without updating belief scores."""
        node = self._graph.get("concepts", {}).get(concept, {})
        if node:
            node.setdefault("error_history", {})[error_type] = (
                node["error_history"].get(error_type, 0) + 1
            )

    def seed_from_onboarding(
            self,
            question: str,
            answer:   str,
            subject:  str,
    ):
        """
        Populate initial belief state from onboarding Q&A.
        Used for ONBOARDING_BELIEF_Qs questions only.
        Gives conservative initial estimates.
        """
        if not answer or len(answer) < 10:
            return
        try:
            resp = _client.models.generate_content(
                model=EXTRACT_MODEL,
                contents=SEED_PROMPT.format(
                    question=question,
                    answer=answer,
                    subject=subject,
                ),
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                    thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                    max_output_tokens=300,
                ),
            )
            data    = json.loads(resp.text)
            concept = data.get("concept", "general")
            concepts = self._graph.setdefault("concepts", {})

            if concept not in concepts:
                concepts[concept] = {
                    "procedural_confidence":   data.get("procedural_confidence",   0.5),
                    "conceptual_confidence":   data.get("conceptual_confidence",   0.2),
                    "metacognitive_awareness": data.get("metacognitive_awareness", 0.3),
                    "root_beliefs": [
                        {"belief": rb, "confidence": 0.6,
                         "resolved": False,
                         "first_seen": datetime.now().isoformat()}
                        for rb in data.get("root_beliefs", [])
                    ],
                    "error_history": {},
                    "stability":    SR_INITIAL_STABILITY,
                    "last_reviewed": datetime.now().isoformat(),
                    "next_review_due": datetime.now().isoformat(),
                }
                logger.info(
                    f"Seeded belief graph: {concept} "
                    f"proc={concepts[concept]['procedural_confidence']:.0%}"
                )
        except Exception as e:
            logger.debug(f"Belief seed failed: {e}")