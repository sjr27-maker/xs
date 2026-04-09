# permanent profile, drift protection, anomaly gate
# memory/profile_manager.py
"""
Permanent student profile with drift-protected updates.
Three-gate system: consistency + magnitude + context.
New gate: session_anomaly blocks ALL updates.
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
from config import (
    ALPHA_SESSION_1, ALPHA_SESSION_10_PLUS,
    DRIFT_GATE_SESSIONS, DRIFT_MAGNITUDE,
    ANOMALY_BLOCK_ALL,
)

logger = logging.getLogger("SYRA.ProfileManager")


class ProfileManager:

    def __init__(self, student_id: str):
        self.student_id = student_id
        self._path      = Path(f"sessions/{student_id}/student_profile.json")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self.profile    = self._load()
        self._session_count = len(self.profile.get("session_history", []))

    def _load(self) -> dict:
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def save(self):
        self.profile["last_saved"] = datetime.now().isoformat()
        self._path.write_text(
            json.dumps(self.profile, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        logger.debug(f"Profile saved: {self.student_id}")

    def get_alpha(self) -> float:
        """
        α controls how much the base profile weights vs live data.
        Starts high (trust onboarding), decreases as sessions accumulate.
        """
        n = self._session_count
        if   n <= 1:   return ALPHA_SESSION_1
        elif n <= 5:   return 0.65
        elif n <= 10:  return 0.45
        else:          return ALPHA_SESSION_10_PLUS

    def get_session_adapted_ipc(self, live_ipc: dict) -> dict:
        """
        α-blend: adapted = α × base + (1-α) × live
        Returns session-only adapted IPC.
        Never writes back to permanent profile here.
        """
        α    = self.get_alpha()
        base = self.profile.get("ipc", {})

        return {
            "dominance":  round(
                α * base.get("dominance", 0.5)
                + (1 - α) * live_ipc.get("dominance", 0.5), 3
            ),
            "warmth":     round(
                α * base.get("warmth", 0.6)
                + (1 - α) * live_ipc.get("warmth", 0.6), 3
            ),
            "pace":       live_ipc.get("pace", base.get("pace", "medium")),
            "giving_up":  live_ipc.get("giving_up", False),
            "filler_count": live_ipc.get("filler_count", 0),
            "assertiveness_delta": live_ipc.get("assertiveness_delta", 0.0),
            "archetype":  base.get("archetype", "lina"),
        }

    def update_base_profile(
            self,
            ipc_summary:   dict,
            turn_count:    int,
            session_anomaly: bool = False,
    ):
        """
        Drift-protected update of permanent IPC profile.
        Gates:
          1. Not anomaly session
          2. Enough turns (≥5)
          3. Magnitude of change > threshold
          4. Consistent direction (tracked via evidence buffer)
        """
        if session_anomaly and ANOMALY_BLOCK_ALL:
            logger.info("Profile update BLOCKED — anomaly session")
            self._log_anomaly(ipc_summary)
            return

        if turn_count < 5:
            logger.debug("Profile update BLOCKED — too few turns")
            return

        base_dom  = self.profile.get("ipc", {}).get("dominance", 0.5)
        base_warm = self.profile.get("ipc", {}).get("warmth",    0.6)
        new_dom   = ipc_summary.get("avg_dominance", base_dom)
        new_warm  = ipc_summary.get("avg_warmth",    base_warm)

        # Gate 2: magnitude
        dom_delta  = abs(new_dom  - base_dom)
        warm_delta = abs(new_warm - base_warm)
        if dom_delta < DRIFT_MAGNITUDE and warm_delta < DRIFT_MAGNITUDE:
            logger.debug("Profile update BLOCKED — below drift threshold")
            return

        # Gate 3: consistency via evidence buffer
        buf = self.profile.setdefault("_evidence_buffer", {
            "dominance": [], "warmth": []
        })
        buf["dominance"].append(new_dom)
        buf["warmth"].append(new_warm)
        buf["dominance"] = buf["dominance"][-8:]
        buf["warmth"]    = buf["warmth"][-8:]

        if len(buf["dominance"]) < DRIFT_GATE_SESSIONS:
            logger.debug("Profile update PENDING — building evidence")
            return

        # Check directional consistency
        recent_dom  = buf["dominance"][-DRIFT_GATE_SESSIONS:]
        recent_warm = buf["warmth"][-DRIFT_GATE_SESSIONS:]
        dom_consistent  = all(d > base_dom for d in recent_dom) or \
                          all(d < base_dom for d in recent_dom)
        warm_consistent = all(w > base_warm for w in recent_warm) or \
                          all(w < base_warm for w in recent_warm)

        if not dom_consistent and not warm_consistent:
            logger.debug("Profile update BLOCKED — inconsistent direction")
            return

        # All gates passed — update
        ipc = self.profile.setdefault("ipc", {})
        if dom_consistent:
            ipc["dominance"] = round(
                sum(recent_dom) / len(recent_dom), 3
            )
        if warm_consistent:
            ipc["warmth"] = round(
                sum(recent_warm) / len(recent_warm), 3
            )

        # Re-derive archetype
        dom  = ipc.get("dominance", 0.5)
        warm = ipc.get("warmth",    0.6)
        if dom < 0.40 and warm > 0.60:
            ipc["archetype"] = "maya"
        elif dom > 0.60:
            ipc["archetype"] = "arjun"
        else:
            ipc["archetype"] = "lina"

        buf["dominance"] = []
        buf["warmth"]    = []
        logger.info(
            f"Profile updated: dom={ipc['dominance']} "
            f"warm={ipc['warmth']} arch={ipc['archetype']}"
        )

    def update_knowledge(
            self,
            mastery_updates:      dict,
            new_misconceptions:   list,
            comprehension_score:  int,
            topics_covered:       list,
            topics_struggling:    list,
            recommended_next:     Optional[str],
            session_anomaly:      bool = False,
    ):
        """Fast updates — no drift gate needed for knowledge."""
        if session_anomaly and ANOMALY_BLOCK_ALL:
            return

        knowledge = self.profile.setdefault("knowledge", {})

        mastery = knowledge.setdefault("mastery_map", {})
        for topic, data in mastery_updates.items():
            mastery[topic] = data

        existing_mc  = knowledge.setdefault("misconceptions", [])
        existing_str = [
            m.get("description", m) if isinstance(m, dict) else m
            for m in existing_mc
        ]
        for mc in new_misconceptions:
            desc = mc.get("description", mc) if isinstance(mc, dict) else mc
            if desc and desc not in existing_str:
                existing_mc.append({
                    "description": desc,
                    "resolved":    False,
                    "found_date":  datetime.now().isoformat(),
                })

        history = self.profile.setdefault("session_history", [])
        history.append({
            "date":                datetime.now().isoformat(),
            "comprehension_score": comprehension_score,
            "topics_covered":      topics_covered,
            "topics_struggling":   topics_struggling,
            "recommended_next_topic": recommended_next,
        })
        history[:] = history[-20:]   # keep last 20 sessions

    def _log_anomaly(self, ipc_summary: dict):
        log = self.profile.setdefault("anomaly_log", [])
        log.append({
            "date":    datetime.now().isoformat(),
            "reason":  "environmental_fatigue",
            "ipc_summary": ipc_summary,
            "profile_update_blocked": True,
        })

    @property
    def context_for_session(self) -> dict:
        """
        Returns context dict consumed by prompt assembler.
        Indicates if student is returning and what happened last time.
        """
        history = self.profile.get("session_history", [])
        if not history:
            return {"returning": False}

        last = history[-1]
        misconceptions = [
            m for m in self.profile.get("knowledge", {}).get("misconceptions", [])
            if isinstance(m, dict) and not m.get("resolved", False)
        ]
        return {
            "returning":                True,
            "total_sessions":           len(history),
            "last_session":             last,
            "unresolved_misconceptions": misconceptions[:3],
            "topics_due_review":        [],   # filled by spaced_repetition
        }