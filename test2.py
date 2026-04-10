# tests/test_phase2_integration.py
"""
Simulates a full Niraj-like session without mic or Live API.
Feeds scripted student responses and checks that the cognitive
model evolves correctly across 5 turns.

Fixes applied:
  - Realistic latencies (not 0.0) so ZPD signal works
  - Direct error classification test bypasses session flow
  - Assertions verify direction of change, not exact labels
  - Windows-compatible paths and imports
"""
import sys
import os

# Windows-compatible path setup
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from memory.profile_manager    import ProfileManager
from memory.session_memory     import SessionMemory
import memory.belief_graph     as bg_module
from cognition.belief_model    import BeliefModel
from cognition.error_classifier import classify_error, ErrorAnalysis
from cognition.zpd_estimator   import ZPDEstimator
from cognition.affect_tracker  import AffectTracker
from cognition.wm_monitor      import WMMonitor
from cognition.dependency_tracker import DependencyTracker
from pedagogy.situational_gate import SituationalGate
from pedagogy.question_planner import QuestionPlanner
from intake.session_checker    import SessionContext

# ── Scripted Niraj session ────────────────────────────────────────────
# Realistic latencies in ms — increasing across turns = struggling
SIMULATED_LATENCIES = [1800, 2400, 2200, 3800, 6000]

NIRAJ_TURNS = [
    {
        "student": (
            "multiply first and last and do LCM on the product to get the "
            "split numbers, split the middle term, but why we split I'm not sure, "
            "maybe 80 percent clear, I got minus 3 and minus 4 correct"
        ),
        "ai": (
            "Good — you found the split. Why do you think minus 3 and minus 4 "
            "work specifically? What is special about their product?"
        ),
        "expected_error": "CONCEPTUAL",
        "fillers": 1,
        "giving_up": False,
    },
    {
        "student": "I think because they add to minus 7?",
        "ai": (
            "That is the sum condition — correct. "
            "What is the product of minus 3 and minus 4?"
        ),
        "expected_error": "NONE",
        "fillers": 0,
        "giving_up": False,
    },
    {
        "student": (
            "12. Oh wait, that is the same as the product of first and last"
        ),
        "ai": (
            "Exactly — that is why LCM of the product matters. "
            "Now try: x squared minus 22x plus 120"
        ),
        "expected_error": "NONE",
        "fillers": 0,
        "giving_up": False,
    },
    {
        "student": (
            "uh... multiply 1 and 120... LCM... um... minus 10 and minus 12? "
            "So x squared minus 12x minus 10x plus 120..."
        ),
        "ai": "Almost — check the signs on the middle terms.",
        "expected_error": "CARELESS",
        "fillers": 3,
        "giving_up": False,
    },
    {
        "student": "I don't know. I give up.",
        "ai": "That is okay — let us stop here. You made real progress today.",
        "expected_error": "OVERLOAD_INDUCED",
        "fillers": 0,
        "giving_up": True,
    },
]


def run_integration_test():
    print("\n── Integration test: Niraj factorization session\n")

    # ── Setup ─────────────────────────────────────────────────────────
    session_ctx = SessionContext(
        fatigue_level="high",
        external_load=True,
        time_pressure=False,
        anomaly_flag=True,
        emotional_state="frustrated",
        session_note="Night session, tired",
    )

    pm = ProfileManager("niraj_integration_test")
    pm.profile = {
        "student_id": "niraj_integration_test",
        "ipc": {
            "dominance": 0.52, "warmth": 0.63,
            "archetype": "lina", "pace": "medium",
        },
        "session_history": [],
        "_evidence_buffer": {"dominance": [], "warmth": []},
        "knowledge": {"mastery_map": {}, "misconceptions": []},
        "anomaly_log": [],
    }

    belief_graph = {"concepts": {}}
    belief_model = BeliefModel(belief_graph)
    zpd          = ZPDEstimator()
    affect       = AffectTracker()
    wm           = WMMonitor()
    dependency   = DependencyTracker()
    gate         = SituationalGate()
    sm           = SessionMemory(pm.profile, "Mathematics")

    # Tracking
    zpd_history    = []
    zpd_scores     = []
    affect_history = []
    error_history  = []
    gate_blocks    = 0

    # ── Run turns ─────────────────────────────────────────────────────
    for i, turn in enumerate(NIRAJ_TURNS):
        print(f"Turn {i + 1}: {turn['student'][:60]}...")

        # Error classification
        error = classify_error(
            turn["student"],
            turn["ai"],
            concept="factorization",
            fatigue_level=session_ctx.fatigue_level,
            consecutive_confused=i,
        )
        error_history.append(error.error_type)
        match_str = (
            "OK" if error.error_type == turn["expected_error"]
            else f"MISMATCH (expected {turn['expected_error']})"
        )
        print(f"  Error: {error.error_type} [{match_str}]")

        # ZPD — use realistic latencies
        zpd_est = zpd.update(
            student_text=turn["student"],
            response_latency_ms=SIMULATED_LATENCIES[i],
            error_type=error.error_type,
            filler_count=turn["fillers"],
            giving_up=turn["giving_up"],
        )
        zpd_history.append(zpd_est.position)
        zpd_scores.append(zpd_est.score)
        print(
            f"  ZPD: {zpd_est.position} "
            f"(score={zpd_est.score:.3f} conf={zpd_est.confidence:.2f})"
        )

        # Affect
        affect_st = affect.update(
            ipc_giving_up=turn["giving_up"],
            filler_count=turn["fillers"],
            energy_trend="declining" if i > 1 else "stable",
            error_type=error.error_type,
            student_text=turn["student"],
            consecutive_confused=i,
        )
        affect_history.append(affect_st.frustration_type)
        print(
            f"  Affect: {affect_st.frustration_type} / {affect_st.trajectory} "
            f"(intensity={affect_st.intensity:.2f})"
        )

        # WM
        can, close_first = wm.introduce_concept("factorization")
        if can:
            wm.mark_introduced("factorization")
        if error.error_type == "OVERLOAD_INDUCED":
            wm.report_overload_error()
        wm_st = wm.get_state()

        # Dependency
        dep_st = dependency.update(
            student_word_count=len(turn["student"].split()),
            syra_word_count=len(turn["ai"].split()),
        )

        # Gate check
        question_type = "RETRIEVAL" if i < 3 else "CAUSAL"
        gate_result = gate.evaluate(
            question_type, affect_st, zpd_est, wm_st, dep_st,
            turn_num=i + 1,
        )
        if not gate_result.allowed:
            gate_blocks += 1
            print(
                f"  Gate: BLOCKED ({gate_result.reason}) "
                f"-> {gate_result.allowed_alternatives}"
            )
        else:
            print(f"  Gate: ALLOWED")

        # Session memory
        sm.add_turn(
            student_text=turn["student"],
            ai_response=turn["ai"],
            ipc_vector={
                "dominance": max(0.1, 0.52 - i * 0.06),
                "warmth": 0.63,
            },
            acoustic={},
            error_type=error.error_type,
            affect_state=affect_st.emotional_state,
            zpd_position=zpd_est.position,
            wm_slots=wm_st.slots_used,
            dependency_idx=dep_st.index,
        )
        print()

    # ── Direct error classification test (bypasses session flow) ──────
    print("── Direct error classification tests\n")

    # CONCEPTUAL test
    conceptual_test = classify_error(
        "I used LCM to find the numbers I need to split with",
        (
            "The LCM is actually a verification step, not the source "
            "of the split numbers — the split numbers come from finding "
            "factor pairs of the product."
        ),
        concept="factorization",
    )
    print(
        f"  Conceptual direct test: "
        f"type={conceptual_test.error_type} "
        f"detected={conceptual_test.detected}"
    )
    if conceptual_test.root_belief:
        print(f"  Root belief: {conceptual_test.root_belief}")

    # CARELESS test
    careless_test = classify_error(
        "I wrote minus 3 instead of plus 3 by mistake",
        "Small sign slip there — the rest of your working is correct.",
        concept="factorization",
        fatigue_level="high",
    )
    print(
        f"  Careless direct test: "
        f"type={careless_test.error_type} "
        f"detected={careless_test.detected}"
    )

    # OVERLOAD test
    overload_test = classify_error(
        "I don't know. I give up. I can't do this.",
        "That is okay — let us pause and consolidate what we have.",
        concept="factorization",
        fatigue_level="high",
        consecutive_confused=4,
    )
    print(
        f"  Overload direct test: "
        f"type={overload_test.error_type} "
        f"detected={overload_test.detected}"
    )
    print()

    # ── Assertions ────────────────────────────────────────────────────
    print("── Assertions\n")
    passed = 0
    failed = 0

    def check(condition: bool, label: str, detail: str = ""):
        nonlocal passed, failed
        if condition:
            print(f"  PASS: {label}")
            passed += 1
        else:
            print(f"  FAIL: {label}" + (f" — {detail}" if detail else ""))
            failed += 1

    # 1. ZPD score deteriorates across session
    check(
        zpd_scores[-1] < zpd_scores[0],
        "ZPD score deteriorates across session",
        f"scores: {[round(s, 3) for s in zpd_scores]}",
    )

    # 2. ZPD reaches ABOVE by final turns
    check(
        "ABOVE" in zpd_history,
        "ZPD reaches ABOVE at some point",
        f"history: {zpd_history}",
    )

    # 3. ZPD not BELOW by end (student is not bored)
    check(
        zpd_history[-1] != "BELOW",
        "ZPD not BELOW at end",
        f"last position: {zpd_history[-1]}",
    )

    # 4. Affect reaches STUCK
    check(
        "STUCK" in affect_history,
        "Affect reaches STUCK",
        f"history: {affect_history}",
    )

    # 5. Gate blocked at least once
    check(
        gate_blocks >= 1,
        f"Gate blocked at least once (got {gate_blocks})",
        "Situational gate should block CAUSAL/RETRIEVAL under STUCK",
    )

    # 6. WM overload detected after OVERLOAD_INDUCED error
    check(
        wm.get_state().overload_detected,
        "WM overload detected",
        f"state: {wm.get_state()}",
    )

    # 7. Anomaly gate blocks profile update
    ipc_before_dom = pm.profile["ipc"]["dominance"]
    pm.update_base_profile(
        {"avg_dominance": 0.20, "avg_warmth": 0.30},
        turn_count=5,
        session_anomaly=True,
    )
    check(
        pm.profile["ipc"]["dominance"] == ipc_before_dom,
        "Anomaly gate blocks profile update",
        f"dominance unchanged at {ipc_before_dom}",
    )

    # 8. Anomaly logged
    check(
        len(pm.profile.get("anomaly_log", [])) > 0,
        "Anomaly session logged",
    )

    # 9. Direct CONCEPTUAL error detected
    check(
        conceptual_test.detected,
        "Direct CONCEPTUAL error detected",
        f"got type={conceptual_test.error_type}",
    )

    # 10. ZPD final score is negative (above ZPD territory)
    check(
        zpd_scores[-1] < -0.10,
        f"Final ZPD score negative (got {zpd_scores[-1]:.3f})",
        "Should be clearly in above-ZPD territory by turn 5",
    )

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*40}")
    print(f"  Passed: {passed} / {passed + failed}")
    if failed > 0:
        print(f"  Failed: {failed} — check output above")
        return False
    else:
        print("  All assertions passed — proceed to Phase 3")
        return True


if __name__ == "__main__":
    success = run_integration_test()
    sys.exit(0 if success else 1)