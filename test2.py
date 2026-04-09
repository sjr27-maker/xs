# tests/test_phase2_integration.py
"""
Simulates a full Niraj-like session without mic or Live API.
Feeds scripted student responses, checks that the cognitive
model evolves correctly across 5 turns.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from memory.profile_manager   import ProfileManager
from memory.session_memory    import SessionMemory
import memory.belief_graph as bg_module
from cognition.belief_model   import BeliefModel
from cognition.error_classifier import classify_error
from cognition.zpd_estimator  import ZPDEstimator
from cognition.affect_tracker import AffectTracker
from cognition.wm_monitor     import WMMonitor
from cognition.dependency_tracker import DependencyTracker
from pedagogy.situational_gate import SituationalGate
from pedagogy.question_planner import QuestionPlanner
from intake.session_checker   import SessionContext

# Scripted Niraj session
NIRAJ_TURNS = [
    {
        "student": "multiply first and last and do LCM on the product, "
                   "split the middle term, but why we split I'm not sure, "
                   "maybe 80% clear, I got -3 and -4 correct",
        "ai":      "Good — you found the split. Why do you think -3 and -4 "
                   "work specifically? What's special about their product?",
        "error":   "CONCEPTUAL",
    },
    {
        "student": "I think because they add to -7?",
        "ai":      "That's the sum condition — correct. "
                   "What's the product of -3 and -4?",
        "error":   "NONE",
    },
    {
        "student": "12. Oh wait, that's the same as the product of first and last",
        "ai":      "Exactly — that's why LCM of the product matters. "
                   "Now try: x² - 22x + 120",
        "error":   "NONE",
    },
    {
        "student": "uh... multiply 1 and 120... LCM... um... -10 and -12? "
                   "So x² -12x - 10x + 120...",
        "ai":      "Almost — check the signs on the middle terms.",
        "error":   "CARELESS",
    },
    {
        "student": "I don't know. I give up.",
        "ai":      "That's okay — let's stop here. You made real progress today.",
        "error":   "OVERLOAD_INDUCED",
    },
]

def run_integration_test():
    # Setup — anomaly session (tired student)
    session_ctx = SessionContext(
        fatigue_level="high",
        external_load=True,
        time_pressure=False,
        anomaly_flag=True,
        emotional_state="frustrated",
        session_note="Night session, tired, mosquitoes",
    )

    pm = ProfileManager("niraj_integration_test")
    pm.profile = {
        "student_id": "niraj_integration_test",
        "ipc": {"dominance": 0.52, "warmth": 0.63,
                "archetype": "lina", "pace": "medium"},
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

    print("\n── Integration test: Niraj factorization session\n")

    zpd_history    = []
    affect_history = []
    error_history  = []
    gate_blocks    = 0

    for i, turn in enumerate(NIRAJ_TURNS):
        print(f"Turn {i+1}: {turn['student'][:60]}...")

        # Error classification
        error = classify_error(
            turn["student"], turn["ai"],
            concept="factorization",
            fatigue_level=session_ctx.fatigue_level,
            consecutive_confused=i,
        )
        error_history.append(error.error_type)
        print(f"  Error: {error.error_type} (expected: {turn['error']})")

        # ZPD
        zpd_est = zpd.update(
            student_text=turn["student"],
            response_latency_ms=2000 + i * 500,
            error_type=error.error_type,
            filler_count=turn["student"].lower().count("um")
                         + turn["student"].lower().count("uh"),
            giving_up="give up" in turn["student"].lower(),
        )
        zpd_history.append(zpd_est.position)
        print(f"  ZPD: {zpd_est.position} (score={zpd_est.score:.2f})")

        # Affect
        affect_st = affect.update(
            ipc_giving_up="give up" in turn["student"].lower(),
            filler_count=2 if i > 2 else 0,
            energy_trend="declining" if i > 1 else "stable",
            error_type=error.error_type,
            student_text=turn["student"],
            consecutive_confused=i,
        )
        affect_history.append(affect_st.frustration_type)
        print(f"  Affect: {affect_st.frustration_type} / {affect_st.trajectory}")

        # WM
        can, close = wm.introduce_concept("factorization")
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

        # Gate check — would the Q that was asked be blocked?
        gate_result = gate.evaluate(
            "RETRIEVAL" if i < 3 else "CAUSAL",
            affect_st, zpd_est, wm_st, dep_st, turn_num=i+1,
        )
        if not gate_result.allowed:
            gate_blocks += 1
            print(f"  Gate: BLOCKED ({gate_result.reason})")
            print(f"        → Alternatives: {gate_result.allowed_alternatives}")
        else:
            print(f"  Gate: ALLOWED")

        # Session memory
        sm.add_turn(
            student_text=turn["student"],
            ai_response=turn["ai"],
            ipc_vector={"dominance": 0.52 - i*0.05, "warmth": 0.63},
            acoustic={},
            error_type=error.error_type,
            affect_state=affect_st.emotional_state,
            zpd_position=zpd_est.position,
            wm_slots=wm_st.slots_used,
            dependency_idx=dep_st.index,
        )
        print()

    # ── Assertions ────────────────────────────────────────────────────
    print("── Assertions\n")

    # ZPD should deteriorate
    assert "ABOVE" in zpd_history, \
        f"Should detect ABOVE ZPD by turn 5. Got: {zpd_history}"
    print(f"  ZPD trajectory: {zpd_history} ✓")

    # Affect should reach STUCK
    assert "STUCK" in affect_history, \
        f"Should detect STUCK frustration. Got: {affect_history}"
    print(f"  Affect trajectory: {affect_history} ✓")

    # Gate should have blocked at least once
    assert gate_blocks >= 1, \
        "Situational gate should have blocked at least one question"
    print(f"  Gate blocks: {gate_blocks} ✓")

    # WM should be full
    assert wm.get_state().overload_detected, \
        "WM overload should be detected after OVERLOAD_INDUCED error"
    print(f"  WM overload detected ✓")

    # Profile update blocked due to anomaly
    ipc_before = dict(pm.profile["ipc"])
    pm.update_base_profile(
        {"avg_dominance": 0.30, "avg_warmth": 0.40},
        turn_count=5,
        session_anomaly=True,
    )
    assert pm.profile["ipc"]["dominance"] == ipc_before["dominance"], \
        "Profile should NOT update on anomaly session"
    print(f"  Anomaly gate blocked profile update ✓")
    assert len(pm.profile.get("anomaly_log", [])) > 0, \
        "Anomaly should be logged"
    print(f"  Anomaly logged ✓")

    print("\n  Integration test PASSED")
    return True


if __name__ == "__main__":
    run_integration_test()