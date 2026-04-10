# tests/test_phase1.py
"""
Unit tests — run each module independently.
Windows-compatible. All paths use os.path.

Run: python tests/test_phase1.py
All must pass before Phase 2.
"""
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def test_transcriber():
    """Deepgram HTTP call with silent audio."""
    from intake.transcriber import transcribe
    import numpy as np
    import scipy.io.wavfile as wav_io

    audio = np.zeros(48000, dtype=np.int16)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_io.write(tmp.name, 16000, audio)
    try:
        text, fillers, dur = transcribe(tmp.name)
        print(f"  transcriber: OK — text='{text}' fillers={fillers}")
    finally:
        os.unlink(tmp.name)


def test_acoustic_analyzer():
    """Sine wave as synthetic speech."""
    from intake.acoustic_analyzer import extract_acoustic_vector
    import numpy as np
    import scipy.io.wavfile as wav_io

    t = np.linspace(0, 2, 32000)
    audio = (np.sin(2 * np.pi * 200 * t) * 8000).astype(np.int16)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_io.write(tmp.name, 16000, audio)
    try:
        av = extract_acoustic_vector(tmp.name, filler_count=2)
        print(
            f"  acoustic_analyzer: OK — "
            f"dom={av.dominance:.2f} warm={av.warmth:.2f} "
            f"pace={av.pace} fatigue={av.fatigue_score:.2f}"
        )
        assert 0 <= av.dominance <= 1, "dominance out of range"
        assert 0 <= av.warmth <= 1, "warmth out of range"
        assert av.pace in ("slow", "medium", "fast"), f"bad pace: {av.pace}"
    finally:
        os.unlink(tmp.name)


def test_session_checker_extraction():
    """Gemini extraction — no mic needed."""
    from intake.session_checker import _extract_context
    ctx = _extract_context([
        "I am really tired, had a very long day with exams",
        "Yes, I had three exams and a project due",
        "I only have about 20 minutes, I need to sleep soon",
    ])
    print(
        f"  session_checker: fatigue={ctx.fatigue_level} "
        f"anomaly={ctx.anomaly_flag} emotion={ctx.emotional_state}"
    )
    assert ctx.fatigue_level in ("low", "medium", "high")
    assert isinstance(ctx.anomaly_flag, bool)
    assert ctx.anomaly_flag is True, \
        "Should detect anomaly: tired + exams + time pressure"


def test_error_classifier():
    """Fixed quick screen — should now detect errors correctly."""
    from cognition.error_classifier import classify_error

    # CONCEPTUAL — should NOT be blocked by quick screen
    r = classify_error(
        "multiply first and last and do LCM on the product to find split numbers",
        (
            "The LCM is actually for verification, not for finding the split "
            "numbers — that is exactly where the misconception is."
        ),
        concept="factorization",
    )
    print(
        f"  error_classifier (conceptual): "
        f"type={r.error_type} detected={r.detected}"
    )

    # CARELESS — high fatigue context
    r2 = classify_error(
        "I wrote minus 3 instead of plus 3",
        "Small sign slip — the rest is correct.",
        concept="factorization",
        fatigue_level="high",
    )
    print(
        f"  error_classifier (careless): "
        f"type={r2.error_type} detected={r2.detected}"
    )

    # NONE — full confirmation phrase should skip
    r3 = classify_error(
        "the split numbers are minus 3 and minus 4",
        "That is exactly right — minus 3 times minus 4 is 12 and they add to minus 7.",
        concept="factorization",
    )
    print(
        f"  error_classifier (none): "
        f"type={r3.error_type} detected={r3.detected}"
    )
    assert r3.error_type == "NONE", \
        f"Full confirmation should return NONE, got {r3.error_type}"


def test_zpd_estimator():
    """Deteriorating performance should reach ABOVE."""
    from cognition.zpd_estimator import ZPDEstimator

    zpd    = ZPDEstimator()
    scores = []

    turns = [
        ("let me try, I think it might be minus 3", 1800, "NONE", 0, False),
        ("I think because they add to minus 7 maybe?", 2400, "NONE", 1, False),
        ("um... I am not sure... hmm... maybe 12?", 3200, "PROCEDURAL", 3, False),
        ("I don't know... I give up", 5500, "OVERLOAD_INDUCED", 0, True),
    ]
    for text, lat, err, fill, gu in turns:
        est = zpd.update(
            student_text=text,
            response_latency_ms=lat,
            error_type=err,
            filler_count=fill,
            giving_up=gu,
        )
        scores.append(est.score)

    print(
        f"  zpd_estimator: position={est.position} "
        f"score={est.score:.3f} scores={[round(s, 3) for s in scores]}"
    )
    assert est.position == "ABOVE", \
        f"Should be ABOVE after deteriorating turns, got {est.position}"
    assert scores[-1] < scores[0], \
        "Score should deteriorate (become more negative)"


def test_affect_tracker():
    """Progressive frustration should reach STUCK."""
    from cognition.affect_tracker import AffectTracker

    tracker = AffectTracker()
    for i in range(4):
        state = tracker.update(
            ipc_giving_up=(i == 3),
            filler_count=i * 2,
            energy_trend="declining" if i > 1 else "stable",
            error_type="OVERLOAD_INDUCED" if i > 2 else "NONE",
            student_text="I give up" if i == 3 else "let me try again",
            consecutive_confused=i,
        )
    print(
        f"  affect_tracker: state={state.emotional_state} "
        f"type={state.frustration_type} traj={state.trajectory}"
    )
    assert state.frustration_type == "STUCK", \
        f"Should reach STUCK, got {state.frustration_type}"


def test_wm_monitor():
    """Fill WM to capacity."""
    from cognition.wm_monitor import WMMonitor

    wm = WMMonitor()
    for c in ["factorization", "LCM", "middle term", "taking common"]:
        can, _ = wm.introduce_concept(c)
        if can:
            wm.mark_introduced(c)

    wm.report_overload_error()
    wm.report_overload_error()
    state = wm.get_state()
    print(
        f"  wm_monitor: slots={state.slots_used} "
        f"full={state.is_full} overload={state.overload_detected}"
    )
    assert state.is_full, "Should be full at 4 concepts"
    assert state.overload_detected, "Should detect overload after 2 errors"


def test_situational_gate():
    """STUCK frustration should block RETRIEVAL questions."""
    from cognition.situational_gate import SituationalGate
    from cognition.affect_tracker import AffectState
    from cognition.zpd_estimator import ZPDEstimate
    from cognition.wm_monitor import WMState
    from cognition.dependency_tracker import DependencyState

    gate = SituationalGate()
    affect = AffectState(
        emotional_state="frustrated",
        trajectory="deteriorating",
        frustration_type="STUCK",
        intervention_needed=True,
        intensity=0.7,
    )
    zpd = ZPDEstimate(
        score=-0.35, position="ABOVE",
        confidence=0.80, signals={}
    )
    wm = WMState(
        slots_used=2,
        active_concepts=["factorization"],
        is_full=False,
        overload_detected=False,
        closure_needed=False,
        closure_target=None,
    )
    dep = DependencyState(
        index=0.4, level="MEDIUM",
        alarm_triggered=False,
        turns_at_alarm=0,
        trend="stable",
    )
    result = gate.evaluate("RETRIEVAL", affect, zpd, wm, dep, turn_num=5)
    print(
        f"  situational_gate: allowed={result.allowed} "
        f"blocked={result.blocked_type} "
        f"alts={result.allowed_alternatives}"
    )
    assert result.allowed is False, \
        "RETRIEVAL should be blocked in STUCK state"
    assert "WIN_CREATING" in result.allowed_alternatives


def test_belief_model():
    """Seed from onboarding and check graph populated."""
    from cognition.belief_model import BeliefModel

    graph = {"concepts": {}}
    bm = BeliefModel(graph)
    bm.seed_from_onboarding(
        "Explain what factorization means and why we do it",
        (
            "Factorization means breaking a polynomial into simpler parts. "
            "We multiply first and last term then do LCM on the product "
            "to find the split numbers."
        ),
        "Mathematics",
    )
    snapshot = bm.get_snapshot()
    print(f"  belief_model: {snapshot[:80]}")
    assert "concepts" in graph
    assert len(graph["concepts"]) > 0, "Should have seeded at least one concept"


def test_profile_manager():
    """α-blend and anomaly gate."""
    from memory.profile_manager import ProfileManager

    pm = ProfileManager("test_sanity_student")
    pm.profile = {
        "ipc": {"dominance": 0.5, "warmth": 0.6, "archetype": "lina"},
        "session_history": [],
        "_evidence_buffer": {"dominance": [], "warmth": []},
        "anomaly_log": [],
    }

    adapted = pm.get_session_adapted_ipc({
        "dominance": 0.7, "warmth": 0.4,
        "pace": "fast", "giving_up": False,
        "filler_count": 0, "assertiveness_delta": 0.0,
    })
    print(
        f"  profile_manager: "
        f"adapted dom={adapted['dominance']:.2f} "
        f"warm={adapted['warmth']:.2f} alpha={pm.get_alpha():.2f}"
    )
    assert 0 <= adapted["dominance"] <= 1

    # Anomaly gate
    dom_before = pm.profile["ipc"]["dominance"]
    pm.update_base_profile(
        {"avg_dominance": 0.10, "avg_warmth": 0.10},
        turn_count=8,
        session_anomaly=True,
    )
    assert pm.profile["ipc"]["dominance"] == dom_before, \
        "Profile should not update on anomaly session"
    print(f"  profile_manager anomaly gate: OK")


def test_speaker():
    """Gemini TTS — will actually produce audio."""
    from output.speaker import speak
    print("  speaker: generating test audio...")
    speak("SYRA unit test. Audio confirmed.", archetype="lina", wait=True)
    print("  speaker: OK")


def test_prompt_assembler():
    """Smoke test — produces non-empty instruction string."""
    from prompt.assembler import build_system_prompt, AssemblerInput
    from cognition.zpd_estimator import ZPDEstimate
    from cognition.affect_tracker import AffectState
    from cognition.wm_monitor import WMState, WMMonitor
    from cognition.dependency_tracker import DependencyState
    from cognition.error_classifier import ErrorAnalysis
    from pedagogy.situational_gate import GateResult
    from pedagogy.question_planner import QuestionPlan
    from intake.session_checker import SessionContext

    wm_mon = WMMonitor()
    inp = AssemblerInput(
        student_id="test",
        subject="Mathematics",
        grade=10,
        archetype="lina",
        turn_num=3,
        session_ctx=SessionContext(fatigue_level="low"),
        acoustic=type("AV", (), {
            "dominance": 0.5, "warmth": 0.6, "pace": "medium",
            "giving_up": False, "filler_count": 1,
            "fatigue_score": 0.1, "energy_trend": "stable",
        })(),
        belief_snapshot="factorization: procedure=70% concept=15%",
        zpd=ZPDEstimate(score=-0.1, position="IN", confidence=0.7, signals={}),
        affect=AffectState("neutral", "stable", "NONE", False, 0.1),
        wm=WMState(2, ["factorization"], False, False, False, None),
        dependency=DependencyState(0.3, "LOW", False, 0, "stable"),
        error_analysis=ErrorAnalysis("NONE", False),
        consecutive_confused=0,
        gate_result=GateResult(True, None, "", ["CAUSAL"], ""),
        question_plan=QuestionPlan(
            "CAUSAL", "proc=70% conc=15%", None,
            "Ask why splitting works."
        ),
        wm_monitor=wm_mon,
        conversation_history=[
            {"role": "student", "content": "what is factorization"}
        ],
        unresolved_misconceptions=[],
        topics_due_review=[],
        spaced_review_prompt=None,
        rag_context="",
        style_mirror="",
    )
    prompt = build_system_prompt(inp)
    print(
        f"  assembler: OK — {len(prompt)} chars, "
        f"{len(prompt.split())} words"
    )
    assert "SYRA" in prompt
    assert len(prompt) > 200, "Prompt too short — check assembler"


if __name__ == "__main__":
    tests = [
        ("Transcriber",           test_transcriber),
        ("Acoustic analyzer",     test_acoustic_analyzer),
        ("Session checker",       test_session_checker_extraction),
        ("Error classifier",      test_error_classifier),
        ("ZPD estimator",         test_zpd_estimator),
        ("Affect tracker",        test_affect_tracker),
        ("WM monitor",            test_wm_monitor),
        ("Situational gate",      test_situational_gate),
        ("Belief model",          test_belief_model),
        ("Profile manager",       test_profile_manager),
        ("Speaker (TTS)",         test_speaker),
        ("Prompt assembler",      test_prompt_assembler),
    ]

    passed = 0
    failed = 0

    for name, fn in tests:
        print(f"\n── {name}")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*42}")
    print(f"  Passed: {passed} / {len(tests)}")
    if failed:
        print(f"  Failed: {failed} — fix before Phase 2")
        sys.exit(1)
    else:
        print("  All clear — run Phase 2 next")
        sys.exit(0)