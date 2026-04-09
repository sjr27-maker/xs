# tests/test_phase1.py
"""Run each module independently before wiring together."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_transcriber():
    """Requires DEEPGRAM_API_KEY. Record a 3-second WAV first."""
    from intake.transcriber import transcribe
    import numpy as np
    import scipy.io.wavfile as wav
    # Fake 3s of silence — just tests HTTP call and parse
    audio = np.zeros(48000, dtype=np.int16)
    wav.write("./test_audio.wav", 16000, audio)
    text, fillers, dur = transcribe("./test_audio.wav")
    print(f"  transcriber: OK — text='{text}' fillers={fillers}")

def test_acoustic_analyzer():
    from intake.acoustic_analyzer import extract_acoustic_vector
    import numpy as np
    import scipy.io.wavfile as wav
    # Synthesize a simple sine wave as "speech"
    t = np.linspace(0, 2, 32000)
    audio = (np.sin(2 * np.pi * 200 * t) * 8000).astype(np.int16)
    wav.write("./test_sine.wav", 16000, audio)
    av = extract_acoustic_vector("./test_sine.wav", filler_count=2)
    print(f"  acoustic_analyzer: OK — dom={av.dominance:.2f} "
          f"warm={av.warmth:.2f} pace={av.pace} fatigue={av.fatigue_score:.2f}")
    assert 0 <= av.dominance <= 1
    assert 0 <= av.warmth <= 1
    assert av.pace in ("slow", "medium", "fast")

def test_session_checker_extraction():
    """Tests Gemini extraction only — no mic."""
    from intake.session_checker import _extract_context
    ctx = _extract_context([
        "I'm really tired, had a long day",
        "Yes, I had exams and projects",
        "I don't have much time, maybe 20 minutes",
    ])
    print(f"  session_checker: fatigue={ctx.fatigue_level} "
          f"anomaly={ctx.anomaly_flag} emotion={ctx.emotional_state}")
    assert ctx.fatigue_level in ("low", "medium", "high")
    assert isinstance(ctx.anomaly_flag, bool)
    # This input should trigger anomaly
    assert ctx.anomaly_flag == True, "Should detect anomaly from tired + exam + time pressure"

def test_error_classifier():
    from cognition.error_classifier import classify_error
    # CARELESS case
    r = classify_error(
        "I wrote -3x instead of +3x",
        "Small slip there — sign mistake. The rest is correct.",
        concept="factorization",
        fatigue_level="high",
    )
    print(f"  error_classifier (careless): type={r.error_type} detected={r.detected}")

    # CONCEPTUAL case
    r2 = classify_error(
        "I used LCM to find the numbers I need to split with",
        "The LCM is actually a verification step, not the source of the split numbers.",
        concept="factorization",
    )
    print(f"  error_classifier (conceptual): type={r2.error_type} "
          f"root='{r2.root_belief}'")

def test_zpd_estimator():
    from cognition.zpd_estimator import ZPDEstimator
    zpd = ZPDEstimator()
    # Simulate deteriorating performance
    for i in range(4):
        est = zpd.update(
            student_text="I don't know, I think maybe" + "um " * i,
            response_latency_ms=2000 + i * 800,
            error_type="PROCEDURAL" if i > 1 else "NONE",
            filler_count=i * 2,
            giving_up=i == 3,
        )
    print(f"  zpd_estimator: position={est.position} "
          f"score={est.score:.2f} confidence={est.confidence:.2f}")
    assert est.position in ("ABOVE", "IN", "BELOW")

def test_affect_tracker():
    from cognition.affect_tracker import AffectTracker
    tracker = AffectTracker()
    # Simulate progressive frustration
    for i in range(4):
        state = tracker.update(
            ipc_giving_up=(i == 3),
            filler_count=i * 2,
            energy_trend="declining" if i > 1 else "stable",
            error_type="OVERLOAD_INDUCED" if i > 2 else "NONE",
            student_text="I give up" if i == 3 else "let me try again",
            consecutive_confused=i,
        )
    print(f"  affect_tracker: state={state.emotional_state} "
          f"type={state.frustration_type} trajectory={state.trajectory}")

def test_wm_monitor():
    from cognition.wm_monitor import WMMonitor
    wm = WMMonitor()
    concepts = ["factorization", "LCM", "middle term", "taking common"]
    for c in concepts:
        can, to_close = wm.introduce_concept(c)
        if can:
            wm.mark_introduced(c)
    state = wm.get_state()
    print(f"  wm_monitor: slots={state.slots_used} full={state.is_full} "
          f"closure_needed={state.closure_needed}")
    assert state.is_full

def test_situational_gate():
    from pedagogy.situational_gate import SituationalGate
    from cognition.affect_tracker import AffectState
    from cognition.zpd_estimator import ZPDEstimate
    from cognition.wm_monitor import WMState
    from cognition.dependency_tracker import DependencyState

    gate = SituationalGate()
    # STUCK state should block RETRIEVAL
    affect = AffectState(
        emotional_state="frustrated", trajectory="deteriorating",
        frustration_type="STUCK", intervention_needed=True, intensity=0.7,
    )
    zpd = ZPDEstimate(score=-0.3, position="ABOVE", confidence=0.8, signals={})
    wm  = WMState(slots_used=2, active_concepts=["factorization"],
                  is_full=False, overload_detected=False,
                  closure_needed=False, closure_target=None)
    dep = DependencyState(index=0.4, level="MEDIUM", alarm_triggered=False,
                          turns_at_alarm=0, trend="stable")

    result = gate.evaluate("RETRIEVAL", affect, zpd, wm, dep, turn_num=5)
    print(f"  situational_gate: allowed={result.allowed} "
          f"blocked={result.blocked_type} reason='{result.reason}'")
    assert result.allowed == False, "RETRIEVAL should be blocked in STUCK state"
    assert "WIN_CREATING" in result.allowed_alternatives

def test_belief_model():
    from cognition.belief_model import BeliefModel
    graph = {"concepts": {}}
    bm    = BeliefModel(graph)
    bm.seed_from_onboarding(
        "Explain what factorization means and why we do it",
        "Factorization means breaking numbers into smaller parts. "
        "We multiply first and last term then do LCM on the product "
        "to find the split numbers.",
        "Mathematics",
    )
    snapshot = bm.get_snapshot()
    print(f"  belief_model seed: {snapshot}")
    assert "concepts" in graph

def test_profile_manager():
    from memory.profile_manager import ProfileManager
    pm = ProfileManager("test_student_sanity")
    pm.profile = {
        "ipc": {"dominance": 0.5, "warmth": 0.6, "archetype": "lina"},
        "session_history": [],
        "_evidence_buffer": {"dominance": [], "warmth": []},
    }
    adapted = pm.get_session_adapted_ipc({"dominance": 0.7, "warmth": 0.4,
                                           "pace": "fast", "giving_up": False})
    print(f"  profile_manager: adapted dom={adapted['dominance']:.2f} "
          f"warm={adapted['warmth']:.2f} α={pm.get_alpha():.2f}")
    assert 0 <= adapted["dominance"] <= 1

def test_speaker():
    """Tests Gemini TTS — will actually speak."""
    from output.speaker import speak
    print("  speaker: Speaking test phrase...")
    speak("Hello. SYRA voice test. One two three.", archetype="lina", wait=True)
    print("  speaker: OK")

def test_prompt_assembler():
    """Smoke test — just checks it produces a non-empty string."""
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
        student_id="test", subject="Mathematics", grade=10,
        archetype="lina", turn_num=3,
        session_ctx=SessionContext(fatigue_level="low"),
        acoustic=type("A", (), {
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
        question_plan=QuestionPlan("CAUSAL", "test", None,
                                   "Ask why splitting works."),
        wm_monitor=wm_mon,
        conversation_history=[{"role": "student", "content": "what is factorization"}],
        unresolved_misconceptions=[],
        topics_due_review=[],
        spaced_review_prompt=None,
        rag_context="",
        style_mirror="",
    )
    prompt = build_system_prompt(inp)
    print(f"  assembler: OK — {len(prompt)} chars, "
          f"{len(prompt.split())} words")
    assert "SYRA" in prompt
    assert "CONSTRAINTS" in prompt or len(prompt) > 100

if __name__ == "__main__":
    tests = [
        ("Transcriber",        test_transcriber),
        ("Acoustic analyzer",  test_acoustic_analyzer),
        ("Session checker",    test_session_checker_extraction),
        ("Error classifier",   test_error_classifier),
        ("ZPD estimator",      test_zpd_estimator),
        ("Affect tracker",     test_affect_tracker),
        ("WM monitor",         test_wm_monitor),
        ("Situational gate",   test_situational_gate),
        ("Belief model",       test_belief_model),
        ("Profile manager",    test_profile_manager),
        ("Speaker (TTS)",      test_speaker),
        ("Prompt assembler",   test_prompt_assembler),
    ]

    passed, failed = 0, 0
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

    print(f"\n{'='*40}")
    print(f"  Passed: {passed}/{len(tests)}")
    if failed:
        print(f"  Failed: {failed} — fix before proceeding")
    else:
        print("  All clear — proceed to Phase 2")