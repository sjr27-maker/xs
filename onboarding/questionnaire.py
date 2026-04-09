# 12-question spoken profiler. Unchanged logic.
# onboarding/questionnaire.py
"""
12-question spoken profiler.
Serves two purposes:
  1. IPC baseline (dominance, warmth, pace, archetype)
  2. Belief graph cold-start (procedural/conceptual confidence per concept)

Onboarding questions are upgraded — each is designed to elicit signals
across multiple cognitive dimensions simultaneously.
Session checker (per-session state) is separate — not done here.

ElevenLabs removed — Gemini TTS via output/speaker.py.
"""
import os
import json
import random
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv

from intake.recorder        import record_until_enter, save_wav
from intake.transcriber     import transcribe
from intake.acoustic_analyzer import extract_acoustic_vector, reset_session_baseline
from output.speaker         import speak
from cognition.belief_model import BeliefModel
from config import (
    EXTRACT_MODEL, ONBOARDING_IPC_RICH, ONBOARDING_BELIEF_Qs
)

load_dotenv()
logger  = logging.getLogger("SYRA.Onboarding")
_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Upgraded questions — each extracts 2-4 signals simultaneously
QUESTIONS = [
    ("q1",  "Which subject do you find the most difficult, and what specifically makes it hard — is it understanding the concepts, the steps, or applying them?"),
    ("q2",  "When you get something wrong in maths, what do you usually do — try again yourself, ask someone, look it up, or just move on?"),
    ("q3",  "If you had to learn something new right now, would you want me to explain the big picture first and then the details, or build it up step by step from the basics?"),
    ("q4",  "When someone uses a real-life example or a story to explain maths, does it help you understand better — or do you prefer a direct, clear explanation?"),
    ("q5",  "When you're figuring something out and I ask you questions instead of just explaining, does that feel helpful or does it feel like pressure?"),
    ("q6",  "What's your main goal right now — is it board exams, JEE or NEET prep, or just understanding things better for yourself?"),
    ("q7",  "If I explain something and you don't quite get it, what do you do — ask me to explain again, try to figure it out yourself, or just move on hoping it'll make sense later?"),
    ("q8",  "Tell me something in maths or science that you genuinely feel confident about. Explain it to me like I have no idea what it is — just speak naturally."),
    ("q9",  "How much encouragement do you like while learning — a lot, just a little, or do you prefer I keep things focused and just get on with it?"),
    ("q10", "Do you prefer going deep into one topic until you really get it, or covering several topics across a session?"),
    ("q11", "Is there anything in any subject where you think your understanding might be a bit off, or where you feel like you're just following steps without really knowing why?"),
    ("q12", "Last one — if you could tell every teacher or tutor one thing about how you learn best, what would it be?"),
]

IPC_RICH    = ONBOARDING_IPC_RICH
BELIEF_Qs   = ONBOARDING_BELIEF_Qs

EXTRACTION_PROMPT = """Extract learning profile signals from this student response.

Question: {question}
Answer: {answer}

Return ONLY valid JSON:
{{
  "subject_difficulty": null,
  "failure_response": null,
  "processing_style": null,
  "analogy_receptiveness": null,
  "socratic_tolerance": null,
  "goal_type": null,
  "help_seeking_style": null,
  "explanation_quality": null,
  "encouragement_need": null,
  "depth_vs_breadth": null,
  "misconception_seed": null,
  "emotional_signal": null,
  "persistence_signal": null,
  "metacognition_signal": null,
  "abstraction_comfort": null
}}

Rules:
- failure_response: "retry"|"ask_someone"|"move_on"|"give_up"|null
- processing_style: "top_down"|"bottom_up"|"flexible"|null
- analogy_receptiveness: "high"|"medium"|"low"|null
- socratic_tolerance: "high"|"medium"|"low"|null
- goal_type: "boards"|"jee_neet"|"curiosity"|null
- help_seeking_style: "proactive"|"reactive"|"independent"|null
- explanation_quality: "deep"|"structured"|"surface"|null
- encouragement_need: "high"|"medium"|"low"|null
- depth_vs_breadth: "depth"|"breadth"|"balanced"|null
- emotional_signal: "anxious"|"confident"|"defeated"|"curious"|"neutral"|null
- persistence_signal: "high"|"medium"|"low"|null
- metacognition_signal: "high"|"medium"|"low"|null
- abstraction_comfort: "high"|"medium"|"low"|null"""


class _NavigationCommand(Exception):
    pass


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _most_common(values: list):
    cleaned = [v for v in values if v and v != "null"]
    return max(set(cleaned), key=cleaned.count) if cleaned else None


def _save_checkpoint(path: Path, signals: list, ipc: list):
    path.write_text(json.dumps({
        "onboarding_done":  False,
        "_partial_signals": signals,
        "_partial_ipc":     ipc,
        "_checkpoint_time": datetime.now().isoformat(),
    }, indent=2, ensure_ascii=False), encoding="utf-8")


def _extract_signals(question: str, answer: str) -> dict:
    try:
        resp = _client.models.generate_content(
            model=EXTRACT_MODEL,
            contents=EXTRACTION_PROMPT.format(question=question, answer=answer),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                max_output_tokens=400,
            ),
        )
        return json.loads(resp.text)
    except Exception as e:
        logger.debug(f"Signal extraction failed: {e}")
        return {}


def _detect_navigation(text: str) -> Optional[str]:
    t = text.lower()
    if any(p in t for p in ["repeat", "say that again", "again", "once more"]):
        return "repeat"
    if any(p in t for p in ["go back", "previous", "before that", "last question"]):
        return "previous"
    if any(p in t for p in ["skip", "next", "move on", "pass"]):
        return "skip"
    return None


def _ask_question(
        q_id:        str,
        q_text:      str,
        archetype:   str,
        prev_q_text: Optional[str],
        belief_model: BeliefModel,
        subject:     str,
) -> tuple[dict, dict]:
    """
    Ask one question with navigation support.
    Returns (signals, ipc_dict).
    Raises _NavigationCommand on navigation intent.
    """
    max_retries = 4
    neutral_ipc = {"dominance": 0.5, "warmth": 0.7,
                   "pace": "medium", "giving_up": False}

    for _ in range(max_retries):
        input("  [ Press Enter to answer ]")
        audio = record_until_enter()
        path  = save_wav(audio)

        answer, fillers, _ = transcribe(path)

        if not answer.strip():
            os.unlink(path)
            speak("I didn't catch that — let me ask again.", archetype)
            speak(q_text, archetype)
            continue

        print(f"  Student: {answer}")

        nav = _detect_navigation(answer)
        if nav == "repeat":
            os.unlink(path)
            speak(f"Of course. {q_text}", archetype)
            continue
        if nav == "previous":
            os.unlink(path)
            if prev_q_text:
                speak(f"Sure, going back. {prev_q_text}", archetype)
            else:
                speak("This is the first question.", archetype)
            raise _NavigationCommand("previous")
        if nav == "skip":
            os.unlink(path)
            speak("No problem, moving on.", archetype)
            raise _NavigationCommand("skip")

        # IPC on rich questions
        ipc_dict = dict(neutral_ipc)
        if q_id in IPC_RICH:
            av = extract_acoustic_vector(path, filler_count=fillers)
            ipc_dict = {
                "dominance":  av.dominance,
                "warmth":     av.warmth,
                "pace":       av.pace,
                "giving_up":  av.giving_up,
            }
            print(f"  IPC: dom={av.dominance:.2f} warm={av.warmth:.2f} "
                  f"pace={av.pace}")

        os.unlink(path)

        # Belief seeding on belief questions
        if q_id in BELIEF_Qs:
            belief_model.seed_from_onboarding(q_text, answer, subject)

        signals = _extract_signals(q_text, answer)

        ack = random.choice([
            "Got it.", "Okay, noted.", "That's helpful.",
            "Makes sense.", "Interesting.", "Good to know.",
        ])
        speak(ack, archetype)
        return signals, ipc_dict

    return {}, neutral_ipc


def _build_profile(
        all_signals: list,
        all_ipc:     list,
        student_id:  str,
        belief_graph: dict,
) -> dict:
    dom_vals  = [v.get("dominance", 0.5) for v in all_ipc if v]
    warm_vals = [v.get("warmth",    0.6) for v in all_ipc if v]
    paces     = [v.get("pace") for v in all_ipc if v and v.get("pace")]

    avg_dom  = round(sum(dom_vals)  / len(dom_vals),  3) if dom_vals  else 0.5
    avg_warm = round(sum(warm_vals) / len(warm_vals), 3) if warm_vals else 0.6

    if avg_dom < 0.40 and avg_warm > 0.60:
        archetype = "maya"
    elif avg_dom > 0.60:
        archetype = "arjun"
    else:
        archetype = "lina"

    processing = _most_common([s.get("processing_style") for s in all_signals])
    chunk_size = 1 if processing == "bottom_up" else \
                 3 if processing == "top_down"  else 2

    misconceptions = [
        s["misconception_seed"] for s in all_signals
        if s.get("misconception_seed")
    ]

    return {
        "student_id":      student_id,
        "onboarding_done": True,
        "onboarding_date": datetime.now().isoformat(),

        "ipc": {
            "dominance":           avg_dom,
            "warmth":              avg_warm,
            "archetype":           archetype,
            "pace":                _most_common(paces) or "medium",
            "assertiveness_delta": 0.0,
            "persistence_score":   0.5,
        },

        "learning_style": {
            "processing_style":      processing,
            "analogy_receptiveness": _most_common([s.get("analogy_receptiveness") for s in all_signals]),
            "help_seeking_style":    _most_common([s.get("help_seeking_style") for s in all_signals]),
            "depth_vs_breadth":      _most_common([s.get("depth_vs_breadth") for s in all_signals]),
            "encouragement_need":    _most_common([s.get("encouragement_need") for s in all_signals]),
            "abstraction_comfort":   _most_common([s.get("abstraction_comfort") for s in all_signals]),
            "goal_type":             _most_common([s.get("goal_type") for s in all_signals]) or "boards",
        },

        "emotional": {
            "baseline":            _most_common([s.get("emotional_signal") for s in all_signals]),
            "socratic_tolerance":  _most_common([s.get("socratic_tolerance") for s in all_signals]),
            "persistence":         _most_common([s.get("persistence_signal") for s in all_signals]),
            "frustration_trigger_map": {},
            "giving_up_flag":      False,
        },

        "knowledge": {
            "subject_difficulty": next(
                (s.get("subject_difficulty") for s in all_signals
                 if s.get("subject_difficulty")), None
            ),
            "goal_type":   _most_common([s.get("goal_type") for s in all_signals]) or "boards",
            "mastery_map": {},
            "misconceptions": misconceptions,
        },

        "cognitive_load": {
            "chunk_size":              chunk_size,
            "working_memory_estimate": "medium",
            "consecutive_confused":    0,
            "zpd_sweet_spot":          0.50,
        },

        "comprehension": {
            "teach_back_counter": 0,
            "session_scores":     [],
        },

        "style_profile": {},
        "session_history": [],
        "_evidence_buffer": {"dominance": [], "warmth": []},
        "anomaly_log": [],
    }


def run_onboarding(
        student_id: str,
        subject:    str = "Mathematics",
) -> tuple[dict, dict]:
    """
    Run 12-question spoken onboarding.
    Returns (profile_dict, belief_graph_dict).

    Belief graph is seeded from ONBOARDING_BELIEF_Qs answers.
    IPC baseline set from ONBOARDING_IPC_RICH answers.
    ElevenLabs removed — Gemini TTS via speaker.py.
    """
    profile_path = Path(f"sessions/{student_id}/student_profile.json")
    belief_path  = Path(f"sessions/{student_id}/belief_graph.json")
    profile_path.parent.mkdir(parents=True, exist_ok=True)

    existing = _load_json(profile_path)
    if existing.get("onboarding_done"):
        arch = existing.get("ipc", {}).get("archetype", "unknown")
        print(f"  [SYRA] Profile loaded — {student_id} | arch: {arch}")
        belief_graph = _load_json(belief_path)
        return existing, belief_graph

    all_signals: list = existing.get("_partial_signals", [])
    all_ipc:     list = existing.get("_partial_ipc",     [])
    start_index  = len(all_signals)
    belief_graph: dict = _load_json(belief_path) or {"concepts": {}}
    belief_model = BeliefModel(belief_graph)

    reset_session_baseline()

    neutral_ipc = {"dominance": 0.5, "warmth": 0.7,
                   "pace": "medium", "giving_up": False}

    if start_index > 0:
        archetype = "lina"   # safe default before profile built
        speak(f"Welcome back. Let's continue from question {start_index + 1}.", archetype)
    else:
        archetype = "lina"
        print("\n" + "="*54)
        print("  SYRA — First-time setup")
        print("  Say 'repeat' to hear a question again,")
        print("  'go back' for the previous one, 'skip' to move on.")
        print("="*54 + "\n")
        speak(
            "Hi! I'm SYRA — your adaptive AI tutor. "
            "I'd like to understand how you think and learn. "
            "Twelve quick questions — just answer naturally. "
            "You can say 'repeat' if you want a question again.",
            archetype
        )

    i = start_index
    while i < len(QUESTIONS):
        q_id, q_text = QUESTIONS[i]
        prev_q_text  = QUESTIONS[i - 1][1] if i > 0 else None

        print(f"\n[{q_id.upper()}] SYRA: {q_text}")
        speak(q_text, archetype)

        try:
            signals, ipc_dict = _ask_question(
                q_id, q_text, archetype, prev_q_text,
                belief_model, subject
            )
            # Update archetype from IPC rich questions
            if q_id in IPC_RICH and ipc_dict.get("dominance"):
                dom  = ipc_dict["dominance"]
                warm = ipc_dict["warmth"]
                if dom < 0.40 and warm > 0.60:
                    archetype = "maya"
                elif dom > 0.60:
                    archetype = "arjun"
                else:
                    archetype = "lina"

            if i < len(all_signals):
                all_signals[i] = signals
                all_ipc[i]     = ipc_dict
            else:
                all_signals.append(signals)
                all_ipc.append(ipc_dict)

            _save_checkpoint(profile_path, all_signals, all_ipc)
            i += 1

        except _NavigationCommand as cmd:
            action = cmd.args[0]
            if action == "previous" and i > 0:
                i -= 1
                all_signals = all_signals[:i]
                all_ipc     = all_ipc[:i]
                _save_checkpoint(profile_path, all_signals, all_ipc)
            elif action == "previous" and i == 0:
                speak("This is the very first question.", archetype)
            elif action == "skip":
                if i < len(all_signals):
                    all_signals[i] = {}
                    all_ipc[i]     = neutral_ipc
                else:
                    all_signals.append({})
                    all_ipc.append(neutral_ipc)
                _save_checkpoint(profile_path, all_signals, all_ipc)
                i += 1

    # Build and save profile
    profile = _build_profile(all_signals, all_ipc, student_id, belief_graph)
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=4, ensure_ascii=False)

    # Save belief graph seeded from onboarding
    with open(belief_path, "w", encoding="utf-8") as f:
        json.dump(belief_graph, f, indent=4, ensure_ascii=False)

    arch = profile["ipc"]["archetype"]
    logger.info(f"Onboarding complete — {student_id} | arch: {arch}")
    speak(
        "All set — I've got a good picture of how you learn. Let's get started.",
        arch
    )

    return profile, belief_graph