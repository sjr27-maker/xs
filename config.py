# ALL constants (thresholds, model names, gates)
# config.py
"""
All system constants. Change here, changes everywhere.
Never hardcode thresholds in individual files.
"""
from dataclasses import dataclass

# ── Model strings ─────────────────────────────────────────────────────
LIVE_MODEL      = "gemini-3.1-flash-live-preview"
ANALYSIS_MODEL  = "gemini-2.5-flash"
EXTRACT_MODEL   = "gemini-2.5-flash-lite"
EMBED_MODEL     = "models/text-embedding-004"

# ── Audio ─────────────────────────────────────────────────────────────
IN_RATE           = 16000   # Gemini 3.1 input requirement
OUT_RATE          = 24000   # Gemini 3.1 output rate
FRAME_SAMPLES     = 320     # exactly 20ms at 16kHz — webrtcvad hard requirement
PLAYER_BLOCKSIZE  = 2400    # 100ms at 24kHz

# ── Voice interruption gates ──────────────────────────────────────────
BARGE_IN_RMS          = 0.07   # minimum volume for barge-in
BARGE_IN_SPEECH_GATE  = 3      # consecutive speech frames required
BARGE_IN_COOLDOWN     = 2.5    # seconds between interrupts
END_SILENCE_FRAMES    = 22     # ~440ms silence → student finished

# ── IPC adaptation ────────────────────────────────────────────────────
ALPHA_SESSION_1       = 0.85   # trust onboarding heavily first session
ALPHA_SESSION_10_PLUS = 0.30   # live data dominates after 10 sessions
DRIFT_GATE_SESSIONS   = 3      # sessions required before base profile updates
DRIFT_MAGNITUDE       = 0.12   # minimum delta to trigger update

# ── ZPD estimator ─────────────────────────────────────────────────────
ZPD_ABOVE_THRESHOLD   = -0.20  # score below this = above ZPD
ZPD_BELOW_THRESHOLD   = +0.25  # score above this = below ZPD (too easy)
ZPD_LATENCY_WEIGHT    = 0.30
ZPD_CORRECTION_WEIGHT = 0.25
ZPD_HEDGING_WEIGHT    = 0.20
ZPD_ERROR_WEIGHT      = 0.25

# ── Affect tracker ────────────────────────────────────────────────────
AFFECT_WINDOW         = 4      # turns to consider for trajectory
STUCK_THRESHOLD       = 0.65   # frustration score to classify as STUCK
PRODUCTIVE_MAX        = 0.45   # below this = productive even if frustrated

# ── Working memory ────────────────────────────────────────────────────
WM_CAPACITY           = 4      # chunks (4±1, use conservative 4)
WM_OVERLOAD_ERRORS    = 2      # overload-induced errors before WM reset

# ── Dependency tracking ───────────────────────────────────────────────
DEPENDENCY_ALARM      = 0.70   # index above this triggers Socratic mode
DEPENDENCY_TURNS      = 3      # consecutive turns at alarm before triggering

# ── Spaced repetition ─────────────────────────────────────────────────
SR_INITIAL_STABILITY  = 0.5    # days until first review
SR_STABILITY_GROWTH   = 2.2    # multiplier per successful retrieval
SR_REVIEW_THRESHOLD   = 0.60   # retention below this → review due

# ── Prompt assembly ───────────────────────────────────────────────────
HISTORY_TURNS         = 4      # conversation turns to inject into prompt
MAX_PROMPT_WORDS      = 70     # SYRA response length target
RAG_CHARS             = 500    # max chars from NCERT retrieval

# ── Session management ────────────────────────────────────────────────
STYLE_UPDATE_EVERY    = 3      # turns between style profile updates
ANOMALY_BLOCK_ALL     = True   # if True, anomaly sessions block ALL profile updates

# ── Onboarding ───────────────────────────────────────────────────────
ONBOARDING_IPC_RICH   = {"q1", "q8", "q12"}  # questions with full IPC classification
ONBOARDING_BELIEF_Qs  = {"q1", "q3", "q8", "q11"}  # questions that seed belief graph


# ── Self-talk detection ───────────────────────────────────────────────
SELF_TALK_RMS_CEILING      = 0.05   # below this = likely self-talk
SELF_TALK_MIN_WORDS        = 8      # below this + quiet = self-talk
DIRECTED_SPEECH_MIN_WORDS  = 4      # minimum for directed speech

# ── Silence intervention ──────────────────────────────────────────────
CONFUSED_SILENCE_LIMIT     = 30.0   # seconds before confused-silence hint
PRODUCTIVE_SILENCE_LIMIT   = 90.0   # seconds before productive-silence check-in
SILENCE_GENTLE_CHECKIN_S   = 45.0   # "are you still working?" threshold

# ── Distraction detection ─────────────────────────────────────────────
DISTRACTION_RMS_SPIKE      = 3.0    # ratio of sudden background noise
DISTRACTION_DURATION_S     = 5.0    # sustained spike = distraction confirmed

# ── Give-up handling ──────────────────────────────────────────────────
PUSH_THROUGH_MID_PROBLEM   = True   # always push through mid-problem give-up