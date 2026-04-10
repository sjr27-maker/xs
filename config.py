# config.py
from dataclasses import dataclass

# ── Model strings ─────────────────────────────────────────────────────
LIVE_MODEL      = "gemini-3.1-flash-live-preview"
ANALYSIS_MODEL  = "gemini-2.5-flash"
EXTRACT_MODEL   = "gemini-2.5-flash-lite"
EMBED_MODEL     = "models/text-embedding-004"

# ── Audio ─────────────────────────────────────────────────────────────
IN_RATE           = 16000
OUT_RATE          = 24000
FRAME_SAMPLES     = 320
PLAYER_BLOCKSIZE  = 2400

# ── Voice interruption gates ──────────────────────────────────────────
BARGE_IN_RMS          = 0.07
BARGE_IN_SPEECH_GATE  = 3
BARGE_IN_COOLDOWN     = 2.5
END_SILENCE_FRAMES    = 22

# ── IPC adaptation ────────────────────────────────────────────────────
ALPHA_SESSION_1       = 0.85
ALPHA_SESSION_10_PLUS = 0.30
DRIFT_GATE_SESSIONS   = 3
DRIFT_MAGNITUDE       = 0.12

# ── ZPD estimator ─────────────────────────────────────────────────────
ZPD_ABOVE_THRESHOLD   = -0.15   # -0.20 in production, -0.15 during testing
ZPD_BELOW_THRESHOLD   = +0.25
ZPD_LATENCY_WEIGHT    = 0.30
ZPD_CORRECTION_WEIGHT = 0.25
ZPD_HEDGING_WEIGHT    = 0.20
ZPD_ERROR_WEIGHT      = 0.25

# ── Affect tracker ────────────────────────────────────────────────────
AFFECT_WINDOW         = 4
STUCK_THRESHOLD       = 0.65
PRODUCTIVE_MAX        = 0.45

# ── Working memory ────────────────────────────────────────────────────
WM_CAPACITY           = 4
WM_OVERLOAD_ERRORS    = 2

# ── Dependency tracking ───────────────────────────────────────────────
DEPENDENCY_ALARM      = 0.70
DEPENDENCY_TURNS      = 3

# ── Spaced repetition ─────────────────────────────────────────────────
SR_INITIAL_STABILITY  = 0.5
SR_STABILITY_GROWTH   = 2.2
SR_REVIEW_THRESHOLD   = 0.60

# ── Prompt assembly ───────────────────────────────────────────────────
HISTORY_TURNS         = 4
MAX_PROMPT_WORDS      = 70
RAG_CHARS             = 500

# ── Session management ────────────────────────────────────────────────
STYLE_UPDATE_EVERY    = 3
ANOMALY_BLOCK_ALL     = True

# ── Onboarding ────────────────────────────────────────────────────────
ONBOARDING_IPC_RICH   = {"q1", "q8", "q12"}
ONBOARDING_BELIEF_Qs  = {"q1", "q3", "q8", "q11"}

# ── Self-talk detection ───────────────────────────────────────────────
SELF_TALK_RMS_CEILING     = 0.05
SELF_TALK_MIN_WORDS       = 8
DIRECTED_SPEECH_MIN_WORDS = 4

# ── Silence intervention ──────────────────────────────────────────────
CONFUSED_SILENCE_LIMIT    = 30.0
PRODUCTIVE_SILENCE_LIMIT  = 90.0
SILENCE_GENTLE_CHECKIN_S  = 45.0

# ── Distraction detection ─────────────────────────────────────────────
DISTRACTION_RMS_SPIKE     = 3.0
DISTRACTION_DURATION_S    = 5.0

# ── Give-up handling ──────────────────────────────────────────────────
PUSH_THROUGH_MID_PROBLEM  = True