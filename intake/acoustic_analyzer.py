# librosa IPC + fatigue_score + energy_trend
# intake/acoustic_analyzer.py
"""
Extracts acoustic signals from student audio.
IPC vector + fatigue indicators + session baseline comparison.
All computation is local (librosa) — zero API calls, ~50ms.
"""
import numpy as np
import librosa
from dataclasses import dataclass, field
from typing import Optional
from config import IN_RATE


@dataclass
class AcousticVector:
    # IPC signals
    dominance:           float = 0.5
    warmth:              float = 0.6
    pace:                str   = "medium"   # slow | medium | fast
    giving_up:           bool  = False
    assertiveness_delta: float = 0.0
    filler_count:        int   = 0

    # Fatigue signals (new)
    fatigue_score:       float = 0.0   # 0=fresh, 1=exhausted
    voice_energy:        float = 0.0   # RMS energy of this turn
    energy_trend:        str   = "stable"  # rising | stable | declining
    response_latency_ms: float = 0.0   # time from SYRA stop to student start

    # Session-relative
    pace_delta:          float = 0.0   # vs session baseline pace
    energy_delta:        float = 0.0   # vs session baseline energy


# Per-session baseline — updated on first 2 turns, then fixed
_session_baseline: dict = {
    "speech_rate":  None,  # syllables per second
    "rms_energy":   None,
    "initialized":  False,
}


def reset_session_baseline():
    """Call at session start to clear stale baseline."""
    global _session_baseline
    _session_baseline = {
        "speech_rate": None,
        "rms_energy":  None,
        "initialized": False,
    }


def _estimate_speech_rate(y: np.ndarray, sr: int) -> float:
    """Syllables per second via onset detection."""
    try:
        onsets     = librosa.onset.onset_detect(y=y, sr=sr, units="time")
        duration   = len(y) / sr
        return len(onsets) / duration if duration > 0.5 else 4.0
    except Exception:
        return 4.0


def _estimate_dominance(y: np.ndarray, sr: int) -> float:
    """
    Dominance from: pitch variance + volume + articulation rate.
    High dominance = confident, clear, varied pitch.
    """
    try:
        rms       = float(np.sqrt(np.mean(y ** 2)))
        f0, _, _  = librosa.pyin(y, fmin=80, fmax=400, sr=sr)
        f0_clean  = f0[~np.isnan(f0)]
        if len(f0_clean) > 5:
            pitch_var = float(np.std(f0_clean) / (np.mean(f0_clean) + 1e-6))
        else:
            pitch_var = 0.0
        raw = (rms * 3.0) + (pitch_var * 2.0)
        return float(np.clip(raw / 5.0, 0.0, 1.0))
    except Exception:
        return 0.5


def _estimate_warmth(y: np.ndarray, sr: int, dominance: float) -> float:
    """
    Warmth from: speaking rate (not rushed), pitch range, fluency.
    Inversely related to dominance at extremes.
    """
    try:
        rate  = _estimate_speech_rate(y, sr)
        # Warmth peaks at medium pace (3–5 syl/sec)
        pace_score = 1.0 - abs(rate - 4.0) / 4.0
        pace_score = float(np.clip(pace_score, 0.0, 1.0))

        # Spectral centroid — lower = warmer tone
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        cent_mean = float(np.mean(centroid))
        cent_score = 1.0 - np.clip(cent_mean / 4000.0, 0.0, 1.0)

        warmth = (pace_score * 0.5) + (cent_score * 0.3) + (0.2 * (1 - dominance))
        return float(np.clip(warmth, 0.0, 1.0))
    except Exception:
        return 0.6


def _estimate_pace(y: np.ndarray, sr: int,
                   baseline_rate: Optional[float]) -> tuple[str, float]:
    """
    Returns pace label and delta vs baseline.
    """
    rate = _estimate_speech_rate(y, sr)
    if baseline_rate is not None:
        delta = rate - baseline_rate
    else:
        delta = 0.0

    if rate < 2.8:
        label = "slow"
    elif rate > 5.2:
        label = "fast"
    else:
        label = "medium"

    return label, float(delta)


def _estimate_fatigue(
        y:              np.ndarray,
        sr:             int,
        rms:            float,
        baseline_rms:   Optional[float],
        baseline_rate:  Optional[float],
        filler_count:   int,
) -> float:
    """
    Fatigue score 0–1.
    Combines: energy drop vs baseline + pace drop + filler elevation.
    """
    score = 0.0
    weight_total = 0.0

    # Energy drop
    if baseline_rms and baseline_rms > 0:
        energy_ratio = rms / baseline_rms
        energy_fatigue = 1.0 - np.clip(energy_ratio, 0.0, 1.0)
        score        += energy_fatigue * 0.45
        weight_total += 0.45

    # Pace drop
    current_rate = _estimate_speech_rate(y, sr)
    if baseline_rate and baseline_rate > 0:
        rate_ratio = current_rate / baseline_rate
        rate_fatigue = 1.0 - np.clip(rate_ratio, 0.0, 1.0)
        score        += rate_fatigue * 0.35
        weight_total += 0.35

    # Filler elevation (normalised — 5+ fillers in one turn is high)
    filler_fatigue = np.clip(filler_count / 5.0, 0.0, 1.0)
    score        += filler_fatigue * 0.20
    weight_total += 0.20

    if weight_total == 0:
        return 0.0
    return float(np.clip(score / weight_total, 0.0, 1.0))


def _detect_giving_up(
        rms:            float,
        dominance:      float,
        filler_count:   int,
        prev_dominance: Optional[float],
) -> tuple[bool, float]:
    """
    Giving up: very low energy + low dominance + high fillers.
    Returns (giving_up, assertiveness_delta).
    """
    giving_up = (rms < 0.04 and dominance < 0.30 and filler_count >= 4)
    delta     = (dominance - prev_dominance) if prev_dominance is not None else 0.0
    return giving_up, float(delta)


def extract_acoustic_vector(
        audio_path:          str,
        filler_count:        int       = 0,
        prev_dominance:      Optional[float] = None,
        response_latency_ms: float     = 0.0,
) -> AcousticVector:
    """
    Main entry point.
    Loads audio, extracts all acoustic signals, updates session baseline.
    """
    global _session_baseline

    try:
        y, sr = librosa.load(audio_path, sr=IN_RATE, mono=True)
    except Exception as e:
        print(f"  [Acoustic load error: {e}]")
        return AcousticVector(filler_count=filler_count,
                              response_latency_ms=response_latency_ms)

    if len(y) < sr * 0.3:   # less than 300ms — not enough signal
        return AcousticVector(filler_count=filler_count,
                              response_latency_ms=response_latency_ms)

    rms        = float(np.sqrt(np.mean(y ** 2)))
    dominance  = _estimate_dominance(y, sr)
    warmth     = _estimate_warmth(y, sr, dominance)
    pace, pace_delta = _estimate_pace(
        y, sr, _session_baseline.get("speech_rate")
    )
    giving_up, assert_delta = _detect_giving_up(
        rms, dominance, filler_count, prev_dominance
    )
    fatigue = _estimate_fatigue(
        y, sr, rms,
        baseline_rms=_session_baseline.get("rms_energy"),
        baseline_rate=_session_baseline.get("speech_rate"),
        filler_count=filler_count,
    )
    energy_delta = rms - (_session_baseline.get("rms_energy") or rms)
    if   energy_delta >  0.02: energy_trend = "rising"
    elif energy_delta < -0.02: energy_trend = "declining"
    else:                      energy_trend = "stable"

    # Update baseline on first two turns (before contamination risk)
    if not _session_baseline["initialized"]:
        _session_baseline["speech_rate"] = _estimate_speech_rate(y, sr)
        _session_baseline["rms_energy"]  = rms
        _session_baseline["initialized"] = True

    return AcousticVector(
        dominance           = dominance,
        warmth              = warmth,
        pace                = pace,
        giving_up           = giving_up,
        assertiveness_delta = assert_delta,
        filler_count        = filler_count,
        fatigue_score       = fatigue,
        voice_energy        = rms,
        energy_trend        = energy_trend,
        response_latency_ms = response_latency_ms,
        pace_delta          = pace_delta,
        energy_delta        = energy_delta,
    )