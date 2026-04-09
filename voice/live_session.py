# restart-per-turn, Gemini, full adaptation
# voice/live_session.py
"""
Full duplex tutoring session — restart-per-turn architecture.
Every turn reconnects with a fresh adaptive system prompt.
All cognitive and pedagogical layers wired in.
ElevenLabs removed — Gemini native audio only.
"""
import asyncio
import os
import json
import logging
import threading
import tempfile
import queue
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io.wavfile as wav_io
import sounddevice as sd
import webrtcvad

from google import genai
from google.genai import types
from dotenv import load_dotenv

from intake.acoustic_analyzer  import extract_acoustic_vector, reset_session_baseline
from intake.session_checker    import SessionContext
from intake.transcriber        import transcribe

from cognition.belief_model     import BeliefModel
from cognition.error_classifier import classify_error
from cognition.zpd_estimator    import ZPDEstimator
from cognition.affect_tracker   import AffectTracker
from cognition.wm_monitor       import WMMonitor
from cognition.dependency_tracker import DependencyTracker

from pedagogy.situational_gate  import SituationalGate
from pedagogy.question_planner  import QuestionPlanner
from pedagogy.spaced_repetition import get_review_prompt

from prompt.assembler    import build_system_prompt, AssemblerInput
from output.speaker      import StreamingPlayer, stop_speaking, OUT_RATE
from memory.profile_manager import ProfileManager
from memory.session_memory  import SessionMemory
import memory.belief_graph as bg_module

from style.vocabulary_mirror import get_mirror_instruction, \
    extract_session_style, update_style_profile
from feedback.session_scorer import score_session

from config import (
    LIVE_MODEL, IN_RATE, FRAME_SAMPLES,
    BARGE_IN_RMS, BARGE_IN_SPEECH_GATE,
    BARGE_IN_COOLDOWN, END_SILENCE_FRAMES,
    STYLE_UPDATE_EVERY,
)

load_dotenv()
logger  = logging.getLogger("SYRA.Live")
_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


class FullDuplexSession:

    def __init__(
            self,
            profile:      dict,
            pm:           ProfileManager,
            sm:           SessionMemory,
            belief_graph: dict,
            session_ctx:  SessionContext,
            subject:      str,
            grade:        int,
    ):
        self.profile      = profile
        self.pm           = pm
        self.sm           = sm
        self.belief_graph = belief_graph
        self.session_ctx  = session_ctx
        self.subject      = subject
        self.grade        = grade
        self.student_id   = profile.get("student_id", "student")
        self.archetype    = profile.get("ipc", {}).get("archetype", "lina")

        # Adapted IPC for this session
        self.adapted_ipc  = {
            "dominance":  profile["ipc"].get("dominance",  0.5),
            "warmth":     profile["ipc"].get("warmth",     0.6),
            "pace":       profile["ipc"].get("pace",       "medium"),
            "giving_up":  False,
            "filler_count": 0,
            "assertiveness_delta": 0.0,
        }
        self.prev_dominance = profile["ipc"].get("dominance", 0.5)

        # Cognitive modules — persist across turns
        self.belief_model  = BeliefModel(belief_graph)
        self.zpd           = ZPDEstimator()
        self.affect        = AffectTracker()
        self.wm            = WMMonitor()
        self.dependency    = DependencyTracker()
        self.gate          = SituationalGate()
        self.planner       = QuestionPlanner()

        # Session state
        self.turn_num             = 0
        self.consecutive_confused = 0
        self.current_topic: Optional[str] = None
        self.last_student_text    = ""
        self.last_ai_text         = ""
        self.style_profile        = profile.get("style_profile", {})
        self.session_count        = len(profile.get("session_history", []))

        # Conversation continuity across reconnects
        self.conversation_history: list[dict] = []

        # Per-turn audio capture for IPC
        self._turn_audio: list = []

        # Voice
        self._player          = StreamingPlayer(OUT_RATE)
        self._ai_speaking     = False
        self._last_barge_time = 0.0
        self._consec_speech   = 0
        self._running         = True
        self._current_out_txt = ""

    # ── IPC classification ────────────────────────────────────────────

    def _classify_turn_ipc(self) -> dict:
        if not self._turn_audio:
            return self.adapted_ipc
        try:
            audio = np.concatenate(self._turn_audio).astype(np.float32)
            tmp   = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            wav_io.write(tmp.name, IN_RATE, (audio * 32767).astype(np.int16))

            filler = self.adapted_ipc.get("filler_count", 0)
            av     = extract_acoustic_vector(
                tmp.name,
                filler_count=filler,
                prev_dominance=self.prev_dominance,
            )
            os.unlink(tmp.name)

            blended = self.pm.get_session_adapted_ipc({
                "dominance":           av.dominance,
                "warmth":              av.warmth,
                "pace":                av.pace,
                "giving_up":           av.giving_up,
                "filler_count":        av.filler_count,
                "assertiveness_delta": av.assertiveness_delta,
            })
            self.prev_dominance = av.dominance
            return {**blended, "fatigue_score": av.fatigue_score,
                    "energy_trend": av.energy_trend}

        except Exception as e:
            logger.debug(f"IPC classify error: {e}")
            return self.adapted_ipc

    # ── Build system prompt ───────────────────────────────────────────

    def _build_config(self) -> types.LiveConnectConfig:
        """Fresh adaptive prompt every turn — this IS the adaptation."""

        # Cognitive states
        from intake.acoustic_analyzer import AcousticVector
        from dataclasses import asdict

        # Build acoustic vector stub from adapted_ipc
        acoustic_stub = type("AV", (), {
            "dominance":    self.adapted_ipc.get("dominance",  0.5),
            "warmth":       self.adapted_ipc.get("warmth",     0.6),
            "pace":         self.adapted_ipc.get("pace",       "medium"),
            "giving_up":    self.adapted_ipc.get("giving_up",  False),
            "filler_count": self.adapted_ipc.get("filler_count", 0),
            "fatigue_score": self.adapted_ipc.get("fatigue_score", 0.0),
            "energy_trend": self.adapted_ipc.get("energy_trend", "stable"),
        })()

        # Get current cognitive states
        zpd_est    = self.zpd._compute_estimate()
        affect_st  = self.affect._history[-1] if self.affect._history \
                     else type("A", (), {
                         "frustration_type": "NONE",
                         "trajectory": "stable",
                         "intensity": 0.0,
                         "emotional_state": "neutral",
                         "intervention_needed": False,
                     })()
        wm_st      = self.wm.get_state()
        dep_st     = self.dependency._history[-1] if self.dependency._history \
                     else type("D", (), {
                         "index": 0.0, "level": "LOW",
                         "alarm_triggered": False,
                         "turns_at_alarm": 0,
                         "trend": "stable",
                     })()

        # Error analysis (last turn's)
        from cognition.error_classifier import ErrorAnalysis
        last_error = ErrorAnalysis(error_type="NONE", detected=False)

        # Gate evaluation
        proposed_type = "COMPREHENSION"
        gate_result   = self.gate.evaluate(
            proposed_type, affect_st, zpd_est, wm_st, dep_st, self.turn_num
        )
        allowed = gate_result.allowed_alternatives or [proposed_type]

        # Question plan
        q_plan = self.planner.plan(
            zpd_est, affect_st, self.belief_model,
            self.current_topic or "general",
            self.turn_num, allowed,
        )

        # Spaced review
        review_prompt = get_review_prompt(
            self.belief_graph, self.current_topic or "", self.subject
        )

        # Style mirror
        mirror = get_mirror_instruction(self.style_profile, self.session_count)

        inp = AssemblerInput(
            student_id            = self.student_id,
            subject               = self.subject,
            grade                 = self.grade,
            archetype             = self.archetype,
            turn_num              = self.turn_num,
            session_ctx           = self.session_ctx,
            acoustic              = acoustic_stub,
            belief_snapshot       = self.belief_model.get_snapshot(
                                        self.current_topic
                                    ),
            zpd                   = zpd_est,
            affect                = affect_st,
            wm                    = wm_st,
            dependency            = dep_st,
            error_analysis        = last_error,
            consecutive_confused  = self.consecutive_confused,
            gate_result           = gate_result,
            question_plan         = q_plan,
            wm_monitor            = self.wm,
            conversation_history  = self.conversation_history,
            unresolved_misconceptions = self.sm.context.get(
                "unresolved_misconceptions", []
            ),
            topics_due_review     = [],
            spaced_review_prompt  = review_prompt,
            rag_context           = "",   # added below if available
            style_mirror          = mirror,
        )

        sys_prompt = build_system_prompt(inp)

        voice = {"maya": "Leda", "lina": "Aoede", "arjun": "Charon"}.get(
            self.archetype, "Aoede"
        )

        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=types.Content(
                parts=[types.Part(text=sys_prompt)]
            ),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice
                    )
                )
            ),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
        )

    # ── Send loop ─────────────────────────────────────────────────────

    async def _send_turn(self, session) -> bool:
        """
        Stream one student turn. Returns True if barge-in occurred.
        4-gate interruption system.
        """
        audio_q = queue.Queue(maxsize=60)
        vad     = webrtcvad.Vad(2)
        self._turn_audio = []

        def mic_cb(indata, frames, time_info, status):
            try:
                audio_q.put_nowait(indata[:, 0].copy())
            except queue.Full:
                pass

        in_speech, silence_frames = False, 0
        self._consec_speech = 0

        with sd.InputStream(
            samplerate=IN_RATE, channels=1,
            device=int(os.getenv("DEVICE_INDEX", "1")),
            dtype="float32", blocksize=FRAME_SAMPLES, callback=mic_cb,
        ):
            while True:
                try:
                    chunk = await asyncio.to_thread(audio_q.get, True, 0.5)
                except queue.Empty:
                    continue

                chunk = np.clip(chunk, -1.0, 1.0)
                pcm16 = (chunk * 32767).astype(np.int16)
                rms   = float(np.sqrt(np.mean(chunk ** 2)))
                now   = asyncio.get_event_loop().time()

                try:
                    has_speech = vad.is_speech(pcm16.tobytes(), IN_RATE)
                except Exception:
                    has_speech = rms > 0.02

                # Track consecutive speech frames for gate 3
                self._consec_speech = self._consec_speech + 1 \
                    if has_speech else 0

                # 4-gate barge-in
                if (
                    self._ai_speaking
                    and has_speech                                         # gate 1
                    and rms > BARGE_IN_RMS                                 # gate 2
                    and self._consec_speech >= BARGE_IN_SPEECH_GATE        # gate 3
                    and now - self._last_barge_time > BARGE_IN_COOLDOWN    # gate 4
                ):
                    self._player.clear()
                    self._ai_speaking     = False
                    self._last_barge_time = now
                    self._consec_speech   = 0
                    print("\n  [Interrupted]")
                    try:
                        await session.send_realtime_input(audio_stream_end=True)
                    except Exception:
                        pass
                    await asyncio.sleep(0.1)
                    return True   # was interrupted

                if has_speech:
                    silence_frames = 0
                    in_speech      = True
                    self._turn_audio.append(chunk.copy())
                    await session.send_realtime_input(
                        audio=types.Blob(
                            data=pcm16.tobytes(),
                            mime_type="audio/pcm;rate=16000",
                        )
                    )
                elif in_speech:
                    silence_frames += 1
                    if silence_frames >= END_SILENCE_FRAMES:
                        await session.send_realtime_input(audio_stream_end=True)
                        return False   # normal end

    # ── Receive loop ──────────────────────────────────────────────────

    async def _receive_turn(self, session) -> str:
        ai_text     = ""
        self._current_out_txt = ""

        async for response in session.receive():
            sc = getattr(response, "server_content", None)
            if not sc:
                continue

            # Student transcript
            it = getattr(sc, "input_transcription", None)
            if it:
                txt = getattr(it, "text", "").strip()
                if txt:
                    print(f"\n  You: {txt}")
                    self.last_student_text = txt
                    words = [w for w in txt.split() if len(w) > 3]
                    if words:
                        self.current_topic = words[-1].lower()
                    if any(w in txt.lower() for w in [
                        "bye syra", "goodbye syra",
                        "stop session", "end session",
                    ]):
                        self._running = False
                        return ""

            # AI audio
            mt = getattr(sc, "model_turn", None)
            if mt and mt.parts:
                now = asyncio.get_event_loop().time()
                if now - self._last_barge_time > 0.8:
                    self._ai_speaking = True
                    for part in mt.parts:
                        d = getattr(part, "inline_data", None)
                        if d and getattr(d, "data", None):
                            self._player.feed(d.data)

            # AI transcript
            ot = getattr(sc, "output_transcription", None)
            if ot:
                txt = getattr(ot, "text", "").strip()
                if txt and len(txt) > len(self._current_out_txt):
                    self._current_out_txt = txt
                    print(f"\r  SYRA: {self._current_out_txt}",
                          end="", flush=True)
                    ai_text = txt

            if getattr(sc, "turn_complete", False):
                if self._current_out_txt:
                    print()
                self._ai_speaking = False
                self.last_ai_text = ai_text
                return ai_text

        return ai_text

    # ── Post-turn adaptation ──────────────────────────────────────────

    def _adapt_after_turn(self, student_text: str, ai_text: str):
        """All adaptation runs here between turns. ~150ms total."""

        # 1. IPC from this turn's audio
        self.adapted_ipc = self._classify_turn_ipc()

        # 2. Error classification (sync, ~200ms — acceptable)
        error = classify_error(
            student_text, ai_text,
            concept=self.current_topic or "general",
            fatigue_level=self.session_ctx.fatigue_level,
            consecutive_confused=self.consecutive_confused,
        )

        # 3. ZPD update
        zpd_est = self.zpd.update(
            student_text=student_text,
            response_latency_ms=0.0,
            error_type=error.error_type,
            filler_count=self.adapted_ipc.get("filler_count", 0),
            giving_up=self.adapted_ipc.get("giving_up", False),
        )

        # 4. Affect update
        affect_st = self.affect.update(
            ipc_giving_up=self.adapted_ipc.get("giving_up", False),
            filler_count=self.adapted_ipc.get("filler_count", 0),
            energy_trend=self.adapted_ipc.get("energy_trend", "stable"),
            error_type=error.error_type,
            student_text=student_text,
            consecutive_confused=self.consecutive_confused,
        )

        # 5. WM update
        if error.error_type == "OVERLOAD_INDUCED":
            self.wm.report_overload_error()
        if self.current_topic:
            self.wm.mark_introduced(self.current_topic)

        # 6. Dependency update
        dep_st = self.dependency.update(
            student_word_count=len(student_text.split()),
            syra_word_count=len(ai_text.split()),
        )

        # 7. Confusion counter
        was_correct = any(
            w in ai_text.lower()
            for w in ["exactly", "correct", "right", "perfect", "well done"]
        )
        self.consecutive_confused = max(
            0, self.consecutive_confused + (-1 if was_correct else 1)
        )

        # 8. Session memory
        self.sm.add_turn(
            student_text=student_text,
            ai_response=ai_text,
            ipc_vector=self.adapted_ipc,
            acoustic={},
            error_type=error.error_type,
            affect_state=affect_st.emotional_state,
            zpd_position=zpd_est.position,
            wm_slots=self.wm.get_state().slots_used,
            dependency_idx=dep_st.index,
        )

        # 9. Conversation history
        if student_text:
            self.conversation_history.append(
                {"role": "student", "content": student_text}
            )
        if ai_text:
            self.conversation_history.append(
                {"role": "syra", "content": ai_text}
            )
        self.conversation_history = self.conversation_history[-16:]

        # 10. Background: belief update
        if student_text and ai_text and self.current_topic:
            threading.Thread(
                target=self.belief_model.update_from_exchange,
                args=(student_text, ai_text,
                      self.current_topic, error.error_type),
                daemon=True,
            ).start()

        # 11. Background: style update every N turns
        if self.turn_num % STYLE_UPDATE_EVERY == 0 and len(self.sm.turns) >= 3:
            def _update_style():
                new = extract_session_style(
                    [{"student_text": t.student_text} for t in self.sm.turns[-10:]]
                )
                self.style_profile = update_style_profile(
                    self.style_profile, new
                )
            threading.Thread(target=_update_style, daemon=True).start()

        print(
            f"  [Adapt] dom={self.adapted_ipc['dominance']:.2f} "
            f"zpd={zpd_est.position} affect={affect_st.frustration_type} "
            f"error={error.error_type} confused={self.consecutive_confused}"
        )

    # ── Audio drain ───────────────────────────────────────────────────

    async def _drain_audio(self, timeout: float = 10.0):
        await asyncio.sleep(0.3)
        elapsed = 0.0
        while self._player.playing and elapsed < timeout:
            await asyncio.sleep(0.08)
            elapsed += 0.08
        await asyncio.sleep(0.12)

    # ── Main run ──────────────────────────────────────────────────────

    async def run(self):
        print("\n" + "="*52)
        print(f"  SYRA Live | Class {self.grade} {self.subject} "
              f"| {self.archetype.upper()}")
        print("  Speak naturally. Say 'bye SYRA' to end.")
        print("="*52)

        reset_session_baseline()

        # Greeting turn
        print("\n  Starting session...\n")
        try:
            async with _client.aio.live.connect(
                model=LIVE_MODEL, config=self._build_config()
            ) as session:
                await session.send_realtime_input(
                    text=(
                        f"[Session started. Greet {self.student_id} warmly. "
                        f"{'Student is fatigued today — keep it light and short.' if self.session_ctx.anomaly_flag else ''} "
                        f"Ask what topic in {self.subject} they want to work on.]"
                    )
                )
                await self._receive_turn(session)
            await self._drain_audio()
        except Exception as e:
            print(f"  Greeting error: {e}")

        # Main loop
        while self._running:
            self.turn_num += 1
            print(f"\n  Listening (turn {self.turn_num})...")

            try:
                async with _client.aio.live.connect(
                    model=LIVE_MODEL,
                    config=self._build_config(),  # fresh adaptive prompt
                ) as session:
                    _, ai_text = await asyncio.gather(
                        self._send_turn(session),
                        self._receive_turn(session),
                    )

                self._adapt_after_turn(
                    self.last_student_text, ai_text or ""
                )
                await self._drain_audio()

            except asyncio.TimeoutError:
                print("  Timeout — restarting turn...")
                await asyncio.sleep(0.3)
            except Exception as e:
                print(f"  Turn error: {e}")
                if not self._running:
                    break
                await asyncio.sleep(0.3)

        await self._end_session()

    # ── Session end ───────────────────────────────────────────────────

    async def _end_session(self):
        self._player.close()
        if not self.sm.turns:
            print("\n  No turns to save.")
            return

        print("\n  Scoring session...")
        session_data = score_session(
            self.sm.turns, self.subject,
            self.session_ctx.anomaly_flag,
        )
        print(
            f"  Comprehension: {session_data['comprehension_score']}/100 "
            f"[{session_data['session_classification']}]"
        )
        print(f"  Key insight: {session_data['key_insight']}")

        # IPC summary
        dom_vals  = [t.ipc_vector.get("dominance", 0.5) for t in self.sm.turns]
        warm_vals = [t.ipc_vector.get("warmth",    0.6) for t in self.sm.turns]
        ipc_summary = {
            "avg_dominance": round(sum(dom_vals)/len(dom_vals), 3),
            "avg_warmth":    round(sum(warm_vals)/len(warm_vals), 3),
        }

        # Style final update
        final_style = extract_session_style(
            [{"student_text": t.student_text} for t in self.sm.turns]
        )
        self.style_profile = update_style_profile(self.style_profile, final_style)

        # Profile updates (gated)
        anomaly = self.session_ctx.anomaly_flag
        self.pm.update_base_profile(ipc_summary, len(self.sm.turns), anomaly)
        self.pm.update_knowledge(
            mastery_updates={},
            new_misconceptions=session_data.get("new_misconceptions", []),
            comprehension_score=session_data.get("comprehension_score", 0),
            topics_covered=session_data.get("topics_covered", []),
            topics_struggling=session_data.get("topics_struggling", []),
            recommended_next=session_data.get("recommended_next_topic"),
            session_anomaly=anomaly,
        )
        self.pm.profile["style_profile"] = self.style_profile
        if not anomaly:
            bg_module.save(self.student_id, self.belief_graph)
        self.pm.save()

        # Save log
        out_dir  = Path(f"sessions/{self.student_id}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        log = {
            **self.sm.to_log(),
            **session_data,
            "ipc_summary":    ipc_summary,
            "session_anomaly": anomaly,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2, ensure_ascii=False)

        print(f"  Saved → {out_path}")
        print(f"  α next: {self.pm.get_alpha():.2f} | "
              f"arch: {self.pm.profile['ipc']['archetype']}")