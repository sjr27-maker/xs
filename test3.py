# tests/test_phase3_audio.py
"""Live audio pipeline test. Requires mic + API keys."""
import asyncio
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_audio_pipeline():
    from intake.recorder import record_until_enter, save_wav
    from intake.transcriber import transcribe
    from intake.acoustic_analyzer import extract_acoustic_vector
    from output.speaker import speak

    print("\n── Audio pipeline test")
    print("  Speak a few sentences when prompted.")
    print("  Press Enter BEFORE speaking, Enter again to STOP.\n")

    input("  Press Enter to start recording...")
    print("  Recording — speak now...")
    audio = record_until_enter()
    path  = save_wav(audio)
    print(f"  Recorded {len(audio)/16000:.1f}s of audio")

    print("  Transcribing via Deepgram...")
    text, fillers, dur = transcribe(path)
    print(f"  Transcript: '{text}'")
    print(f"  Fillers: {fillers} | Duration: {dur:.1f}s")

    print("  Analyzing acoustics...")
    av = extract_acoustic_vector(path, filler_count=fillers)
    print(f"  Dominance: {av.dominance:.2f}")
    print(f"  Warmth:    {av.warmth:.2f}")
    print(f"  Pace:      {av.pace}")
    print(f"  Fatigue:   {av.fatigue_score:.2f}")
    print(f"  Energy:    {av.voice_energy:.3f}")
    print(f"  Giving up: {av.giving_up}")

    os.unlink(path)

    print("\n  Testing Gemini TTS output...")
    speak("Audio pipeline test complete. SYRA is ready.", "lina", wait=True)
    print("  TTS: OK")

    if not text:
        print("\n  WARNING: Empty transcript — check Deepgram key and mic")
    elif av.dominance == 0.5 and av.warmth == 0.6:
        print("\n  WARNING: Acoustic signals at defaults — check audio length (need >2s)")
    else:
        print("\n  Audio pipeline: FULLY OPERATIONAL")

if __name__ == "__main__":
    test_audio_pipeline()