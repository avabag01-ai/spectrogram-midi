"""
Aegis Tuner Pro - FastAPI Backend
Streamlit 대체 → 고성능 REST API 서버
"""

import os
import sys
import io
import json
import base64
import tempfile
import uuid
import traceback
from pathlib import Path

import numpy as np
import librosa
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Aegis Engine imports
from aegis_engine import AegisEngine
from aegis_engine_core.synthesizer import (
    synthesize_midi, get_synthesizer, synthesize_midi_adsr,
    GUITAR_ADSR_PRESETS, get_adsr_synthesizer
)

# ─── App Setup ───────────────────────────────────────────
app = FastAPI(title="Aegis Tuner Pro", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
TEMP_DIR = Path(__file__).parent / "temp"
TEMP_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Engine singleton
engine = AegisEngine()

# Session storage: {session_id: {raw_data, events, midi_data, file_path, ...}}
sessions = {}


# ─── Helper ──────────────────────────────────────────────
def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Upload a file first.")
    return sessions[session_id]


# ─── Routes ──────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/presets")
async def get_presets():
    """ADSR 프리셋 목록"""
    return {
        "adsr_presets": GUITAR_ADSR_PRESETS,
        "effect_presets": list(_get_effect_presets().keys()),
    }

def _get_effect_presets():
    try:
        from aegis_engine_core.effect_learning_loop import EFFECT_PRESETS
        return EFFECT_PRESETS
    except:
        return {}


@app.post("/api/upload")
async def upload_audio(file: UploadFile = File(...)):
    """오디오 파일 업로드 → 세션 생성"""
    session_id = str(uuid.uuid4())[:8]

    # Save file
    suffix = Path(file.filename).suffix or ".mp3"
    file_path = TEMP_DIR / f"{session_id}{suffix}"

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    sessions[session_id] = {
        "file_path": str(file_path),
        "file_name": file.filename,
        "raw_data": None,
        "events": None,
        "midi_data": None,
    }

    return {
        "session_id": session_id,
        "file_name": file.filename,
        "message": "Upload successful"
    }


@app.post("/api/analyze/{session_id}")
async def analyze_audio(
    session_id: str,
    start_time: float = 0,
    end_time: float = None,
    confidence_threshold: float = 0.70,
    min_note_duration_ms: int = 100,
    sustain_ms: int = 200,
    rake_sensitivity: float = 0.6,
    midi_program: int = 27,
):
    """오디오 분석 → MIDI 변환"""
    sess = get_session(session_id)

    try:
        # Phase 1: Perception (heavy analysis)
        raw_data = engine.audio_to_midi(
            sess["file_path"], None,
            start_time=start_time,
            end_time=end_time,
            turbo_mode=False,
            rake_sensitivity=rake_sensitivity,
        )

        if raw_data is None:
            raise HTTPException(status_code=400, detail="Analysis failed - empty audio?")

        sess["raw_data"] = raw_data

        # Phase 2: Extract events + generate MIDI
        midi_buffer = io.BytesIO()
        events = engine.extract_events(
            raw_data, midi_buffer,
            confidence_threshold=confidence_threshold,
            min_note_duration_ms=min_note_duration_ms,
            sustain_ms=sustain_ms,
            midi_program=midi_program,
        )

        midi_buffer.seek(0)
        midi_data = midi_buffer.read()

        sess["events"] = events
        sess["midi_data"] = midi_data
        sess["params"] = {
            "confidence_threshold": confidence_threshold,
            "min_note_duration_ms": min_note_duration_ms,
            "sustain_ms": sustain_ms,
            "midi_program": midi_program,
        }

        # Piano roll data for frontend
        events_json = []
        for e in events:
            events_json.append({
                "note": e["note"],
                "start": int(e["start"]),
                "end": int(e["end"]),
                "velocity": int(e["velocity"]),
                "track": e.get("track", "main"),
                "technique": e.get("technique", "normal"),
            })

        return {
            "session_id": session_id,
            "num_events": len(events),
            "events": events_json,
            "midi_base64": base64.b64encode(midi_data).decode(),
            "message": f"Analysis complete: {len(events)} notes detected"
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/refilter/{session_id}")
async def refilter_events(
    session_id: str,
    confidence_threshold: float = 0.70,
    min_note_duration_ms: int = 100,
    sustain_ms: int = 200,
    midi_program: int = 27,
):
    """캐시된 raw_data로 빠르게 재필터링 (분석 없이 파라미터만 변경)"""
    sess = get_session(session_id)

    if sess["raw_data"] is None:
        raise HTTPException(status_code=400, detail="No analysis data. Run /api/analyze first.")

    try:
        midi_buffer = io.BytesIO()
        events = engine.extract_events(
            sess["raw_data"], midi_buffer,
            confidence_threshold=confidence_threshold,
            min_note_duration_ms=min_note_duration_ms,
            sustain_ms=sustain_ms,
            midi_program=midi_program,
        )

        midi_buffer.seek(0)
        midi_data = midi_buffer.read()

        sess["events"] = events
        sess["midi_data"] = midi_data
        sess["params"] = {
            "confidence_threshold": confidence_threshold,
            "min_note_duration_ms": min_note_duration_ms,
            "sustain_ms": sustain_ms,
            "midi_program": midi_program,
        }

        events_json = []
        for e in events:
            events_json.append({
                "note": e["note"],
                "start": int(e["start"]),
                "end": int(e["end"]),
                "velocity": int(e["velocity"]),
                "track": e.get("track", "main"),
                "technique": e.get("technique", "normal"),
            })

        return {
            "num_events": len(events),
            "events": events_json,
            "midi_base64": base64.b64encode(midi_data).decode(),
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/midi/{session_id}")
async def download_midi(session_id: str):
    """MIDI 파일 다운로드"""
    sess = get_session(session_id)
    if sess["midi_data"] is None:
        raise HTTPException(status_code=400, detail="No MIDI data. Run analysis first.")

    return Response(
        content=sess["midi_data"],
        media_type="audio/midi",
        headers={"Content-Disposition": f'attachment; filename="aegis_{sess["file_name"]}.mid"'}
    )


@app.post("/api/crossfade/{session_id}")
async def crossfade_mix(session_id: str, crossfade: float = 0.5):
    """원본↔MIDI 크로스페이드 믹스"""
    sess = get_session(session_id)
    if sess["midi_data"] is None:
        raise HTTPException(status_code=400, detail="No MIDI. Run analysis first.")

    try:
        # MIDI → WAV
        midi_wav = synthesize_midi(sess["midi_data"], sample_rate=44100)
        if not midi_wav:
            midi_wav = synthesize_midi_adsr(sess["midi_data"], preset='electric_clean')
        if not midi_wav:
            raise HTTPException(status_code=500, detail="MIDI synthesis failed")

        # Load original
        y_orig, _ = librosa.load(sess["file_path"], sr=44100, duration=30)

        # Load MIDI WAV
        y_midi, _ = librosa.load(io.BytesIO(midi_wav), sr=44100)

        # Match lengths
        min_len = min(len(y_orig), len(y_midi))
        y_orig = y_orig[:min_len]
        y_midi = y_midi[:min_len]

        # Mix
        y_mix = (1.0 - crossfade) * y_orig + crossfade * y_midi
        peak = np.max(np.abs(y_mix))
        if peak > 0:
            y_mix = y_mix / peak * 0.9

        # → WAV bytes
        import wave as wave_mod
        mix_int16 = np.clip(y_mix * 32767, -32768, 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave_mod.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(mix_int16.tobytes())

        return Response(content=buf.getvalue(), media_type="audio/wav")

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/midi-wav/{session_id}")
async def get_midi_wav(session_id: str):
    """MIDI → WAV 합성 결과"""
    sess = get_session(session_id)
    if sess["midi_data"] is None:
        raise HTTPException(status_code=400, detail="No MIDI.")

    wav = synthesize_midi(sess["midi_data"], sample_rate=44100)
    if not wav:
        wav = synthesize_midi_adsr(sess["midi_data"], preset='electric_clean')
    if not wav:
        raise HTTPException(status_code=500, detail="Synthesis failed")

    return Response(content=wav, media_type="audio/wav")


@app.get("/api/original-wav/{session_id}")
async def get_original_wav(session_id: str):
    """원본 오디오 WAV 반환 (브라우저 재생용)"""
    sess = get_session(session_id)

    y, _ = librosa.load(sess["file_path"], sr=44100, duration=60)

    import wave as wave_mod
    int16 = np.clip(y * 32767, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave_mod.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(int16.tobytes())

    return Response(content=buf.getvalue(), media_type="audio/wav")


@app.post("/api/adsr-synth/{session_id}")
async def adsr_synthesis(
    session_id: str,
    preset: str = "electric_clean",
    envelope_match: bool = False,
):
    """ADSR 소프트 신스 합성"""
    sess = get_session(session_id)
    if sess["midi_data"] is None:
        raise HTTPException(status_code=400, detail="No MIDI.")

    try:
        overrides = {}
        if envelope_match:
            synth = get_adsr_synthesizer()
            y_orig, _ = librosa.load(sess["file_path"], sr=44100, duration=10)
            overrides = synth.analyze_envelope(y_orig, sr=44100)

        wav = synthesize_midi_adsr(sess["midi_data"], preset=preset, **overrides)
        if not wav:
            raise HTTPException(status_code=500, detail="ADSR synthesis failed")

        result = {"wav_base64": base64.b64encode(wav).decode()}
        if envelope_match and overrides:
            result["envelope_params"] = overrides

        return result

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reverse-analysis/{session_id}")
async def reverse_analysis(session_id: str):
    """역변환 분석"""
    sess = get_session(session_id)
    if sess["midi_data"] is None:
        raise HTTPException(status_code=400, detail="No MIDI.")

    try:
        from aegis_engine_core.reverse_analyzer import reverse_analysis as do_reverse
        result = do_reverse(sess["midi_data"], engine, sample_rate=44100)

        if not result:
            raise HTTPException(status_code=500, detail="Reverse analysis failed")

        return {
            "original_notes": result["original_notes"],
            "reversed_notes": result["reversed_notes"],
            "note_accuracy": round(result["note_accuracy"], 3),
            "pitch_accuracy": round(result["pitch_accuracy"], 3),
            "timing_accuracy": round(result["timing_accuracy"], 3),
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auto-match/{session_id}")
async def auto_match(session_id: str):
    """자동 파라미터 매칭"""
    sess = get_session(session_id)
    if sess["raw_data"] is None:
        raise HTTPException(status_code=400, detail="No analysis data.")

    try:
        from aegis_engine_core.auto_matcher import auto_match_parameters
        result = auto_match_parameters(
            sess["file_path"], engine, sess["raw_data"], sample_rate=44100
        )

        if not result:
            raise HTTPException(status_code=500, detail="Auto-match failed")

        return result

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/effect-loop/{session_id}")
async def effect_learning_loop(
    session_id: str,
    preset: str = "clean",
    max_iterations: int = 5,
):
    """이펙트 학습 루프"""
    sess = get_session(session_id)
    if sess["midi_data"] is None:
        raise HTTPException(status_code=400, detail="No MIDI.")

    try:
        from aegis_engine_core.effect_learning_loop import learning_loop, EFFECT_PRESETS

        if preset not in EFFECT_PRESETS:
            preset = "clean"

        result = learning_loop(
            midi_data=sess["midi_data"],
            engine=engine,
            effects_config=EFFECT_PRESETS[preset],
            max_iterations=max_iterations,
            target_accuracy=0.95,
        )

        if not result:
            raise HTTPException(status_code=500, detail="Learning loop failed")

        return {
            "best_accuracy": result["best_accuracy"],
            "best_params": result["best_params"],
            "iterations": len(result.get("history", [])),
            "history": [
                {"iteration": h["iteration"], "overall": h["accuracy"]["overall"]}
                for h in result.get("history", [])
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/per-note-optimize/{session_id}")
async def per_note_optimize(
    session_id: str,
    quick_mode: bool = True,
    parallel: bool = True,
):
    """노트별 개별 최적화"""
    sess = get_session(session_id)
    if sess["events"] is None:
        raise HTTPException(status_code=400, detail="No events.")

    try:
        from aegis_engine_core.per_note_optimizer import (
            optimize_all_notes, optimize_all_notes_parallel,
            synthesize_with_per_note_params, generate_optimization_report
        )

        y_orig, _ = librosa.load(sess["file_path"], sr=44100, duration=30)
        events = sess["events"]

        if parallel and len(events) >= 10:
            opt_events = optimize_all_notes_parallel(
                events, y_orig, sr=44100, hop_length=512, quick_mode=quick_mode
            )
        else:
            opt_events = optimize_all_notes(
                events, y_orig, sr=44100, hop_length=512, quick_mode=quick_mode
            )

        if not opt_events:
            raise HTTPException(status_code=500, detail="Optimization failed")

        report = generate_optimization_report(opt_events)

        # Synthesize with optimized params
        opt_params = [e.get("adsr_params", {}) for e in opt_events]
        wav_data = synthesize_with_per_note_params(events, opt_params, sr=44100)

        result = {
            "report": report,
        }
        if wav_data:
            result["wav_base64"] = base64.b64encode(wav_data).decode()

        return result

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/fret-filter/{session_id}")
async def fret_filter(
    session_id: str,
    max_fret_speed: float = 40.0,
    protect_long_notes_ms: float = 200.0,
):
    """운지 기반 물리적 노이즈 필터"""
    sess = get_session(session_id)
    if sess["events"] is None:
        raise HTTPException(status_code=400, detail="No events. Run analysis first.")

    try:
        from aegis_engine_core.guitar_fret_filter import apply_fret_filter

        filtered_events, report = apply_fret_filter(
            sess["events"],
            sr=44100,
            hop_length=512,
            max_fret_speed=max_fret_speed,
            protect_long_notes_ms=protect_long_notes_ms,
        )

        # 세션 업데이트
        sess["events"] = filtered_events

        # MIDI 재생성
        midi_buffer = io.BytesIO()
        params = sess.get("params", {})
        events_for_midi = engine.extract_events(
            sess["raw_data"], midi_buffer,
            confidence_threshold=params.get("confidence_threshold", 0.70),
            min_note_duration_ms=params.get("min_note_duration_ms", 100),
            sustain_ms=params.get("sustain_ms", 200),
            midi_program=params.get("midi_program", 27),
        )
        midi_buffer.seek(0)
        midi_data = midi_buffer.read()
        sess["midi_data"] = midi_data

        events_json = []
        for e in filtered_events:
            events_json.append({
                "note": e["note"],
                "start": int(e["start"]),
                "end": int(e["end"]),
                "velocity": int(e["velocity"]),
                "track": e.get("track", "main"),
                "technique": e.get("technique", "normal"),
            })

        return {
            "report": report,
            "events": events_json,
            "midi_base64": base64.b64encode(midi_data).decode(),
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tabs/{session_id}")
async def get_tabs(session_id: str):
    """기타 탭 악보 데이터 생성"""
    sess = get_session(session_id)
    if sess["events"] is None:
        raise HTTPException(status_code=400, detail="No events. Run analysis first.")

    try:
        from aegis_engine_core.tabs import generate_tabs

        tab_data = generate_tabs(sess["events"])

        hop_length = 512
        sr = 44100
        for t in tab_data:
            t['time_sec'] = round(t['m_start'] * hop_length / sr, 4)
            t['end_sec'] = round(t['m_end'] * hop_length / sr, 4)

        return {"tabs": tab_data, "total_notes": len(tab_data)}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ─── Run ─────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8508, reload=False)
