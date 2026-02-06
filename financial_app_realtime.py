"""
🎸 Aegis Financial Engine - Real-time Interactive App
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
벡터 피아노롤 + 실시간 파라미터 반영 (Dual-Phase Architecture)

Phase 1: 오디오 분석 (1회, 캐싱)
Phase 2: MIDI 이벤트 추출 (실시간, 파라미터 변경 시)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import streamlit as st
import os
import base64
import mido
import numpy as np
import librosa
from aegis_engine_financial import AegisFinancialEngine
from aegis_engine_core.vision import detect_rake_patterns
from aegis_engine_core_v2.midi_logic_financial import get_midi_events_financial

st.set_page_config(
    page_title="Aegis Financial - Real-time",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper: Vector Piano Roll (Pure Python SVG)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def render_vector_piano_roll(events, height=500, theme="dark"):
    """
    Pure Python SVG 피아노롤 (100% 로컬, CDN 없음)
    """
    if not events:
        return "<div style='color:red;'>노트 없음</div>"

    bg_color = "#1a1d23" if theme == "dark" else "#F5F5DC"
    grid_color = "rgba(255,255,255,0.1)" if theme == "dark" else "rgba(0,0,0,0.15)"
    bar_color = "rgba(255,255,255,0.25)" if theme == "dark" else "rgba(0,0,0,0.3)"
    note_color = "#ff00cc" if theme == "dark" else "#4a90e2"
    text_color = "#ff00cc" if theme == "dark" else "#8b4513"

    # 음역대 계산
    pitches = [e['note'] for e in events]
    min_pitch = min(pitches) - 2
    max_pitch = max(pitches) + 2
    pitch_range = max(12, max_pitch - min_pitch)

    # 시간 범위
    max_time = max(e['end'] for e in events)

    # SVG 크기
    view_width = 1000
    view_height = height - 40

    time_scale = view_width / max_time if max_time > 0 else 1
    pitch_scale = view_height / pitch_range

    # SVG 생성
    svg_parts = [
        f'<svg width="100%" height="{view_height}" viewBox="0 0 {view_width} {view_height}" '
        f'xmlns="http://www.w3.org/2000/svg" style="background:{bg_color}; border-radius:8px; border:1px solid #444;">'
    ]

    # 그리드 (수평 - 피치)
    for p in range(int(min_pitch), int(max_pitch) + 1):
        y = view_height - (p - min_pitch) * pitch_scale
        svg_parts.append(
            f'<line x1="0" y1="{y}" x2="{view_width}" y2="{y}" stroke="{grid_color}" stroke-width="0.5" />'
        )

    # 그리드 (수직 - 시간)
    beats = int(max_time / 10)  # 대략 10프레임당 1비트
    for b in range(0, beats):
        x = b * 10 * time_scale
        color = bar_color if b % 4 == 0 else grid_color
        svg_parts.append(
            f'<line x1="{x}" y1="0" x2="{x}" y2="{view_height}" stroke="{color}" stroke-width="1" />'
        )

    # 노트 그리기
    for event in events:
        x = event['start'] * time_scale
        w = (event['end'] - event['start']) * time_scale
        y = view_height - (event['note'] - min_pitch + 1) * pitch_scale
        h = pitch_scale - 1

        # Confidence 기반 opacity
        opacity = 0.5 + (event.get('confidence', 0.7) * 0.5)

        # Track 기반 색상
        if event.get('track') == 'main':
            color = "#00ff88"  # 초록 (Main)
        else:
            color = note_color  # 기본 (Safe)

        svg_parts.append(
            f'<rect x="{x}" y="{y}" width="{max(2, w)}" height="{max(2, h)}" '
            f'fill="{color}" fill-opacity="{opacity}" rx="2" stroke="white" stroke-width="0.3" />'
        )

    svg_parts.append('</svg>')

    # 상단 정보
    main_count = sum(1 for e in events if e.get('track') == 'main')
    safe_count = len(events) - main_count

    header_html = f"""
    <div style="background:{bg_color}; border:1px solid #444; border-radius:8px; padding:10px; overflow:hidden;">
        <div style="font-family:monospace; font-size:10px; color:{text_color}; margin-bottom:5px; display:flex; justify-content:space-between;">
            <span>🎸 AEGIS FINANCIAL VECTOR ENGINE</span>
            <span>NOTES: {len(events)} (Main: {main_count}, Safe: {safe_count})</span>
        </div>
        {"".join(svg_parts)}
    </div>
    """

    return header_html


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Session State 초기화
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'audio_loaded' not in st.session_state:
    st.session_state.audio_loaded = False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Title
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.title("🎸 Aegis Financial Engine")
st.markdown("### **실시간 파라미터 조절 + 벡터 피아노롤**")
st.markdown("---")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sidebar: Parameters
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.sidebar.header("⚙️ 실시간 파라미터")

use_auto_threshold = st.sidebar.checkbox(
    "🧠 자동 Threshold",
    value=True,
    help="Bollinger Bands 자동 계산"
)

if not use_auto_threshold:
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        0.3, 0.9, 0.7, 0.05,
        help="낮을수록 더 많은 노트"
    )
else:
    confidence_threshold = None
    st.sidebar.success("✅ 자동 최적화")

rake_sensitivity = st.sidebar.slider(
    "Rake 민감도",
    0.1, 0.9, 0.6, 0.05,
    help="높을수록 엄격"
)

noise_gate_db = st.sidebar.slider(
    "Noise Gate (dB)",
    -80, -10, -40, 5
)

min_note_duration_ms = st.sidebar.slider(
    "최소 노트 길이 (ms)",
    10, 200, 50, 10
)

use_financial = st.sidebar.checkbox(
    "Financial 모드",
    value=True
)

st.sidebar.markdown("---")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 1: Audio Upload & Analysis (1회만, 캐싱)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

uploaded_file = st.file_uploader(
    "🎵 기타 솔로 업로드 (Phase 1: 1회 분석)",
    type=['wav', 'mp3'],
    help="분석은 1회만, 이후 파라미터 변경은 즉시 반영"
)

if uploaded_file and not st.session_state.audio_loaded:
    # 임시 파일 저장
    input_path = f"temp_{uploaded_file.name}"
    with open(input_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("🔍 Phase 1: 오디오 분석 중 (1회만)..."):
        engine = AegisFinancialEngine(sample_rate=22050)

        # 오디오 로드
        y, S_dB = engine.load_audio(input_path)

        # Rake 감지
        rake_mask = detect_rake_patterns(
            S_dB, engine.hop_length, engine.sr, rake_sensitivity
        )

        # PYIN 피치 추출
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('E2'),
            fmax=librosa.note_to_hz('C6'),
            sr=engine.sr,
            hop_length=engine.hop_length
        )

        # RMS 에너지
        rms = librosa.feature.rms(y=y, hop_length=engine.hop_length)[0]

        # 캐싱
        st.session_state.raw_data = {
            'rake_mask': rake_mask,
            'f0': f0,
            'voiced_flag': voiced_flag,
            'voiced_probs': voiced_probs,
            'rms': rms,
            'sr': engine.sr,
            'hop_length': engine.hop_length
        }
        st.session_state.audio_loaded = True

        st.success("✅ Phase 1 완료! 이제 파라미터를 조절하세요")

        # 임시 파일 제거
        os.remove(input_path)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2: Real-time MIDI Event Extraction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if st.session_state.audio_loaded:
    st.markdown("---")
    st.subheader("🎹 Phase 2: 실시간 MIDI 추출")

    # 실시간 추출 (파라미터 변경 시 즉시)
    with st.spinner("⚡ Phase 2: MIDI 이벤트 추출 중..."):
        raw = st.session_state.raw_data

        events = get_midi_events_financial(
            rake_mask=raw['rake_mask'],
            f0=raw['f0'],
            voiced_flag=raw['voiced_flag'],
            active_probs=raw['voiced_probs'],
            rms=raw['rms'],
            sr=raw['sr'],
            hop_length=raw['hop_length'],
            confidence_threshold=confidence_threshold,
            use_financial=use_financial,
            noise_gate_db=noise_gate_db,
            min_note_duration_ms=min_note_duration_ms
        )

    # 통계
    if events:
        main_count = sum(1 for e in events if e['track'] == 'main')
        safe_count = len(events) - main_count
        total = len(events)
        main_pct = (main_count / total * 100) if total > 0 else 0

        # 아티큘레이션 통계
        articulation_counts = {}
        for e in events:
            tech = e.get('technique')
            if tech:
                articulation_counts[tech] = articulation_counts.get(tech, 0) + 1

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Notes", total)

        with col2:
            st.metric("Main Track", f"{main_count} ({main_pct:.0f}%)")

        with col3:
            st.metric("Safe Track", safe_count)

        with col4:
            if use_financial:
                st.metric("Mode", "✅ Financial")
            else:
                st.metric("Mode", "⚠️ Median")

        # 아티큘레이션 분석
        if articulation_counts and use_financial:
            st.markdown("---")
            st.subheader("🎸 Financial 아티큘레이션 분석")

            col_a, col_b, col_c = st.columns(3)

            with col_a:
                bend_count = articulation_counts.get('bend', 0)
                st.metric("🔺 Bend (Bollinger)", bend_count)

            with col_b:
                vibrato_count = articulation_counts.get('vibrato', 0)
                st.metric("〰️ Vibrato (Bollinger)", vibrato_count)

            with col_c:
                slide_count = articulation_counts.get('slide', 0)
                st.metric("📊 Slide (MACD)", slide_count)

        st.markdown("---")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 벡터 피아노롤
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        st.subheader("🎹 벡터 피아노롤 (100% 로컬)")

        piano_roll_html = render_vector_piano_roll(events, height=400, theme="dark")
        st.markdown(piano_roll_html, unsafe_allow_html=True)

        st.markdown("---")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # MIDI 다운로드
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        if st.button("📥 MIDI 파일 생성 및 다운로드"):
            with st.spinner("MIDI 생성 중..."):
                # MIDI 생성
                mid = mido.MidiFile()
                track_main = mido.MidiTrack()
                track_safe = mido.MidiTrack()

                mid.tracks.append(track_main)
                mid.tracks.append(track_safe)

                # MetaMessage로 트랙 이름
                from mido import MetaMessage
                track_main.append(MetaMessage('track_name', name='Main', time=0))
                track_safe.append(MetaMessage('track_name', name='Safe', time=0))

                # 이벤트 추가
                ticks_per_beat = mid.ticks_per_beat
                ms_per_tick = 500 / ticks_per_beat

                last_time_main = 0
                last_time_safe = 0

                for evt in events:
                    track = track_main if evt['track'] == 'main' else track_safe
                    last_time = last_time_main if evt['track'] == 'main' else last_time_safe

                    ms_per_frame = (raw['hop_length'] / raw['sr']) * 1000
                    start_ms = evt['start'] * ms_per_frame
                    duration_ms = (evt['end'] - evt['start']) * ms_per_frame

                    start_ticks = int(start_ms / ms_per_tick)
                    duration_ticks = int(duration_ms / ms_per_tick)

                    delta_start = start_ticks - last_time

                    track.append(mido.Message(
                        'note_on',
                        note=evt['note'],
                        velocity=evt['velocity'],
                        time=delta_start
                    ))

                    track.append(mido.Message(
                        'note_off',
                        note=evt['note'],
                        velocity=0,
                        time=duration_ticks
                    ))

                    if evt['track'] == 'main':
                        last_time_main = start_ticks + duration_ticks
                    else:
                        last_time_safe = start_ticks + duration_ticks

                # 저장
                output_path = "realtime_output.mid"
                mid.save(output_path)

                # 다운로드 버튼
                with open(output_path, 'rb') as f:
                    st.download_button(
                        label="📥 MIDI 다운로드",
                        data=f.read(),
                        file_name="aegis_financial_realtime.mid",
                        mime="audio/midi"
                    )

    else:
        st.warning("⚠️ 노트가 감지되지 않았습니다. 파라미터를 조정하세요.")

else:
    # 시작 화면
    st.info("👆 오디오 파일을 업로드하세요 (Phase 1)")

    st.markdown("---")
    st.markdown("### 🚀 Dual-Phase Architecture")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Phase 1: Analysis (1회만, 30초)**
        - 오디오 로딩
        - Rake 감지
        - PYIN 피치 추출
        - RMS 에너지 계산
        → 캐싱됨
        """)

    with col2:
        st.markdown("""
        **Phase 2: Extraction (즉시, <1초)**
        - Financial 분석
        - MIDI 이벤트 추출
        - 벡터 피아노롤 렌더링
        → 파라미터 변경 시 즉시 반영!
        """)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Footer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 사용 팁")
st.sidebar.markdown("""
- Phase 1은 1회만 (느림)
- 파라미터 조절은 즉시 반영 (빠름!)
- 초록 = Main Track
- 분홍 = Safe Track
""")

st.sidebar.info("🎸 Aegis Financial v2.0 - Real-time")
