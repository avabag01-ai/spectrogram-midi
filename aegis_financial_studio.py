"""
🎸 Aegis Financial Studio
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
파라미터 튜닝 및 원본 비교를 위한 인터랙티브 스튜디오
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import streamlit as st
import os
import base64
import mido
import numpy as np
import matplotlib.pyplot as plt
from aegis_engine_financial import AegisFinancialEngine

st.set_page_config(
    page_title="Aegis Financial Studio",
    page_icon="🎸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Custom CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FF4B4B, #FF914D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #A0A0A0;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #363945;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
        height: 50px;
    }
</style>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sidebar
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.sidebar.markdown("## 🎛️ Control Panel")

# 1. Financial Settings
st.sidebar.markdown("### 📈 Financial Logic")
use_financial = st.sidebar.checkbox("Financial Analysis (주식 기법)", value=True)
if use_financial:
    use_auto_threshold = st.sidebar.checkbox("Auto Threshold (자동)", value=True)
    if not use_auto_threshold:
        confidence_threshold = st.sidebar.slider("Threshold (낮을수록 관대)", 0.1, 0.9, 0.5, 0.05)
    else:
        confidence_threshold = None
else:
    confidence_threshold = st.sidebar.slider("Median Filter Threshold", 0.1, 0.9, 0.6, 0.05)

# 2. Guitar Settings
st.sidebar.markdown("### 🎸 Guitar Filter")
use_guitar_filters = st.sidebar.checkbox("Guitar Filters (디스토션/Mute)", value=True)
rake_sensitivity = st.sidebar.slider("Rake Sensitivity (Rake 감지)", 0.1, 0.9, 0.6, 0.05)
min_note_duration_ms = st.sidebar.slider("Min Note Duration (ms)", 10, 200, 50, 10)
noise_gate_db = st.sidebar.slider("Noise Gate (dB)", -80, -10, -40, 5)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: 디스토션이 심하면 Rake를 낮추고 Duration을 높이세요.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Content
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown('<div class="main-header">🎸 Aegis Financial Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">주식 기술 분석(Bollinger, MACD, RSI)을 이용한 기타 오디오-MIDI 변환</div>', unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns([1, 1])

uploaded_file = None
input_path = "temp_input_studio.mp3"

with col1:
    st.markdown("### 1️⃣ 오디오 업로드")
    uploaded_file = st.file_uploader("MP3/WAV 파일을 드래그하세요", type=['wav', 'mp3'])

with col2:
    st.markdown("### 2️⃣ 원본 미리듣기")
    if uploaded_file:
        st.audio(uploaded_file)
        # 파일 저장
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    else:
        st.info("파일을 업로드하면 플레이어가 표시됩니다.")

if uploaded_file and st.button("🚀 변환 시작 (Start Conversion)", type="primary"):
    st.markdown("---")
    output_path = "output_studio.mid"
    
    with st.spinner("🎸 Aegis Engine이 오디오를 분석 중입니다..."):
        engine = AegisFinancialEngine(sample_rate=22050)
        
        # Engine Execution
        try:
            result = engine.audio_to_midi_financial(
                input_path,
                output_path,
                confidence_threshold=confidence_threshold,
                rake_sensitivity=rake_sensitivity,
                noise_gate_db=noise_gate_db,
                min_note_duration_ms=min_note_duration_ms,
                use_financial=use_financial,
                use_guitar_filters=use_guitar_filters
            )
            
            if result:
                st.success("✅ 변환 완료!")
                
                # MIDI Data Load
                mid = mido.MidiFile(output_path)
                main_notes = [m for m in mid.tracks[0] if m.type == 'note_on' and m.velocity > 0]
                safe_notes = [m for m in mid.tracks[1] if m.type == 'note_on' and m.velocity > 0]
                total = len(main_notes) + len(safe_notes)
                
                # Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Notes", total)
                m2.metric("Main Track (Core)", f"{len(main_notes)}")
                m3.metric("Safe Track (Ghost)", f"{len(safe_notes)}")
                m4.metric("Estimated Key", "Auto-detected") # 실제 키는 로그에만 찍혀서..
                
                st.markdown("---")
                
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                # Comparison Player
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                
                st.markdown("### 🎧 결과 비교 (Original vs MIDI)")
                
                c1, c2 = st.columns(2)
                
                with c1:
                    st.markdown("**원본 오디오 (Original)**")
                    st.audio(uploaded_file)
                    
                with c2:
                    st.markdown("**생성된 MIDI (Preview)**")
                    
                    with open(output_path, 'rb') as f:
                        midi_bytes = f.read()
                        midi_base64 = base64.b64encode(midi_bytes).decode()

                    # HTML MIDI Player (SoundFont based)
                    html_player = f"""
                    <script src="https://cdn.jsdelivr.net/npm/html-midi-player@1.5.0/dist/midi-player.min.js"></script>
                    <midi-player
                        src="data:audio/midi;base64,{midi_base64}"
                        sound-font
                        visualizer="#visualizer">
                    </midi-player>
                    <midi-visualizer
                        id="visualizer"
                        type="piano-roll"
                        src="data:audio/midi;base64,{midi_base64}">
                    </midi-visualizer>
                    """
                    st.components.v1.html(html_player, height=350)

                # Download
                st.download_button(
                    label="📥 MIDI 파일 다운로드",
                    data=midi_bytes,
                    file_name="aegis_studio_output.mid",
                    mime="audio/midi",
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"Error: {e}")
            st.write(e)
            
st.markdown("---")
st.caption("Powered by Aegis Financial Engine v2.0 | Google Deepmind Agentic Coding")
