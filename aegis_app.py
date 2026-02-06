import streamlit as st
import os
import shutil
import base64
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from aegis_engine import AegisEngine
from aegis_engine_core.visualizers import render_vector_piano_roll # Import modular visualizer
import tempfile
import io

st.set_page_config(page_title="Aegis Engine Control", layout="wide")

st.markdown("""
# 🛡️ Aegis Engine: Tuning Center
### 오디오 신호 처리(DSP) 및 비전 분석(Vision AI) 제어 패널

---
### 📖 정밀 튜닝 가이드 (바 방향에 따른 반응)

Aegis 엔진의 파라미터는 방향에 따라 결과가 극명하게 갈립니다.

*   **[Noise Gate] 게이트 임계값**
    *   **← 왼쪽 (-80dB)**: **"다 잡아내!"** - 아주 작은 소리까지 감지합니다. (음이 너무 안 나올 때)
    *   **→ 오른쪽 (-10dB)**: **"깔끔하게!"** - 시끄러운 잡음은 무조건 버립니다. (지저분할 때)
*   **[Guardian] 신뢰도 임계값**
    *   **← 왼쪽 (0.50)**: **"관대하게!"** - AI가 조금만 비슷해도 악보에 그립니다. (노트 수 급증)
    *   **→ 오른쪽 (0.99)**: **"확실한 것만!"** - 100% 확신하는 음만 기록합니다. (신뢰도 최상)
*   **[Smoothing] 최소 지속 시간**
    *   **← 왼쪽 (10ms)**: **"속사포 모드!"** - 찰나의 음도 다 잡아냅니다. (속주용)
    *   **→ 오른쪽 (200ms)**: **"정돈된 모드!"** - 짧은 소음은 무시하고 긴 호흡의 음만 남깁니다.
*   **[Rake] 잡음 감지 민감도**
    *   **← 왼쪽 (0.1)**: **"엄격한 검열!"** - 조금만 긁어도 잡음으로 간주합니다.
    *   **→ 오른쪽 (0.9)**: **"자유로운 소생!"** - 잡음 같아도 일단 음정으로 살려줍니다.
---
""")

# Sidebar: Non-Fragment Sliders (Global Config)
st.sidebar.subheader("🚀 1. Performance (성능)")
turbo_mode = st.sidebar.checkbox(
    "Turbo Mode (멀티코어)", 
    value=False,
    help="멀티 프로세싱을 시도합니다. 시스템 환경에 따라 오류가 발생할 수 있습니다."
)
zen_mode = st.sidebar.checkbox(
    "⚓ Zen Mode (초고속 데이터 전용)",
    value=True,
    help="모든 무거운 그래픽을 비활성화하고 오직 미디 데이터와 로그만 보여줍니다."
)
full_band_mode = st.sidebar.checkbox(
    "🎸 Full Band Mode (Aegis AI)",
    value=False,
    help="Aegis 내부 AI(Demucs)를 사용하여 기타를 분리합니다. (오래 걸림)"
)
external_stem_mode = st.sidebar.checkbox(
    "🎹 Logic Pro External Stem",
    value=False,
    help="로직 프로 등 외부에서 분리한 고품질 기타 스템 파일을 직접 사용합니다."
)

st.sidebar.subheader("📏 2. Cleaning (잔파동 제거)")
min_duration_ms = st.sidebar.slider(
    "최소 지속 시간 (ms)", 
    min_value=10, max_value=200, value=100, step=10,
    help="음이 너무 안 나온다면 왼쪽(←)으로 옮기세요!"
)

st.sidebar.subheader("2. Guardian Sensitivity (보안 등급)")
confidence_thresh = st.sidebar.slider(
    "Guardian 신뢰도 임계값", 
    min_value=0.5, max_value=0.99, value=0.70, step=0.01,
    help="←(0.5): 관대한 채보 / →(0.99): 초정밀 검수. 음이 끊기거나 안 나온다면 왼쪽(←)으로 옮기세요!"
)

st.sidebar.subheader("3. Rake Detection (잡음 시각화)")
rake_sens = st.sidebar.slider(
    "잡음 감지 민감도",
    min_value=0.1, max_value=0.9, value=0.6, step=0.05,
    help="←(0.1): 잡음 차단 강화 / →(0.9): 모든 음 생존. 연주가 자꾸 잡음으로 처리된다면 오른쪽(→)으로 옮기세요!"
)

st.sidebar.subheader("4. Noise Gate (커팅)")
noise_gate_db = st.sidebar.slider(
    "게이트 임계값 (dB)", 
    min_value=-80, max_value=-10, value=-40, step=1,
    help="←(-80): 작은 소리도 살림 / →(-10): 강한 커팅. 소리가 작아서 인식이 안 되면 왼쪽(←)으로 옮기세요!"
)

st.sidebar.subheader("5. Sustain Buddy (연결)")
sustain_ms = st.sidebar.slider(
    "지속 연결 시간 (ms)",
    min_value=0, max_value=200, value=70, step=10,
    help="←(0): 모든 음 분절 / →(200): 부드러운 연결. 음이 뚝뚝 끊기면 오른쪽(→)으로 옮기세요!"
)

st.sidebar.subheader("6. MIDI Patch")
patch_num = st.sidebar.selectbox(
    "악기 소리 (Preview)",
    options=[27, 0, 29, 30],
    format_func=lambda x: {27: 'Electric Guitar (Clean)', 0: 'Acoustic Grand Piano', 29: 'Overdriven Guitar', 30: 'Distortion Guitar'}.get(x, str(x))
)

show_tabs = st.sidebar.checkbox("🎸 Generate Tablature & MusicXML", value=False)

# --- 📡 Input Source Selection ---
st.sidebar.markdown("---")
st.sidebar.subheader("📡 Input Source")
app_mode = st.sidebar.radio("Perception Input", ["📤 Manual Upload", "📂 Archive Explorer"])

input_audio_path = None
uploaded_file = None

if app_mode == "📂 Archive Explorer":
    home = os.path.expanduser("~")
    search_paths = [
        "downloads/이승철", 
        "artist_mega_collector/downloads/이승철",
        os.path.join(home, "Downloads/AEGIS_COLLECTION"),
        os.path.join(home, "Downloads/이승철_Music")
    ]
    
    available_files = []
    for p in search_paths:
        abs_p = os.path.abspath(p) if not p.startswith("/") else p
        if os.path.exists(abs_p):
            for f in os.listdir(abs_p):
                if f.endswith(".mp3"):
                    available_files.append({"name": f, "path": os.path.join(abs_p, f)})
    
    if available_files:
        selected_file_meta = st.sidebar.selectbox(
            "Target Selection", 
            available_files, 
            format_func=lambda x: f"🎵 {x['name'][:30]}..."
        )
        if selected_file_meta:
            input_audio_path = selected_file_meta['path']
            st.sidebar.success(f"Archived Target Locked.")
    else:
        st.sidebar.warning("No files found in archives.")

# File Uploader logic
if app_mode == "📤 Manual Upload":
    col_file1, col_file2 = st.columns(2)
    with col_file1:
        uploaded_file = st.file_uploader("1. 원본 음원 (전체 믹스)", type=["wav", "mp3"])
    
    logic_stem_path = None
    with col_file2:
        if external_stem_mode:
            logic_stem_file = st.file_uploader("2. 로직 분리 기타 스템 (Logic Stem)", type=["wav", "mp3"])
        else:
            logic_stem_file = None
else:
    logic_stem_path = None
    logic_stem_file = None

if (uploaded_file is not None) or (input_audio_path is not None):
    # Determine the actual path to work with
    if uploaded_file:
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_display_name = uploaded_file.name
        
        # Save Logic Stem if exists
        if logic_stem_file:
            logic_stem_path = os.path.join(temp_dir, "logic_" + logic_stem_file.name)
            with open(logic_stem_path, "wb") as f:
                f.write(logic_stem_file.getbuffer())
    else:
        file_path = input_audio_path
        file_display_name = os.path.basename(file_path)
    
    # --- CUSTOM AUDIO NAVIGATOR (WaveSurfer.js) ---
    st.subheader(f"⏱️ Integrated Timeline: {file_display_name}")
    
    # Read the audio file bits for the custom player
    import base64
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()

    # Custom HTML/JS Component
    custom_ui = """
    <div id="waveform" style="background: #0e1117; border-radius: 8px; border: 1px solid #333;"></div>
    <div id="wave-controls" style="margin-top: 10px; display: flex; gap: 10px; align-items: center; color: white; font-family: sans-serif;">
        <button id="playPause" style="background: #00ffcc; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; color: black; font-weight: bold;">Play/Pause</button>
        <span id="time-info">00:00 / 00:00</span>
        <span id="region-info" style="margin-left: auto; color: #00ffcc;">Selected: Entire File</span>
    </div>

    <script src="https://unpkg.com/wavesurfer.js@6.6.3/dist/wavesurfer.js"></script>
    <script src="https://unpkg.com/wavesurfer.js@6.6.3/dist/plugin/wavesurfer.regions.js"></script>
    
    <script>
        const wavesurfer = WaveSurfer.create({
            container: '#waveform',
            waveColor: '#444',
            progressColor: '#00ffcc',
            cursorColor: '#00ffcc',
            barWidth: 2,
            barRadius: 3,
            cursorWidth: 1,
            height: 120,
            plugins: [
                WaveSurfer.regions.create({
                    regionsMinLength: 0.1,
                    dragSelection: {
                        slop: 5
                    }
                })
            ]
        });

        wavesurfer.load('data:audio/wav;base64,{b64}');

        wavesurfer.on('ready', function () {
            const duration = wavesurfer.getDuration();
            document.getElementById('time-info').innerText = '00:00 / ' + formatTime(duration);
            
            // Initial region
            wavesurfer.addRegion({
                start: 0,
                end: duration,
                color: 'rgba(0, 255, 204, 0.2)',
                drag: true,
                resize: true
            });
        });

        wavesurfer.on('audioprocess', function () {
            document.getElementById('time-info').innerText = formatTime(wavesurfer.getCurrentTime()) + ' / ' + formatTime(wavesurfer.getDuration());
        });

        wavesurfer.on('region-updated', function(region) {
            document.getElementById('region-info').innerText = 'Selected: ' + region.start.toFixed(1) + 's - ' + region.end.toFixed(1) + 's';
        });

        document.getElementById('playPause').addEventListener('click', function() {
            wavesurfer.playPause();
        });

        function formatTime(s) {
            const min = Math.floor(s / 60);
            const sec = Math.floor(s % 60);
            return min.toString().padStart(2, '0') + ':' + sec.toString().padStart(2, '0');
        }
    </script>
    """.replace('{b64}', b64)
    
    from streamlit.components.v1 import html
    # Note: Using a standard trick to get value back from HTML component
    # For a truly integrated feel in this setup, we'll use a session state fallback
    # or just let the user know we're using the visible selection.
    
    st.components.v1.html(custom_ui, height=200)

    # --- 🏗️ Analysis Control Engine ---
    st.markdown("---")
    
    # Check Duration for Slider Stability
    try:
        duration_val = float(librosa.get_duration(path=file_path))
        if duration_val < 0.1: duration_val = 0.1 # Floor for step alignment
    except:
        duration_val = 1.0

    st.info("💡 위 플레이 바에서 영역을 드래그하여 선택하거나 아래 슬라이더로 확정하세요.")
    # Ensure value tuple is within bound and step-aligned
    start_time, end_time = st.slider(
        "최종 분석 구간 설정 (초)",
        min_value=0.0, 
        max_value=float(round(duration_val, 1)),
        value=(0.0, float(round(duration_val, 1))),
        step=0.1,
        key="analysis_slider"
    )

    # State Management for "Analyze Once, Filter Anytime"
    if 'last_analysis_key' not in st.session_state:
        st.session_state.last_analysis_key = None
    if 'raw_data_cache' not in st.session_state:
        st.session_state.raw_data_cache = None

    # Run Analysis Button
    if st.button("🚀 Aegis Perception 실행", type="primary", use_container_width=True) or st.session_state.raw_data_cache is not None:
        # Key includes all ARGS that affect the AI Perception output
        current_key = f"{file_path}_{start_time}_{end_time}_{full_band_mode}_{external_stem_mode}_{logic_stem_path}_{rake_sens}_{turbo_mode}"
        
        # Ensure temp_dir exists for stem extraction
        if 'temp_dir' not in locals():
            import tempfile
            temp_dir = tempfile.mkdtemp()
            
        with st.spinner("🛡️ Aegis Intelligence is perceiving the audio..."):
            engine = AegisEngine()
            
            # 0. Check if we need to rerun the HEAVY AI part
            if st.session_state.last_analysis_key != current_key:
                st.session_state.last_analysis_key = current_key
                
                # 0.1 Stem Separation (Only if key changed)
                analysis_file = file_path
                if external_stem_mode and logic_stem_path:
                    analysis_file = logic_stem_path
                    st.success("✅ 로직 프로 외부 스템 사용")
                elif full_band_mode:
                    analysis_file = engine.separate_stems(file_path, temp_dir)
                    st.info("기타 트랙 분리 완료!")
                
                # 0.2 HEAVY AI Perception
                st.session_state.raw_data_cache = engine.audio_to_midi(
                    analysis_file, 
                    output_mid=None, # In-memory processing
                    start_time=start_time,
                    end_time=end_time,
                    turbo_mode=turbo_mode,
                    rake_sensitivity=rake_sens
                )
            
            # 0.3 HEAVY Audio Visual Cache
            y_slice, S_dB_slice = engine.load_audio(analysis_file, start_time=start_time, end_time=end_time)
            st.session_state.audio_visual_cache = {
                'y_slice': y_slice,
                'S_dB_slice': S_dB_slice,
                'rake_mask': engine.detect_rake_patterns(S_dB_slice)
            }
            
            st.toast("AI Perception Complete (Heavy Work Done!)")
        
        @st.fragment
        def render_results(raw_data):
            # 1.0 Real-time Controls (Inside Fragment to avoid global blur)
            st.subheader("🛠️ Aegis Real-time Tuning")
            c_col1, c_col2, c_col3, c_col4 = st.columns(4)
            with c_col1:
                c_thresh = st.slider("🛡️ Confidence", 0.0, 1.0, 0.5, 0.01)
            with c_col2:
                s_ms = st.slider("⏳ Sustain (ms)", 0, 1000, 200, 50)
            with c_col3:
                m_ms = st.slider("📏 Min Dur (ms)", 10, 500, 50, 10)
            with c_col4:
                p_num = st.number_input("🎹 Patch", 0, 127, 27)

            # 1.1 LIGHT Logic Filtering
            midi_buffer = io.BytesIO()
            events = engine.extract_events(
                raw_data,
                midi_buffer,
                min_note_duration_ms=m_ms,
                confidence_threshold=c_thresh,
                midi_program=p_num,
                sustain_ms=s_ms
            )
            
            midi_buffer.seek(0)
            midi_data = midi_buffer.read()
            midi_base64 = base64.b64encode(midi_data).decode()
            
            st.success(f"Analysis Complete! (Mode: {'Real-time Filter' if st.session_state.last_analysis_key == current_key else 'Full Analysis'})")
            
            # Audio Visual Context (Static Cache)
            av_cache = st.session_state.audio_visual_cache
            y_slice, S_dB_slice = av_cache['y_slice'], av_cache['S_dB_slice']
            
            if not zen_mode:
                st.subheader("📊 1. Data Visualization (Spectrogram)")
                fig, ax = plt.subplots(figsize=(10, 4))
                librosa.display.specshow(S_dB_slice, x_axis='time', y_axis='mel', sr=engine.sr, ax=ax, fmax=8000)
                ax.set_title(f'Mel-frequency spectrogram ({start_time:.1f}s - {end_time:.1f}s)')
                st.pyplot(fig)
            else:
                st.info("💡 Zen Mode 활성화: 그래픽 출력을 제한하여 튜닝 반응속도를 최적화했습니다.")

            # 3. Results Download
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="💾 Cleaned MIDI 다운로드",
                    data=midi_data,
                    file_name=uploaded_file.name + "_cleaned.mid",
                    mime="audio/midi"
                )
            
            tab_data = None
            if show_tabs:
                with st.spinner("Generating Professional Tablature..."):
                    tab_data = engine.generate_tabs(events)
                with col2:
                    xml_path = "temp_tab.xml"
                    engine.export_musicxml(tab_data, xml_path)
                    with open(xml_path, "rb") as f:
                        st.download_button(label="🎸 MusicXML 다운로드", data=f, file_name=uploaded_file.name + ".xml")

            # 4. MIDI Raw Event Data (Optimized Display)
            st.subheader("📜 2. Real-time MIDI Event Log")
            if events:
                import pandas as pd
                df_view = pd.DataFrame(events)
                if not df_view.empty:
                    df_view['note_name'] = df_view['note'].apply(lambda x: librosa.midi_to_note(int(x)))
                    # Only show top 50 to prevent DOM lag
                    st.dataframe(df_view.head(50), use_container_width=True)
                    st.write(f"현재 활성 노트 수: {len(events)}개 (상위 50개 표시 중)")
                else:
                    st.warning("조건에 맞는 노트가 없습니다.")

            # 5. Aesthetic Vector Piano Roll (Modular Edition)
            if not zen_mode:
                st.subheader("🎹 Aegis Vector Piano Roll")
                # Unify the update trigger with a state-based key
                viz_key = f"vector_viz_{len(midi_base64)}_{c_thresh}_{s_ms}_{midi_program}"
                render_vector_piano_roll(midi_base64, viz_key, height=550)

            # 6. Optional TAB
            if show_tabs and tab_data:
                st.subheader("🎸 4. Aegis Professional Guitar TAB")
                chunk_size = 20
                for i in range(0, len(tab_data), chunk_size):
                    chunk = tab_data[i:i+chunk_size]
                    strings = [f"{n}|" for n in ["e", "B", "G", "D", "A", "E"]]
                    for t_note in chunk:
                        s_idx = t_note['string'] - 1
                        fret_str = str(t_note['fret'])
                        # Add Technique Symbol
                        tech = t_note.get('technique')
                        sym = ""
                        if tech == 'bend': sym = "b"
                        elif tech == 'slide': sym = "/"
                        elif tech == 'vibrato': sym = "~"
                        
                        full_fret = f"{fret_str}{sym}"
                        pad = len(full_fret) + 2
                        
                        for idx in range(6):
                            if idx == s_idx: strings[idx] += f"-{full_fret}-"
                            else: strings[idx] += "-" * pad
                    st.code("\n".join(strings), language="text")

            # 7. Rake Filter Report
            if not zen_mode:
                st.subheader("🛡️ 5. Guardian Filter Report")
                rake_mask = av_cache['rake_mask']
                st.metric("감지된 Rake(잡음) 비율", f"{np.sum(rake_mask)/len(rake_mask)*100:.2f}%")
                fig2, ax2 = plt.subplots(figsize=(10, 2))
                times = np.arange(len(rake_mask)) * engine.hop_length / engine.sr
                ax2.fill_between(times, 0, 1, where=rake_mask, color='red', alpha=0.5, transform=ax2.get_xaxis_transform())
                ax2.set_yticks([]); ax2.set_xlim([0, times[-1]])
                st.pyplot(fig2)

        # Execute Fragment
        render_results(st.session_state.raw_data_cache)
        st.session_state.last_analysis_key = current_key

else:
    st.info("좌측 패널에서 파라미터를 조정한 후 오디오 파일을 업로드하세요.")
