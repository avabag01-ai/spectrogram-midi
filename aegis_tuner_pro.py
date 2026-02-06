import streamlit as st
import os
import base64
import numpy as np
import librosa
import io
import pandas as pd
import tempfile
from aegis_engine import AegisEngine
from aegis_engine_core.visualizers import render_vector_piano_roll
from aegis_engine_core.synthesizer import synthesize_midi, get_synthesizer, synthesize_midi_adsr, GUITAR_ADSR_PRESETS

# --- 🏛️ Aegis Tuner Pro: Ultra-Minimal Real-time Edition ---
st.set_page_config(page_title="Aegis Tuner Pro", layout="wide")

# 1. Core Engine Setup
@st.cache_resource
def get_engine():
    return AegisEngine()

engine = get_engine()

# 2. Sidebar Controls
st.sidebar.title("🛡️ Aegis Tuner Pro")

# --- 📂 Local File Library ---
st.sidebar.subheader("📂 Audio Library")
local_files = [f for f in os.listdir(".") if f.endswith((".wav", ".mp3"))]
default_idx = 0
if "untitled.mp3" in local_files:
    default_idx = local_files.index("untitled.mp3") + 1

selected_local = st.sidebar.selectbox("Select Audio Source", ["None"] + local_files, index=default_idx)
uploaded_file = st.sidebar.file_uploader("Or upload new", type=["wav", "mp3"])

# Determine final source
active_file_path = None
active_file_name = None

if uploaded_file:
    active_file_name = uploaded_file.name
    # Save to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(active_file_name)[1]) as tmp:
        tmp.write(uploaded_file.getvalue())
        active_file_path = tmp.name
elif selected_local != "None":
    active_file_name = selected_local
    active_file_path = os.path.abspath(selected_local)

# --- ⚙️ Tuning Parameters (Inside or Outside?) ---
# To avoid full page reload "gray out", we can use a fragment for the PARAMETERS too, 
# but usually sidebar is global. Let's keep them in a specific section.

if active_file_path:
    # 원본 파일 경로를 session_state에 저장 (fragment 내부에서 접근하기 위해)
    st.session_state.active_file_path = active_file_path
    st.session_state.active_file_name = active_file_name

    # 1. Analyze Once Stage (Global Cache)
    if 'raw_data_cache' not in st.session_state or st.session_state.get('last_file') != active_file_name:
        with st.status(f"🛠️ AI perception Analyzing: {active_file_name} (First 30s)..."):
            # Limit analysis to 30 seconds for speed + Turbo Mode
            raw_data = engine.audio_to_midi(active_file_path, "dummy.mid", end_time=30, turbo_mode=True)
            st.session_state.raw_data_cache = raw_data
            st.session_state.last_file = active_file_name
            st.toast("Ready for Real-time Tuning!")

    # 2. TUNING CORE (Fragmentation for Zero-Lag)
    @st.fragment
    def tuner_core(raw_data):
        # UI Layout: Control Bar (Left) | Data & Viz (Right)
        col_ctrl, col_res = st.columns([1, 4])
        
        with col_ctrl:
            st.subheader("🎚️ Aegis Tuning Bars")
            c_thresh = st.slider("🛡️ Guardian (Conf)", 0.0, 1.0,
                st.session_state.get('auto_conf', 0.4), 0.01)
            s_ms = st.slider("⏳ Sustain (ms)", 0, 1000,
                st.session_state.get('auto_sustain', 300), 50)
            m_ms = st.slider("📏 Min Dur (ms)", 10, 500,
                st.session_state.get('auto_mindur', 100), 10)

            # 기타 프리셋 선택
            guitar_presets = {
                "🎸 Nylon Guitar": 24,
                "🎸 Steel Guitar": 25,
                "🎸 Jazz Guitar": 26,
                "🎸 Clean Electric": 27,
                "🎸 Muted Guitar": 28,
                "🔥 Overdrive": 29,
                "🔥 Distortion": 30,
                "🎹 Custom (숫자 입력)": -1
            }
            preset_name = st.selectbox("🎸 Guitar Preset", list(guitar_presets.keys()), index=3)
            if guitar_presets[preset_name] == -1:
                p_val = st.number_input("Custom MIDI Patch", 0, 127, 27)
            else:
                p_val = guitar_presets[preset_name]

            # 비브라토 파라미터
            st.markdown("### 🎸 Vibrato Effect")
            v_rate = st.slider("🎸 Vibrato Rate (Hz)", 1.0, 10.0, 5.0, 0.5)
            v_depth = st.slider("🎸 Vibrato Depth", 0.0, 1.0, 0.3, 0.05)

            # 테크닉 검증 옵션
            st.markdown("### 🔬 Technique Verification")
            enable_verification = st.checkbox("🎯 오디오 패턴 매칭 검증", value=False,
                help="벤딩/해머링/풀오프를 오디오 패턴 매칭으로 검증 (느림)")

            st.markdown("---")
            st.caption("슬라이더 조작 시 화면이 흐려지지 않습니다.")

            # 자동 파라미터 매칭 버튼
            st.markdown("---")
            if st.button("🤖 Auto Match", use_container_width=True):
                with st.spinner("🔍 최적 파라미터 탐색 중..."):
                    from aegis_engine_core.auto_matcher import auto_match_parameters
                    best = auto_match_parameters(
                        st.session_state.active_file_path,
                        engine,
                        raw_data
                    )
                    if best:
                        st.session_state.auto_conf = best['confidence_threshold']
                        st.session_state.auto_sustain = best['sustain_ms']
                        st.session_state.auto_mindur = best['min_note_duration_ms']
                        st.success(f"✅ 최적값 찾음! Score: {best['score']:.3f}")
                        st.rerun()
                    else:
                        st.error("❌ 최적값을 찾지 못했습니다.")

        # Real-time Logic Filtering
        midi_buffer = io.BytesIO()
        events = engine.extract_events(
            raw_data,
            midi_buffer,
            min_note_duration_ms=m_ms,
            confidence_threshold=c_thresh,
            midi_program=p_val,
            sustain_ms=s_ms,
            vibrato_rate=v_rate,
            vibrato_depth=v_depth
        )

        # 테크닉 검증 (선택적)
        if enable_verification and events:
            with st.spinner("🔬 테크닉 검증 중..."):
                from aegis_engine_core.technique_verifier import verify_technique_by_audio_matching
                synth = get_synthesizer()
                if synth.is_available():
                    events = verify_technique_by_audio_matching(
                        events, raw_data, engine, synth,
                        engine.sr, engine.hop_length
                    )
                    st.toast("✅ 테크닉 검증 완료!")
                else:
                    st.warning("FluidSynth 없음. 검증 스킵.")
        
        midi_buffer.seek(0)
        midi_data = midi_buffer.read()
        midi_base64 = base64.b64encode(midi_data).decode()
        update_key = f"viz_{len(midi_data)}_{c_thresh}_{s_ms}" # unique key to force refresh

        with col_res:
            inner_log, inner_viz = st.columns([1, 2])

            with inner_log:
                st.subheader("📜 Live Event Log")
                if events:
                    df = pd.DataFrame(events)
                    df['note_name'] = df['note'].apply(lambda x: librosa.midi_to_note(int(x)))
                    # technique 컬럼 추가
                    df['technique'] = df.get('technique', pd.Series([None] * len(df))).fillna('-')
                    # slope 값도 표시 (벤딩용)
                    if 'slope' in df.columns:
                        df['slope'] = df['slope'].apply(lambda x: f"{x:.3f}" if x != 0 else '-')
                    else:
                        df['slope'] = '-'
                    st.dataframe(df[['note_name', 'velocity', 'confidence', 'technique', 'slope']].head(50),
                                 use_container_width=True, height=450)
                    st.write(f"Active Notes: {len(events)}")
                else:
                    st.warning("Empty filter results.")

            with inner_viz:
                st.subheader("🎹 Aegis Live Piano Roll")
                render_vector_piano_roll(midi_base64, height=500, engine="python", theme="beige")

                # 오디오 플레이어 섹션
                st.markdown("---")
                st.subheader("🎧 Audio Comparison")
                audio_col1, audio_col2 = st.columns(2)

                with audio_col1:
                    st.markdown("**원본 음원**")
                    if 'active_file_path' in st.session_state:
                        try:
                            st.audio(st.session_state.active_file_path)
                        except Exception as e:
                            st.error(f"원본 재생 실패: {e}")
                    else:
                        st.info("원본 파일 없음")

                with audio_col2:
                    st.markdown("**MIDI 합성 (FluidSynth)**")
                    # FluidSynth로 MIDI → WAV 변환
                    synth = get_synthesizer()
                    if synth.is_available():
                        try:
                            with st.spinner("🎹 합성 중..."):
                                wav_data = synthesize_midi(midi_data, sample_rate=44100)
                            if wav_data:
                                st.audio(wav_data, format="audio/wav")
                            else:
                                st.error("MIDI 합성 실패")
                        except Exception as e:
                            st.error(f"합성 오류: {e}")
                    else:
                        st.warning("FluidSynth를 찾을 수 없습니다")
                        st.caption("설치: `brew install fluid-synth`")

                st.markdown("---")
                st.download_button(f"💾 Download {st.session_state.get('active_file_name', 'output')}.mid",
                                 data=midi_data,
                                 file_name=f"aegis_{st.session_state.get('active_file_name', 'output')}.mid")

                # 역변환 분석 섹션
                st.markdown("---")
                st.subheader("🔄 역변환 분석")
                st.caption("MIDI → 합성 음원 → 다시 MIDI로 변환하여 정확도 측정")

                if st.button("🔬 역변환 분석 실행", use_container_width=True):
                    with st.spinner("🔄 분석 중..."):
                        from aegis_engine_core.reverse_analyzer import reverse_analysis
                        result = reverse_analysis(midi_data, engine)

                        if result:
                            st.success("✅ 역변환 분석 완료!")

                            # 메트릭 표시
                            m1, m2, m3 = st.columns(3)
                            m1.metric("원본 노트", result['original_notes'])
                            m2.metric("역변환 노트", result['reversed_notes'])
                            m3.metric("노트 일치율", f"{result['note_accuracy']:.1%}")

                            # 추가 정확도 지표
                            acc1, acc2 = st.columns(2)
                            acc1.metric("피치 정확도", f"{result['pitch_accuracy']:.1%}")
                            acc2.metric("타이밍 정확도", f"{result['timing_accuracy']:.1%}")

                            # 역변환 MIDI 다운로드
                            st.download_button(
                                "💾 역변환 MIDI 다운로드",
                                data=result['reversed_midi'],
                                file_name=f"reversed_{st.session_state.get('active_file_name', 'output')}.mid",
                                use_container_width=True
                            )
                        else:
                            st.error("❌ 역변환 분석 실패")

                # === 🎛️ ADSR 소프트 신스 섹션 ===
                st.markdown("---")
                st.subheader("🎛️ ADSR Soft Synth")
                st.caption("직접 파형/엔벨로프를 조절해서 원본 음색에 가깝게 합성")

                adsr_col1, adsr_col2 = st.columns(2)
                with adsr_col1:
                    adsr_preset = st.selectbox("🎸 ADSR 기타 프리셋",
                        list(GUITAR_ADSR_PRESETS.keys()), index=2)
                    use_envelope_match = st.checkbox("🔍 원본 엔벨로프 자동 분석", value=False,
                        help="원본 음원의 ADSR 특성을 분석해서 자동 적용")

                with adsr_col2:
                    preset_info = GUITAR_ADSR_PRESETS[adsr_preset]
                    st.caption(f"Attack: {preset_info['attack_ms']}ms | Decay: {preset_info['decay_ms']}ms | "
                              f"Sustain: {preset_info['sustain_level']} | Release: {preset_info['release_ms']}ms | "
                              f"Wave: {preset_info['waveform']}")

                if st.button("🎹 ADSR 합성", use_container_width=True):
                    with st.spinner("🎛️ ADSR 소프트 신스 합성 중..."):
                        adsr_overrides = {}
                        if use_envelope_match and 'active_file_path' in st.session_state:
                            try:
                                from aegis_engine_core.synthesizer import get_adsr_synthesizer
                                adsr_synth = get_adsr_synthesizer()
                                y_orig, _ = librosa.load(st.session_state.active_file_path, sr=44100, duration=10)
                                env_params = adsr_synth.analyze_envelope(y_orig, sr=44100)
                                adsr_overrides = env_params
                                st.info(f"🔍 분석 결과: A={env_params['attack_ms']:.0f}ms D={env_params['decay_ms']:.0f}ms "
                                       f"S={env_params['sustain_level']:.2f} R={env_params['release_ms']:.0f}ms")
                            except Exception as e:
                                st.warning(f"엔벨로프 분석 실패: {e}")

                        adsr_wav = synthesize_midi_adsr(midi_data, preset=adsr_preset, **adsr_overrides)
                        if adsr_wav:
                            st.audio(adsr_wav, format="audio/wav")
                            st.success("✅ ADSR 합성 완료!")
                        else:
                            st.error("❌ ADSR 합성 실패")

                # === 🔄 이펙트 학습 루프 섹션 ===
                st.markdown("---")
                st.subheader("🧠 Effect Learning Loop")
                st.caption("MIDI → 이펙트 음원 → 재분석 → 파라미터 자동 최적화 반복")

                loop_col1, loop_col2 = st.columns(2)
                with loop_col1:
                    from aegis_engine_core.effect_learning_loop import EFFECT_PRESETS
                    effect_preset = st.selectbox("🎸 이펙트 프리셋",
                        list(EFFECT_PRESETS.keys()), index=0)
                with loop_col2:
                    max_iters = st.slider("🔄 최대 반복 횟수", 1, 10, 5)

                if st.button("🧠 학습 루프 시작", use_container_width=True):
                    with st.spinner("🧠 학습 루프 실행 중... (시간이 좀 걸립니다)"):
                        from aegis_engine_core.effect_learning_loop import learning_loop, EFFECT_PRESETS
                        loop_result = learning_loop(
                            midi_data=midi_data,
                            engine=engine,
                            effects_config=EFFECT_PRESETS[effect_preset],
                            max_iterations=max_iters,
                            target_accuracy=0.95
                        )

                        if loop_result:
                            st.success(f"✅ 학습 완료! Overall: {loop_result['best_accuracy']['overall']:.1%}")

                            # 결과 메트릭
                            lm1, lm2, lm3, lm4 = st.columns(4)
                            lm1.metric("노트 일치", f"{loop_result['best_accuracy']['note_accuracy']:.1%}")
                            lm2.metric("피치 정확도", f"{loop_result['best_accuracy']['pitch_accuracy']:.1%}")
                            lm3.metric("타이밍 정확도", f"{loop_result['best_accuracy']['timing_accuracy']:.1%}")
                            lm4.metric("종합", f"{loop_result['best_accuracy']['overall']:.1%}")

                            # 최적 파라미터 표시
                            bp = loop_result['best_params']
                            st.info(f"🎯 최적 파라미터: Conf={bp['confidence_threshold']:.2f} | "
                                   f"MinDur={bp['min_note_duration_ms']}ms | Sustain={bp['sustain_ms']}ms")

                            # 학습 히스토리 차트
                            if loop_result['history']:
                                hist_df = pd.DataFrame([
                                    {'iteration': h['iteration'],
                                     'overall': h['accuracy']['overall']}
                                    for h in loop_result['history']
                                ])
                                st.line_chart(hist_df.set_index('iteration'))

                            # 최적값 적용 버튼
                            if st.button("📥 최적 파라미터 적용", use_container_width=True):
                                st.session_state.auto_conf = bp['confidence_threshold']
                                st.session_state.auto_sustain = bp['sustain_ms']
                                st.session_state.auto_mindur = bp['min_note_duration_ms']
                                st.rerun()
                        else:
                            st.error("❌ 학습 루프 실패")

    # Run the ultra-stable loop
    tuner_core(st.session_state.raw_data_cache)

else:
    st.title("⚓ Aegis Tuner Pro")
    st.info("👈 Select a file from the library or upload to start.")
