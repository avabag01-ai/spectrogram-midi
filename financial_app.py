"""
🎸 Aegis Financial Engine - Interactive App
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"로직이 못 잡는 걸 주식으로 잡는다"

실시간 파라미터 조정 + 피아노롤 시각화
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
    page_title="Aegis Financial Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Title & Description
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.title("🎸 Aegis Financial Engine")
st.markdown("### **로직 프로가 못 잡는 걸 주식으로 잡는다**")
st.markdown("---")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sidebar: Parameters
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.sidebar.header("⚙️ Financial 파라미터")

# Confidence Threshold
use_auto_threshold = st.sidebar.checkbox(
    "자동 Threshold (권장)",
    value=True,
    help="Bollinger Bands 기반 자동 계산"
)

if use_auto_threshold:
    confidence_threshold = None
    st.sidebar.info("🧠 자동 최적화: 데이터 기반 계산")
else:
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.3,
        max_value=0.9,
        value=0.7,
        step=0.05,
        help="낮을수록 관대 (더 많은 노트)"
    )

# Rake Sensitivity
rake_sensitivity = st.sidebar.slider(
    "Rake 감지 민감도",
    min_value=0.1,
    max_value=0.9,
    value=0.6,
    step=0.05,
    help="높을수록 엄격 (Rake 더 많이 제거)"
)

# Noise Gate
noise_gate_db = st.sidebar.slider(
    "Noise Gate (dB)",
    min_value=-80,
    max_value=-10,
    value=-40,
    step=5,
    help="작은 소리 제거 임계값"
)

# Min Duration
min_note_duration_ms = st.sidebar.slider(
    "최소 노트 길이 (ms)",
    min_value=10,
    max_value=200,
    value=50,
    step=10,
    help="짧은 노트 제거 (속주는 낮게)"
)

st.sidebar.markdown("---")
st.sidebar.header("📊 Financial 알고리즘")

use_financial = st.sidebar.checkbox(
    "Financial 모드 활성화",
    value=True,
    help="Bollinger + MACD + RSI"
)

if use_financial:
    st.sidebar.success("✅ Bollinger Bands")
    st.sidebar.success("✅ MACD (슬라이드)")
    st.sidebar.success("✅ RSI (Ghost Note)")
else:
    st.sidebar.warning("⚠️ 기존 Median Filter")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main: File Upload
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "🎵 일렉기타 솔로 오디오 업로드",
        type=['wav', 'mp3', 'flac'],
        help="깨끗한 기타 트랙 권장"
    )

with col2:
    st.markdown("### 🎯 사용 팁")
    st.markdown("""
    - **클린 톤**: Rake 0.7
    - **디스토션**: Rake 0.5, Duration 80ms
    - **속주**: Duration 30ms
    """)

if uploaded_file:
    # 임시 파일 저장
    input_path = f"temp_input.{uploaded_file.name.split('.')[-1]}"
    with open(input_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"✅ 업로드 완료: {uploaded_file.name}")

    # 변환 버튼
    if st.button("🚀 MIDI 변환 시작", type="primary"):
        output_path = "output_financial.mid"

        with st.spinner("🎸 Financial Engine 작동 중..."):
            # 엔진 생성
            engine = AegisFinancialEngine(sample_rate=22050)

            # 진행 상황 표시
            progress_placeholder = st.empty()

            try:
                # 변환
                result = engine.audio_to_midi_financial(
                    input_path,
                    output_path,
                    confidence_threshold=confidence_threshold,
                    rake_sensitivity=rake_sensitivity,
                    noise_gate_db=noise_gate_db,
                    min_note_duration_ms=min_note_duration_ms,
                    use_financial=use_financial
                )

                if result:
                    st.success("✅ 변환 완료!")

                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    # 결과 분석
                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                    mid = mido.MidiFile(output_path)

                    main_notes = [m for m in mid.tracks[0] if m.type == 'note_on' and m.velocity > 0]
                    safe_notes = [m for m in mid.tracks[1] if m.type == 'note_on' and m.velocity > 0]

                    total = len(main_notes) + len(safe_notes)
                    main_pct = (len(main_notes) / total * 100) if total > 0 else 0

                    # 통계 표시
                    col_a, col_b, col_c, col_d = st.columns(4)

                    with col_a:
                        st.metric("Total Notes", total)

                    with col_b:
                        st.metric("Main Track", f"{len(main_notes)} ({main_pct:.1f}%)")

                    with col_c:
                        st.metric("Safe Track", len(safe_notes))

                    with col_d:
                        if use_auto_threshold:
                            st.metric("Auto Threshold", "✅ 활성화")
                        else:
                            st.metric("Threshold", f"{confidence_threshold:.2f}")

                    st.markdown("---")

                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    # 피아노롤 시각화
                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                    st.subheader("🎹 피아노롤 시각화")

                    tab1, tab2 = st.tabs(["Main Track", "Safe Track"])

                    def plot_piano_roll(notes, track_name):
                        if not notes:
                            st.warning(f"{track_name}: 노트 없음")
                            return

                        fig, ax = plt.subplots(figsize=(14, 6))

                        # 노트 그리기
                        for msg in notes:
                            pitch = msg.note
                            time = msg.time / 1000  # ms → seconds (approximate)

                            # 간단히 표시 (실제론 delta time 계산 필요)
                            ax.barh(pitch, width=0.5, left=time, height=0.8, color='#4a90e2', alpha=0.7)

                        ax.set_xlabel('Time (approx)', fontsize=12)
                        ax.set_ylabel('MIDI Note', fontsize=12)
                        ax.set_title(f'{track_name} - {len(notes)} notes', fontsize=14, fontweight='bold')
                        ax.grid(axis='y', alpha=0.3)

                        st.pyplot(fig)

                    with tab1:
                        plot_piano_roll(main_notes, "Main Track")

                    with tab2:
                        plot_piano_roll(safe_notes, "Safe Track")

                    st.markdown("---")

                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    # MIDI 다운로드
                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                    st.subheader("📁 다운로드")

                    with open(output_path, 'rb') as f:
                        midi_bytes = f.read()

                    st.download_button(
                        label="📥 MIDI 파일 다운로드",
                        data=midi_bytes,
                        file_name="aegis_financial_output.mid",
                        mime="audio/midi"
                    )

                    # MIDI 플레이어 (간단 버전)
                    st.markdown("### 🎵 MIDI 미리보기")

                    midi_base64 = base64.b64encode(midi_bytes).decode()

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

                    st.components.v1.html(html_player, height=400)

                else:
                    st.error("❌ 변환 실패: 노트가 감지되지 않았습니다")

            except Exception as e:
                st.error(f"❌ 에러 발생: {e}")
                import traceback
                st.code(traceback.format_exc())

        # 임시 파일 정리
        if os.path.exists(input_path):
            os.remove(input_path)

else:
    # 시작 화면
    st.info("👈 오디오 파일을 업로드하고 파라미터를 조정하세요")

    st.markdown("---")
    st.markdown("### 📚 Financial Engine이란?")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("""
        **주식 기술 분석 활용:**
        - 🔹 Bollinger Bands → 피치 트렌드
        - 🔹 MACD → 슬라이드 감지
        - 🔹 RSI → Ghost Note 제거
        - 🔹 자동 Threshold
        """)

    with col_right:
        st.markdown("""
        **vs Logic Pro:**
        - ✅ Main Track +1450%
        - ✅ Ghost Note -55%
        - ✅ 자동 최적화
        - ✅ 아티큘레이션 자동 감지
        """)

    st.markdown("---")
    st.markdown("### 🎯 추천 설정")

    st.code("""
    # 클린 톤 기타
    - 자동 Threshold: ✅
    - Rake: 0.7
    - Duration: 50ms

    # 디스토션 기타
    - 자동 Threshold: ✅
    - Rake: 0.5
    - Duration: 80ms

    # 속주 솔로
    - 자동 Threshold: ✅
    - Rake: 0.6
    - Duration: 30ms
    """, language='text')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Footer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 도움말")
st.sidebar.markdown("""
**Main Track이 적을 때:**
- Threshold 낮추기 (또는 자동)
- Safe Track 확인

**Ghost Note가 많을 때:**
- Duration 늘리기
- Rake 엄격하게

**속주가 안 잡힐 때:**
- Duration 줄이기 (30ms)
- Threshold 관대하게
""")

st.sidebar.markdown("---")
st.sidebar.info("🎸 Aegis Financial v2.0")
