import streamlit as st
import os
import subprocess
import base64

st.set_page_config(page_title="Demucs Stem Lab", layout="wide")

st.title("🧪 Demucs Stem Lab")
st.markdown("Meta(Facebook)의 오픈소스 AI **Demucs**를 이용한 스템 분리 전용 테스트 앱입니다.")

# Configuration
DEMUCS_PATH = "/Users/mac/Library/Python/3.9/bin/demucs"
OUTPUT_DIR = "stem_lab_output"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

st.sidebar.header("⚙️ Settings")
model_name = st.sidebar.selectbox("Model Selection", ["htdemucs", "htdemucs_ft", "mdx_extra"])
st.sidebar.info("htdemucs_ft가 가장 정밀하지만 시간이 더 오래 걸립니다.")

uploaded_file = st.file_uploader("음원 파일을 업로드하세요 (mp3, wav)", type=["mp3", "wav"])

if uploaded_file:
    # Save uploaded file
    input_path = os.path.join(OUTPUT_DIR, uploaded_file.name)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.subheader("Original Audio")
    st.audio(input_path)

    if st.button("🚀 Start Separation"):
        with st.spinner(f"AI가 {model_name} 모델로 분리 중입니다... (1~3분 소요)"):
            cmd = [DEMUCS_PATH, "-n", model_name, "-o", OUTPUT_DIR, input_path]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                st.success("분리 완료!")
            except Exception as e:
                st.error(f"Error: {e}")
                st.code(str(e))

        # Path logic
        folder_name = uploaded_file.name.split('.')[0]
        stems_dir = os.path.join(OUTPUT_DIR, model_name, folder_name)

        if os.path.exists(stems_dir):
            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)
            
            stems = {
                "🎸 Other (Guitar/Synth)": "other.wav",
                "🎤 Vocals": "vocals.wav",
                "🥁 Drums": "drums.wav",
                "🎸 Bass": "bass.wav"
            }
            
            for i, (name, filename) in enumerate(stems.items()):
                target_col = [col1, col2, col3, col4][i]
                path = os.path.join(stems_dir, filename)
                if os.path.exists(path):
                    target_col.markdown(f"### {name}")
                    target_col.audio(path)
                    with open(path, "rb") as f:
                        target_col.download_button(f"Download {name}", f, file_name=f"{folder_name}_{filename}")
        else:
            st.error("분리된 파일을 찾을 수 없습니다.")
            st.write(f"Searched in: {stems_dir}")
