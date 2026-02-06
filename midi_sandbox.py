import streamlit as st
import base64
import os
from aegis_engine_core.visualizers import render_vector_piano_roll

# --- 🛰️ AEGIS MULTI-ENGINE SANDBOX v2.0 ---
st.set_page_config(page_title="Multi-Engine Lab", layout="wide")

st.markdown("""
<style>
    .main { background-color: #000000; }
    .stSelectbox label, .stRadio label { color: #ff00cc !important; font-weight: bold; }
    h1 { color: #ff00cc; font-family: 'Courier New', monospace; border-bottom: 2px solid #ff00cc; }
</style>
""", unsafe_allow_html=True)

st.title("🧪 AEGIS MULTI-ENGINE LAB")

# 1. Engine Selection (The Architect's Choice)
st.sidebar.header("🛠️ ENGINE ARCHITECTURE")
engine_choice = st.sidebar.radio(
    "Select Rendering Engine", 
    ["Aegis Python (Pure Local)", "Tone.js (SVG Components)", "PianoRoll.js Style (Canvas)", "WebAudioFont (Vector Path)"],
    index=0 # Default to Pure Local Python for 100% reliability
)

engine_map = {
    "Aegis Python (Pure Local)": "python",
    "Tone.js (SVG Components)": "tonejs",
    "PianoRoll.js Style (Canvas)": "canvas",
    "WebAudioFont (Vector Path)": "webaudiofont"
}

# 2. Theme Selection
st.sidebar.markdown("---")
st.sidebar.header("🎨 AESTHETICS")
theme_choice = st.sidebar.select_slider("System Theme", options=["Charcoal", "Beige"], value="Beige")
theme_map = {"Charcoal": "charcoal", "Beige": "beige"}

st.sidebar.markdown("---")
st.sidebar.header("📥 MIDI INPUT")
src_mode = st.sidebar.segmented_control("Select Mode", ["Downloads Folder", "Manual Upload"], default="Downloads Folder")

midi_data_b64 = None
target_name = ""

if src_mode == "Downloads Folder":
    home = os.path.expanduser("~")
    dl_path = os.path.join(home, "Downloads")
    midis = [f for f in os.listdir(dl_path) if f.lower().endswith(('.mid', '.midi'))] if os.path.exists(dl_path) else []
    
    if midis:
        selected = st.sidebar.selectbox("Choose MIDI", midis)
        if selected:
            target_name = selected
            with open(os.path.join(dl_path, selected), "rb") as f:
                midi_data_b64 = base64.b64encode(f.read()).decode()
    else:
        st.sidebar.error("No MIDI in Downloads.")
else:
    uploaded = st.sidebar.file_uploader("Drop .mid here", type=["mid", "midi"])
    if uploaded:
        target_name = uploaded.name
        midi_data_b64 = base64.b64encode(uploaded.read()).decode()

# 2. Rendering Phase
if midi_data_b64:
    st.subheader(f"📡 Engine: {engine_choice} | Target: {target_name}")
    
    col1, col2 = st.columns([4, 1])
    with col2:
        viz_height = st.slider("Height (px)", 300, 1000, 600)
        if engine_choice == "PianoRoll.js Style (Canvas)":
            st.warning("Raster rendering: Highest performance.")
        elif engine_choice == "Tone.js (SVG Components)":
            st.info("Vector components: Interactive SVG.")
        else:
            st.success("SVG Path: Custom synthesis.")

    with col1:
        # Pass the selected engine and theme to the visualizer
        render_vector_piano_roll(
            midi_data_base64=midi_data_b64, 
            update_key=f"lab_{engine_map[engine_choice]}_{len(midi_data_b64)}", 
            height=viz_height,
            engine=engine_map[engine_choice],
            theme=theme_map[theme_choice]
        )
else:
    st.info("왼쪽 사이드바에서 MIDI 파일과 엔진을 선택하십시오.")
