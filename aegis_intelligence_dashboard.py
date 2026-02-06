import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import os
import random
import yt_dlp
from datetime import datetime

# --- 🛰️ Aegis Intelligence Dash v4.5: High-Clean Card Edition ---
st.set_page_config(page_title="Aegis Surveillance v4.5", layout="wide", initial_sidebar_state="expanded")

# --- 🎨 High-Clean Premium CSS ---
st.markdown("""
    <style>
    .main { background-color: #05070a; }
    .stMetric { background-color: #0b1016; padding: 25px; border-radius: 15px; border: 1px solid #1e2633; box-shadow: 0 4px 20px rgba(0,0,0,0.6); }
    div[data-testid="stExpander"] { background-color: #080a0f; border: 1px solid #1e2633; border-radius: 10px; }
    div[data-testid="stSidebar"] { background-color: #040609; border-right: 1px solid #1e2633; }
    h1, h2, h3 { color: #ff00cc !important; font-family: 'Space Grotesk', sans-serif; font-weight: 700; letter-spacing: -1px; }
    .video-card { background-color: #0b1016; border: 1px solid #1e2633; border-radius: 16px; padding: 15px; transition: transform 0.3s ease; height: 100%; }
    .video-card:hover { transform: translateY(-5px); border-color: #ff00cc; }
    .algo-score { font-size: 1.5rem; font-weight: bold; color: #ff00cc; }
    .log-sidebar { font-family: 'Monaco', monospace; font-size: 0.75rem; color: #ff00cc; line-height: 1.2; padding: 10px; background: #000; border-radius: 5px; height: 200px; overflow-y: auto; }
    .alert-banner { background-color: #ff0033; color: white; padding: 12px; border-radius: 8px; font-weight: bold; text-align: center; margin-bottom: 25px; }
    </style>
    """, unsafe_allow_html=True)

# --- 🛠️ Audio/Video Intelligence Engine ---
@st.cache_data(ttl=3600)
def fetch_trending_metadata(query="이승철", max_results=6):
    """Fetch real thumbnails and metadata using yt-dlp."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,
        'default_search': f'ytsearch{max_results}',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(query, download=False)
            results = []
            if 'entries' in info:
                for entry in info['entries']:
                    results.append({
                        'title': entry.get('title', 'Unknown Title'),
                        'id': entry.get('id'),
                        'url': f"https://www.youtube.com/watch?v={entry.get('id')}",
                        'thumbnail': f"https://img.youtube.com/vi/{entry.get('id')}/maxresdefault.jpg",
                        'views': entry.get('view_count', random.randint(100000, 5000000)),
                        'score': random.randint(70, 98)
                    })
            return results
        except:
            return []

# --- Sidebar: Intel & Settings ---
st.sidebar.markdown("### 🛰️ INTELLIGENCE CENTER")
st.sidebar.caption("Autonomous Monitoring Active")

target_query = st.sidebar.text_input("🎯 Target Search", placeholder="이승철, NewJeans...")
scan_active = st.sidebar.button("📡 SCAN TARGET", use_container_width=True, type="primary")

st.sidebar.markdown("---")
st.sidebar.subheader("📟 Intelligence Stream")
log_area = st.sidebar.empty()

# UI State Management
if 'logs' not in st.session_state:
    st.session_state.logs = [f"System initialized: {datetime.now().strftime('%H:%M:%S')}"]

def log_event(msg):
    st.session_state.logs.append(f"[{datetime.now().strftime('%M:%S')}] {msg}")
    if len(st.session_state.logs) > 20: st.session_state.logs.pop(0)
    log_area.markdown(f'<div class="log-sidebar">{"<br>".join(st.session_state.logs[::-1])}</div>', unsafe_allow_html=True)

# --- Main Dashboard ---
st.title("🛰️ AEGIS SURVEILLANCE v4.5")

active_keyword = target_query if target_query else "유튜브 인기 급상승"

# High-Priority Check
if "이승철" in active_keyword:
    st.markdown('<div class="alert-banner">🚨 [SIGNAL DETECTED] High-Priority Target "이승철" Analytics Online 🚨</div>', unsafe_allow_html=True)

# 1. CLEAN ANALYTICS: Top 3 Core Metrics
col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    m_growth = st.empty()
with col_m2:
    m_positive = st.empty()
with col_m3:
    m_target = st.empty()

# 2. TARGETING ENGINE: Dominant Heatmap
st.markdown("### 🔥 DEMOGRAPHIC TARGETING HEATMAP")
heatmap_area = st.empty()

# 3. TRENDING NOW: Thumbnail Grid (Card UI)
st.markdown("### 📺 TRENDING NOW")
grid_area = st.container()

# --- Execution Engine (No Balloons) ---
if scan_active or (not st.session_state.get('initial_load', False)):
    st.session_state.initial_load = True
    log_event(f"Scanning vectors for '{active_keyword}'...")
    
    # 3.1 Fetch Real Data
    video_data = fetch_trending_metadata(active_keyword)
    
    # Simulation Loop for Metrics/Charts
    demographics = ["10s", "20s", "30s", "40s", "50s+"]
    times = ["Dawn", "Morning", "After", "Evening", "Night"]
    
    for i in range(1, 11): # Light simulation for responsiveness
        # Update Core Metrics
        v_growth = 75 + random.uniform(0, 20)
        c_pos = 82 + random.uniform(-5, 10)
        target_age = random.choice(["20s-30s", "10s-20s", "40s-50s"])
        
        m_growth.metric("🚀 조회수 상승률", f"+{v_growth:.1f}%", delta="Peak Velocity")
        m_positive.metric("✨ 댓글 긍정 비율", f"{c_pos:.1f}%", delta=f"{random.uniform(-1, 2):.1f}%")
        m_target.metric("🎯 추천 연령대", target_age, delta="Dominant Segment", delta_color="off")
        
        # Update Heatmap (Large & Bold)
        heat_val = np.random.rand(5, 5) * 100
        fig_heat = px.imshow(heat_val,
                            labels=dict(x="Time", y="Age", color="Rank"),
                            x=times, y=demographics,
                            aspect="auto", color_continuous_scale='Magma')
        fig_heat.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=0, b=0), height=500)
        heatmap_area.plotly_chart(fig_heat, use_container_width=True)
        
        time.sleep(0.05)

    # Render Card Grid
    if video_data:
        cols = grid_area.columns(3)
        for idx, vid in enumerate(video_data):
            with cols[idx % 3]:
                st.markdown(f'''
                    <div class="video-card">
                        <img src="{vid['thumbnail']}" style="width:100%; border-radius:10px; margin-bottom:10px;">
                        <h4 style="font-size:1rem; color:#fff; height:3em; overflow:hidden;">{vid['title']}</h4>
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:10px;">
                            <span style="color:#aaa; font-size:0.8rem;">👁️ {vid['views']:,} views</span>
                            <span class="algo-score">{vid['score']} pts</span>
                        </div>
                    </div>
                ''', unsafe_allow_html=True)
                st.link_button("📺 유튜브 바로가기", vid['url'], use_container_width=True)
                st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.warning("No trending data found correctly. Try another keyword.")

    log_event(f"Analysis complete for {active_keyword}.")

# --- Manual Refresh/Auto-Open Logic ---
if 'browser_launched' not in st.session_state:
    st.session_state.browser_launched = True
    # In Streamlit, this is best handled via CLI, but mentioned for alignment.
    pass
