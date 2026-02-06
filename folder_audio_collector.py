import streamlit as st
import yt_dlp
import os
import re
import time
import io
import pandas as pd
from aegis_engine import AegisEngine

# --- 📁 Folder-Based Audio Collector (Physical Copy Optimization) ---
st.set_page_config(page_title="Physical Audio Collector", layout="centered")

# 🔐 Security Code (Same as previous for continuity)
ACCESS_CODE = "yi2026"

def authenticate():
    if "phys_auth" not in st.session_state:
        st.session_state.phys_auth = False
    
    if not st.session_state.phys_auth:
        st.title("🛡️ Secure Access")
        pw = st.text_input("액세스 코드를 입력하세요", type="password")
        if st.button("접속"):
            if pw == ACCESS_CODE:
                st.session_state.phys_auth = True
                st.rerun()
            else:
                st.error("승인되지 않은 사용자입니다. 아키텍트 Yi에게 문의하세요.")
        return False
    return True

if authenticate():
    st.title("📁 Physical Audio Collector")
    st.markdown("가수별 폴더를 생성하고 정제된 MP3를 **직접 저장**합니다.")

    # 1. Input Section
    artist_name = st.text_input("🎤 가수 이름 입력", placeholder="예: 이승철, NewJeans").strip()
    
    with st.expander("⚙️ 고급 수집 설정"):
        audio_quality = st.selectbox("음질 (kbps)", ["320", "192"], index=0)
        max_songs = st.slider("최대 수집 곡 수", 5, 100, 30)
        base_path = st.text_input("저장 경로 (Root)", value="downloads")

    # 2. Logic: Cleaning & Normalization
    def get_pure_title(title, artist):
        # 1. Remove brackets content
        title = re.sub(r'\[.*?\]|\(.*?\)', '', title)
        # 2. Remove artist & junk keywords
        junk = [artist, "Special", "OST", "Live", "라이브", "Official", "MV", "Lyrics", "M/V", "High Quality"]
        pattern = '|'.join([re.escape(k) for k in junk])
        title = re.sub(pattern, '', title, flags=re.IGNORECASE)
        # 3. Remove special chars except for filename safety
        title = re.sub(r'[^\w\s가-힣]', '', title)
        title = " ".join(title.split()) # Normalize spaces
        return title.strip()

    if st.button("🚀 폴더 기반 일괄 수집 시작", use_container_width=True, type="primary"):
        if not artist_name:
            st.warning("가수 이름을 입력해주세요.")
        else:
            # Create physical directory
            target_dir = os.path.join(base_path, artist_name)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            # Monitoring Areas
            status_msg = st.empty()
            progress_bar = st.progress(0)
            log_box = st.empty()
            
            # Tracking
            downloaded_count = 0
            existing_files = [f.lower() for f in os.listdir(target_dir)]
            
            # yt-dlp Options
            ydl_opts = {
                'format': '18/bestaudio/best',
                'outtmpl': f"{target_dir}/%(title)s.%(ext)s",
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': audio_quality,
                }],
                'ignoreerrors': False,
                'no_warnings': True,
                'quiet': True,
                'nocheckcertificate': True,
                'default_search': f'ytsearch{max_songs}',
                'extractor_args': {
                    'youtube': {
                        'player_client': ['android'],
                        'skip': ['webpage']
                    }
                },
            }

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    status_msg.info(f"🔍 '{artist_name}' 데이터 확보 중...")
                    search_data = ydl.extract_info(artist_name, download=False)
                    
                    if 'entries' in search_data:
                        entries = search_results = search_data['entries']
                        total = len(entries)
                        
                        for i, entry in enumerate(entries):
                            if not entry: continue
                            
                            p_title = entry.get('title', 'Unknown')
                            duration = entry.get('duration', 0)
                            
                            # A. Time Limit (30s ~ 330s)
                            if not (30 <= duration <= 330):
                                continue
                            
                            # B. Normalization & Deduplication
                            pure_name = get_pure_title(p_title, artist_name)
                            if not pure_name: continue
                            
                            filename = f"{pure_name}.mp3"
                            if filename.lower() in existing_files:
                                continue # Skip if already exists
                            
                            # C. Physical Save
                            status_msg.warning(f"📥 수집 중: {pure_name}")
                            # Custom output template for this specific file
                            curr_opts = dict(ydl_opts)
                            curr_opts['outtmpl'] = f"{target_dir}/{pure_name}.%(ext)s"
                            
                            with yt_dlp.YoutubeDL(curr_opts) as ydl_down:
                                ydl_down.download([entry['webpage_url']])
                            
                            downloaded_count += 1
                            existing_files.append(filename.lower())
                            progress_bar.progress((i + 1) / total)
                            log_box.success(f"💾 저장됨: {filename}")
                        
                        # Final Message
                        abs_path = os.path.abspath(target_dir)
                        st.divider()
                        st.subheader("🏁 수집 프로세스 종료")
                        st.success(f"**{artist_name}** 폴더에 총 **{downloaded_count}곡**이 저장되었습니다.")
                        st.info(f"📂 **저장 경로:** `{abs_path}`")
                        st.balloons()
                        
                        if downloaded_count > 0:
                            files = os.listdir(target_dir)
                            st.dataframe(pd.DataFrame({"파일명": files}), use_container_width=True)
                    else:
                        st.error("영상을 찾을 수 없습니다.")
            except Exception as e:
                st.error(f"시스템 오류: {e}")

    # Logout & Footer
    st.sidebar.markdown("---")
    if st.sidebar.button("로그아웃"):
        st.session_state.phys_auth = False
        st.rerun()
    st.sidebar.caption(f"Physical Copy Engine v1.0\nArchitect: Yi")
