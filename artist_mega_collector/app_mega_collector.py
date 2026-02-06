import streamlit as st
import yt_dlp
import os
import re
import io
import time
import pandas as pd
from mega_batch_collector import MegaBatchCollector, MyLogger

# --- 🛰️ Artist Mega-Batch Collector: GUI Edition ---
st.set_page_config(page_title="Artist Mega Collector", layout="wide")

st.title("🛰️ Artist Mega-Batch Collector (Clean Mode)")
st.markdown("특정 가수 이름을 입력하면 중복과 지저분한 영상(메들리, 풀앨범 등)을 제외하고 **순수 곡들만 고음질로 수집**합니다.")

# 1. Sidebar Configuration
st.sidebar.header("⚙️ Collection Settings")
artist_name = st.sidebar.text_input("가수 이름 (e.g. 이승철, NewJeans)", value="").strip()
audio_quality = st.sidebar.selectbox("오디오 음질 (kbps)", ["320", "192"], index=0)

st.sidebar.subheader("📏 Duration Filter")
min_sec = st.sidebar.slider("Minimum (Sec)", 10, 60, 30)
max_sec = st.sidebar.slider("Maximum (Sec)", 120, 600, 330)

st.sidebar.subheader("🚫 Block Keywords")
custom_blocks = st.sidebar.text_area("쉼표로 구분", "Full Album, 모음, Mix, 1시간, Loop, Medley, Collection, Playlist, 연속듣기")
block_list = [k.strip().lower() for k in custom_blocks.split(",")]

# 2. Main Logic Override for Streamlit
class StreamlitCollector(MegaBatchCollector):
    def __init__(self, artist_name, quality, min_s, max_s, b_list):
        super().__init__(artist_name, quality)
        self.min_s = min_s
        self.max_s = max_s
        self.b_list = b_list
        self.status_area = st.empty()
        self.progress_bar = st.progress(0)
        self.log_area = st.empty()
        self.logs = []

    def is_valid_title(self, title):
        title_lower = title.lower()
        for kw in self.b_list:
            if kw in title_lower: return False
        return True

    def st_log(self, msg):
        self.logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        # Keep only last 10 logs
        log_text = "\n".join(self.logs[-10:])
        self.log_area.code(log_text)

    def progress_hook(self, d):
        if d['status'] == 'downloading':
            p_str = d.get('_percent_str', '0%').replace('%', '')
            try:
                p_float = float(p_str) / 100.0
                self.progress_bar.progress(p_float)
            except: pass
            
    def process_and_download_st(self):
        entries = self.get_video_list()
        self.stats['total_found'] = len(entries)
        unique_entries = list({e['id']: e for e in entries if e}.values())
        
        self.st_log(f"🔍 Found {len(unique_entries)} potential match videos.")
        
        ydl_opts = {
            'format': '18/bestaudio/best',
            'outtmpl': f"{self.output_dir}/%(title)s.%(ext)s",
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': self.quality,
            }],
            'ignoreerrors': False,
            'nocheckcertificate': True,
            'no_warnings': True,
            'postprocessor_args': ['-ar', '44100', '-ac', '2'],
            'logger': MyLogger(),
            'progress_hooks': [self.progress_hook],
            'extractor_args': {
                'youtube': {
                    'player_client': ['android'],
                    'skip': ['webpage']
                }
            },
        }

        total = len(unique_entries)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            for i, entry in enumerate(unique_entries):
                title = entry.get('title', 'Unknown')
                url = f"https://www.youtube.com/watch?v={entry['id']}"
                
                # Update Dashboard
                self.status_area.markdown(f"""
                ### 📊 Collection Dashboard
                | Category | Count |
                | :--- | :--- |
                | 🔎 Total Scanned | {self.stats['total_found']} |
                | ✅ **Downloaded** | **{self.stats['downloaded']}** |
                | ⏭️ Skipped (Keyword) | {self.stats['skipped_keyword']} |
                | ⏭️ Skipped (Duration) | {self.stats['skipped_duration']} |
                | ⏭️ Skipped (Duplicate) | {self.stats['skipped_duplicate']} |
                | ❌ Failed | {self.stats['failed']} |
                """)

                # 1. Keyword
                if not self.is_valid_title(title):
                    self.stats['skipped_keyword'] += 1
                    continue

                # 2. Duplicate
                clean = self.clean_title(title)
                if clean in self.downloaded_titles:
                    self.stats['skipped_duplicate'] += 1
                    continue

                # 3. Duration
                try:
                    info = ydl.extract_info(url, download=False)
                    duration = info.get('duration', 0)
                    if not (self.min_s <= duration <= self.max_s):
                        self.stats['skipped_duration'] += 1
                        continue
                except:
                    self.stats['failed'] += 1
                    continue

                self.st_log(f"🚀 Downloading: {title}")
                ydl.download([url])
                self.downloaded_titles.add(clean)
                self.stats['downloaded'] += 1

        st.balloons()
        st.success(f"🏁 Collection Complete! Total {self.stats['downloaded']} songs saved.")

# 3. UI Layout
if artist_name:
    if st.button("🚀 Start Mega Collection"):
        collector = StreamlitCollector(artist_name, audio_quality, min_sec, max_sec, block_list)
        collector.process_and_download_st()

else:
    st.info("👈 사이드바에 가수 이름을 입력하고 버튼을 누르세요.")

# 4. Results Viewer
if artist_name:
    target_path = f"downloads/{artist_name}"
    if os.path.exists(target_path):
        st.divider()
        st.subheader(f"📂 My '{artist_name}' Collection")
        files = [f for f in os.listdir(target_path) if f.endswith(".mp3")]
        if files:
            df = pd.DataFrame({"Song Title": files})
            st.dataframe(df, use_container_width=True)
            
            # Simple list for audit
            st.write(f"총 {len(files)}곡 보유 중")
        else:
            st.write("아직 다운로드된 곡이 없습니다.")
