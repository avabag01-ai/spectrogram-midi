import streamlit as st
import yt_dlp
import os
import re
import shutil
import zipfile
import time
import tempfile
import io

# --- 🔐 Secure Mobile Audio Collector: Cloud/Mobile Edition ---
st.set_page_config(page_title="Secure Collector", layout="centered", initial_sidebar_state="collapsed")

# 1. Security Logic
ACCESS_CODE = "yi2026"

def check_access():
    if "authorized" not in st.session_state:
        st.session_state.authorized = False

    if not st.session_state.authorized:
        st.title("🛡️ Access Security")
        code = st.text_input("액세스 코드를 입력하세요", type="password")
        if st.button("인증하기"):
            if code == ACCESS_CODE:
                st.session_state.authorized = True
                st.rerun()
            else:
                st.error("승인되지 않은 사용자입니다. 아키텍트 Yi에게 문의하세요.")
        return False
    return True

if check_access():
    # --- 🏗️ Main UI (Mobile Optimized) ---
    st.title("🛰️ Secure Audio Collector")
    st.markdown("가수 이름을 입력하여 고음질 MP3를 일괄 수집하세요.")
    
    artist_name = st.text_input("🎤 가수 이름 (Artist Name)", placeholder="예: 이승철, NewJeans").strip()
    
    # Advanced Options (Hidden by default for mobile simplicity)
    with st.expander("⚙️ 고급 필터 설정"):
        audio_quality = st.selectbox("품질", ["320", "192"], index=0)
        max_results = st.slider("수집 곡 수", 5, 50, 20)

    if st.button("🚀 일괄 수집 시작", use_container_width=True, type="primary"):
        if not artist_name:
            st.warning("가수 이름을 입력해주세요.")
        else:
            # Setup working directories
            temp_dir = tempfile.mkdtemp()
            download_dir = os.path.join(temp_dir, artist_name)
            os.makedirs(download_dir)
            
            # Progress tracking
            status_text = st.empty()
            progress_bar = st.progress(0)
            log_area = st.empty()
            
            # Metadata for deduplication
            seen_titles = set()
            
            # yt-dlp Configuration
            ydl_opts = {
                'format': '18/bestaudio/best',
                'outtmpl': f"{download_dir}/%(title)s.%(ext)s",
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': audio_quality,
                }],
                'ignoreerrors': False,
                'no_warnings': True,
                'quiet': True,
                'default_search': f'ytsearch{max_results}',
                'nocheckcertificate': True,
                'extractor_args': {
                    'youtube': {
                        'player_client': ['android'],
                        'skip': ['webpage']
                    }
                },
            }

            def clean_title_advanced(t, a_name):
                # 1. Brackets/Parentheses removal
                t = re.sub(r'\[.*?\]|\(.*?\)', '', t)
                # 2. Dynamic Artist Name & Keywords removal
                # Common keywords to strip for cleaner match
                keywords = [a_name, "Special", "OST", "Live", "라이브", "부활", "Official", "MV", "Lyrics"]
                pattern = '|'.join([re.escape(k) for k in keywords])
                t = re.sub(pattern, '', t, flags=re.IGNORECASE)
                # 3. Special chars & Whitespace removal
                t = re.sub(r'[^\w\s]', '', t)
                return t.replace(" ", "").strip().lower()

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    status_text.info(f"🔍 '{artist_name}' 곡 검색 중...")
                    search_results = ydl.extract_info(f"{artist_name}", download=False)
                    
                    if 'entries' in search_results:
                        entries = search_results['entries']
                        total_found = len(entries)
                        count = 0
                        
                        for i, entry in enumerate(entries):
                            if not entry: continue
                            
                            title = entry.get('title', 'Unknown')
                            duration = entry.get('duration', 0)
                            
                            # 1. Duration Filter (30s ~ 330s)
                            if not (30 <= duration <= 330):
                                continue
                            
                            # 2. Keyword Filter (Strict Blocks)
                            blocked = ["full album", "모음", "mix", "1시간", "연속듣기", "playlist", "medley"]
                            if any(k in title.lower() for k in blocked):
                                continue
                                
                            # 3. Advanced Deduplication
                            ctitle = clean_title_advanced(title, artist_name)
                            if not ctitle or ctitle in seen_titles:
                                continue
                            
                            # Start Download
                            status_text.warning(f"📥 필터 통과! 다운로드 중: {title[:20]}...")
                            ydl.download([entry['webpage_url']])
                            
                            seen_titles.add(ctitle)
                            count += 1
                            progress_bar.progress((i + 1) / total_found)
                            log_area.caption(f"✅ {title} 수집 완료")
                        
                        if count > 0:
                            # 3. ZIP Compression
                            status_text.info("📦 파일 압축 중...")
                            zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                                for root, dirs, files in os.walk(download_dir):
                                    for file in files:
                                        zf.write(os.path.join(root, file), arcname=file)
                            
                            st.success(f"🏁 총 {count}곡 수집 및 압축이 완료되었습니다!")
                            
                            # ZIP Download Button
                            st.download_button(
                                label="💾 최종 결과물 다운로드 (ZIP)",
                                data=zip_buffer.getvalue(),
                                file_name=f"{artist_name}_collection.zip",
                                mime="application/zip",
                                use_container_width=True
                            )
                        else:
                            st.error("조건에 맞는 곡을 찾지 못했습니다. 필터를 조정해보세요.")
                    else:
                        st.error("검색 결과를 가져올 수 없습니다.")
            except Exception as e:
                st.error(f"오류 발생: {e}")
            finally:
                # Cleanup
                shutil.rmtree(temp_dir)

    # logout
    if st.sidebar.button("로그아웃"):
        st.session_state.authorized = False
        st.rerun()

    st.markdown("---")
    st.caption("Powered by Aegis AI Perception Engine | Architect: Yi")
