import yt_dlp
import os
import sys
import subprocess

def check_ffmpeg():
    """Check if ffmpeg is installed and accessible."""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

def download_youtube_audio(query, quality='320'):
    """Search and download the highest quality audio from YouTube and convert to MP3."""
    
    if not check_ffmpeg():
        print("❌ Error: FFmpeg is not installed or not found in PATH.")
        print("Please install FFmpeg to use this script. (e.g., 'brew install ffmpeg' on macOS)")
        return

    # yt-dlp options
    ydl_opts = {
        # Format: best audio or format 18 (fallback for 403)
        'format': '18/bestaudio/best',
        
        # Search query handling: if it's not a URL, prepend ytsearch1:
        'default_search': 'ytsearch1',
        
        # Security/Compatibility Fixes
        'nocheckcertificate': True,
        'ignoreerrors': False, # Change to False to catch real errors
        'no_warnings': False,
        'quiet': False,
        'extract_flat': False,
        
        # Bypassing 403 Forbidden (Working Android Client Trick)
        'extractor_args': {
            'youtube': {
                'player_client': ['android'],
                'skip': ['webpage'] # Sometimes skipping webpage helps
            }
        },
        
        # Post-processing: Convert to mp3
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': quality,
        }],
        
        # Metadata and Filename
        'outtmpl': '%(title)s.%(ext)s', 
        
        # Audio optimization
        'postprocessor_args': [
            '-ar', '44100', 
            '-ac', '2',
        ],
        
        # Logging
        'logger': MyLogger(),
        'progress_hooks': [my_hook],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"🔍 Searching for: {query}...")
            info = ydl.extract_info(query, download=True)
            
            if 'entries' in info:
                video_title = info['entries'][0]['title']
                video_id = info['entries'][0]['id']
            else:
                video_title = info['title']
                video_id = info['id']
            
            # Check if the file actually exists (yt-dlp sometimes lies)
            expected_file = f"{video_title}.mp3"
            if os.path.exists(expected_file) or any(f.endswith(".mp3") and video_title[:10] in f for f in os.listdir('.')):
                print(f"\n✅ Successfully downloaded and converted: {video_title}.mp3")
            else:
                print(f"\n❌ Error: File was not created despite reported success.")
            
    except Exception as e:
        print(f"❌ An unexpected error occurred: {str(e)}")

class MyLogger:
    def debug(self, msg):
        # For searching status, let's keep it quiet unless it's a real message
        if msg.startswith('[debug] '):
            pass
        else:
            self.info(msg)

    def info(self, msg):
        print(msg)

    def warning(self, msg):
        print(f"⚠️ Warning: {msg}")

    def error(self, msg):
        print(f"❌ Error: {msg}")

def my_hook(d):
    if d['status'] == 'downloading':
        p = d.get('_percent_str', '0%')
        s = d.get('_speed_str', 'N/A')
        t = d.get('_eta_str', 'N/A')
        print(f"\r📥 Downloading: {p} at {s} (ETA: {t})", end='', flush=True)
    elif d['status'] == 'finished':
        print("\n✨ Download complete, converting to MP3...")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("📖 Usage: python3 youtube_audio_extractor.py \"Search Query or URL\" [quality: 192 or 320]")
        sys.exit(1)
    
    search_query = sys.argv[1]
    audio_quality = sys.argv[2] if len(sys.argv) > 2 else '320'
    
    if audio_quality not in ['192', '320']:
        print("⚠️ Warning: Quality should be 192 or 320. Defaulting to 320.")
        audio_quality = '320'
        
    download_youtube_audio(search_query, audio_quality)
