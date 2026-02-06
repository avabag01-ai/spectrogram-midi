import yt_dlp
import os
import re
import sys
import time

class MegaBatchCollector:
    def __init__(self, artist_name, quality='320'):
        self.artist_name = artist_name
        self.quality = quality
        self.output_dir = f"downloads/{artist_name}"
        self.downloaded_titles = set()
        self.stats = {
            'total_found': 0,
            'downloaded': 0,
            'skipped_duration': 0,
            'skipped_keyword': 0,
            'skipped_duplicate': 0,
            'failed': 0
        }
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        else:
            # Load existing files for deduplication
            for f in os.listdir(self.output_dir):
                if f.endswith(".mp3"):
                    clean_name = self.clean_title(f.replace(".mp3", ""))
                    self.downloaded_titles.add(clean_name)

    def clean_title(self, title):
        """Advanced cleaning for superior deduplication."""
        # 1. Remove anything in brackets [] or ()
        title = re.sub(r'\[.*?\]|\(.*?\)', '', title)
        # 2. Dynamic Artist Name & Keywords removal
        keywords = [self.artist_name, "Special", "OST", "Live", "라이브", "부활", "Official", "MV", "Lyrics", "ft.", "feat"]
        pattern = '|'.join([re.escape(k) for k in keywords])
        title = re.sub(pattern, '', title, flags=re.IGNORECASE)
        # 3. Special chars & Whitespace removal
        title = re.sub(r'[^\w\s]', '', title)
        return title.replace(" ", "").strip().lower()

    def is_valid_title(self, title):
        """Check for blocked keywords."""
        block_keywords = ['full album', '모음', 'mix', '1시간', 'loop', 'medley', 'collection', 'playlist', '연속듣기']
        title_lower = title.lower()
        for kw in block_keywords:
            if kw in title_lower:
                return False
        return True

    def get_video_list(self):
        """Search and extract video IDs from various sources."""
        queries = [
            f"ytsearch50:{self.artist_name} official album",
            f"ytsearch50:{self.artist_name} songs playlist",
            f"ytsearch50:{self.artist_name} top hits"
        ]
        
        ydl_opts = {
            'flat_playlist': True,
            'ignoreerrors': True,
            'nocheckcertificate': True,
            'quiet': True,
            'no_warnings': True,
            'extract_flat': 'in_playlist',
        }
        
        all_entries = []
        print(f"🔍 Searching for songs by '{self.artist_name}'...")
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            for q in queries:
                try:
                    info = ydl.extract_info(q, download=False)
                    if 'entries' in info:
                        all_entries.extend(info['entries'])
                except Exception as e:
                    print(f"⚠️ Search error for '{q}': {e}")
        
        return all_entries

    def process_and_download(self):
        entries = self.get_video_list()
        self.stats['total_found'] = len(entries)
        
        # Deduplicate entries by ID first
        unique_entries = {e['id']: e for e in entries if e}.values()
        
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

        print(f"🎯 Found {len(unique_entries)} potential videos. Starting filtering and download...")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            for entry in unique_entries:
                try:
                    title = entry.get('title', 'Unknown')
                    url = f"https://www.youtube.com/watch?v={entry['id']}"
                    
                    # 1. Keyword Filter
                    if not self.is_valid_title(title):
                        self.stats['skipped_keyword'] += 1
                        continue

                    # 2. Duplicate Filter (Core name match)
                    clean = self.clean_title(title)
                    if clean in self.downloaded_titles:
                        self.stats['skipped_duplicate'] += 1
                        continue

                    # 3. Duration Filter (Requires individual info extraction)
                    # We do this only for filtered titles to save time
                    info = ydl.extract_info(url, download=False)
                    duration = info.get('duration', 0)
                    
                    if not (30 <= duration <= 330):
                        self.stats['skipped_duration'] += 1
                        continue

                    # Start Download
                    print(f"\n🚀 Downloading: {title}")
                    ydl.download([url])
                    
                    self.downloaded_titles.add(clean)
                    self.stats['downloaded'] += 1
                    
                except Exception as e:
                    self.stats['failed'] += 1
                    print(f"\n❌ Error downloading {entry.get('id')}: {e}")

        self.print_final_summary()

    def progress_hook(self, d):
        if d['status'] == 'downloading':
            p = d.get('_percent_str', '0%')
            s = d.get('_speed_str', 'N/A')
            print(f"\r   Progress: {p} | Speed: {s} | Stats: [D:{self.stats['downloaded']} S:{self.stats['skipped_duration']+self.stats['skipped_keyword']+self.stats['skipped_duplicate']} F:{self.stats['failed']}]", end='', flush=True)

    def print_final_summary(self):
        print("\n\n" + "="*50)
        print(f"🏁 Collection Summary for: {self.artist_name}")
        print("="*50)
        print(f"✅ Downloaded: {self.stats['downloaded']}")
        print(f"⏭️ Skipped (Duration): {self.stats['skipped_duration']}")
        print(f"⏭️ Skipped (Keyword): {self.stats['skipped_keyword']}")
        print(f"⏭️ Skipped (Duplicate): {self.stats['skipped_duplicate']}")
        print(f"❌ Failed: {self.stats['failed']}")
        print(f"📂 Location: {os.path.abspath(self.output_dir)}")
        print("="*50)

class MyLogger:
    def debug(self, msg): pass
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): print(f"\n❌ {msg}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("📖 Usage: python3 mega_batch_collector.py \"Artist Name\" [quality: 192 or 320]")
        sys.exit(1)
    
    name = sys.argv[1]
    qual = sys.argv[2] if len(sys.argv) > 2 else '320'
    
    collector = MegaBatchCollector(name, qual)
    collector.process_and_download()
