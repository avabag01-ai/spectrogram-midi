import os
import sys
import subprocess
import time
import re
import yt_dlp
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn
from rich.live import Live
from rich.table import Table

console = Console()

def get_pure_title(title, artist="이승철"):
    """Advanced title cleaning and normalization."""
    title = re.sub(r'\[.*?\]|\(.*?\)', '', title)
    junk = [artist, "Special", "OST", "Live", "라이브", "Official", "MV", "Lyrics", "M/V", "High Quality"]
    pattern = '|'.join([re.escape(k) for k in junk])
    title = re.sub(pattern, '', title, flags=re.IGNORECASE)
    title = re.sub(r'[^\w\s가-힣]', '', title)
    title = " ".join(title.split())
    return title.strip()

def format_size(num):
    """File size human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}TB"

def run_collector(artist="이승철", max_count=100):
    home = os.path.expanduser("~")
    save_path = os.path.join(home, "Downloads", f"{artist}_Music")
    abs_save_path = os.path.abspath(save_path)

    # 1. Initialize UI
    console.clear()
    console.print(Panel.fit(
        f"[bold magenta]🛡️ Aegis Deep Scan Mode (가속 수집)[/bold magenta]\n"
        f"현재 저장 위치: [bold yellow]{abs_save_path}[/bold yellow]\n"
        f"최대 타겟 검색수: [bold green]{max_count}[/bold green]",
        title="[bold white]Deep Data Scraper Activated[/bold white]",
        border_style="magenta"
    ))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    downloaded_files = []
    
    # 2. Progress Engine
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:

        try:
            ydl_opts = {
                'format': '18/bestaudio/best',
                'outtmpl': f"{save_path}/%(title)s.%(ext)s",
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '320',
                }],
                'ignoreerrors': False,
                'quiet': True,
                'no_warnings': True,
                'nocheckcertificate': True,
                'default_search': f'ytsearch{max_count}',
                'extractor_args': {'youtube': {'player_client': ['android'], 'skip': ['webpage']}},
            }

            main_task = progress.add_task(f"[bold cyan]🔍 {artist} 딥 스캔 중...", total=max_count)

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"ytsearch{max_count}:{artist}", download=False)
                if 'entries' not in info:
                    console.print("[red]데이터를 찾을 수 없습니다.[/red]")
                    return

                entries = info['entries']
                for idx, entry in enumerate(entries):
                    if not entry: continue
                    
                    raw_title = entry.get('title', 'Unknown')
                    duration = entry.get('duration', 0)
                    pure_name = get_pure_title(raw_title, artist)
                    
                    # Filtering Logic
                    if not (30 <= duration <= 330) or not pure_name:
                        progress.update(main_task, advance=1, description=f"[dim]Filter out: {raw_title[:20]}...[/dim]")
                        continue

                    final_filename = f"{pure_name}.mp3"
                    final_path = os.path.join(save_path, final_filename)

                    # 🛡️ Architect's Double Check: Existence + File Health (0-byte check)
                    skip_download = False
                    if os.path.exists(final_path):
                        if os.path.getsize(final_path) > 1024: # 1KB 이상이면 완상 본으로 간주
                            skip_download = True
                        else:
                            console.print(f"[bold red]Found Damaged File[/bold red]: '{final_filename}' (0-byte). [bold yellow]Re-downloading...[/bold yellow]")

                    if skip_download:
                        progress.update(main_task, advance=1, description=f"[bold blue]Keep: {pure_name}[/bold blue]")
                        continue

                    # 3. Real-time Download monitoring
                    progress.update(main_task, description=f"[bold green]📥 Deep Scraping: {pure_name}[/bold green]")
                    
                    curr_opts = dict(ydl_opts)
                    curr_opts['outtmpl'] = os.path.join(save_path, f"{pure_name}.%(ext)s")
                    
                    with yt_dlp.YoutubeDL(curr_opts) as ydl_down:
                        ydl_down.download([entry['webpage_url']])

                    # Final verification and Reporting
                    if os.path.exists(final_path):
                        f_size = os.path.getsize(final_path)
                        console.print(f"[bold white]ls -lh[/bold white] [green]'{final_filename}'[/green] [bold yellow]{format_size(f_size)}[/bold yellow] [dim]저장 완료[/dim]")
                        downloaded_files.append(final_filename)

                    progress.update(main_task, advance=1)

        except KeyboardInterrupt:
            console.print("\n\n [bold red]⚠️  사용자 강제 중단 (Ctrl+C)[/bold red]")
        except Exception as e:
            console.print(f"\n [bold red]ERROR:[/bold red] {e}")

    # 3. Completion Summary
    console.print("\n" + "="*50)
    summary_table = Table(title="Collection Summary", show_header=True, header_style="bold magenta")
    summary_table.add_column("No", style="dim", width=4)
    summary_table.add_column("FileName", style="cyan")
    summary_table.add_column("Status", justify="right", style="green")

    for i, f in enumerate(downloaded_files):
        summary_table.add_row(str(i+1), f, "SUCCESS")

    console.print(summary_table)
    console.print(f"\n[bold green]✅ 총 {len(downloaded_files)}곡이 {abs_save_path} 폴더에 저장되었습니다.[/bold green]")
    
    # 4. Auto Finder Popup
    console.print("[bold yellow]🚀 Finder에서 폴더를 여는 중...[/bold yellow]")
    subprocess.run(["open", save_path])

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "이승철"
    run_collector(target)
