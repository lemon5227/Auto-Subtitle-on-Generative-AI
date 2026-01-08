import os
import uuid
import glob
import shutil
import subprocess
try:
    import yt_dlp as ytdlp_api
except ImportError:
    ytdlp_api = None

class VideoDownloader:
    def __init__(self, save_dir: str = './save'):
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def get_video_id(self, url: str) -> str:
        """Get a stable ID for the video URL"""
        if ytdlp_api is not None:
            try:
                with ytdlp_api.YoutubeDL({'quiet': True}) as ydl:
                    info = ydl.extract_info(url, download=False)
                    return info.get('id') or str(uuid.uuid4())
            except Exception:
                pass
        
        # Fallback to system binary
        if shutil.which('yt-dlp') is not None:
            try:
                res = subprocess.run(['yt-dlp', '--get-id', url], capture_output=True, text=True, check=True)
                return res.stdout.strip() or str(uuid.uuid4())
            except Exception:
                pass
        
        return str(uuid.uuid4())

    def download_video(self, url: str, video_id: str = None) -> str:
        """
        Download video from URL.
        Returns the absolute path to the downloaded file.
        Raises Exception on failure.
        """
        if not video_id:
            video_id = self.get_video_id(url)

        output_template = os.path.join(self.save_dir, f"{video_id}.%(ext)s")
        
        # Check cache: if file exists, return it immediately
        existing_matches = glob.glob(os.path.join(self.save_dir, f"{video_id}.*"))
        if existing_matches:
            # Filter distinct extensions to avoid partials or subtitles if we want just video
            # Reuse the selection logic found at bottom
            preferred_exts = ['.mp4', '.mkv', '.webm', '.mov', '.mp3', '.m4a', '.wav', '.aac', '.flac', '.ogg', '.opus']
            for ext in preferred_exts:
                for m in existing_matches:
                    if m.lower().endswith(ext):
                        print(f"File cached: {m}")
                        return os.path.abspath(m)
        
        if ytdlp_api is not None:
            # Check for ffmpeg to decide format strategy
            has_ffmpeg = shutil.which('ffmpeg') is not None
            if has_ffmpeg:
                # Merge logic needs ffmpeg
                format_str = 'bestvideo[height<=720]+bestaudio/best[height<=720]'
            else:
                # Fallback to single file if no ffmpeg
                print("⚠️ ffmpeg not found. Falling back to single file download (format='best').")
                format_str = 'best[ext=mp4]/best'

            ydl_opts = {
                'outtmpl': output_template,
                'format': format_str,
                'merge_output_format': 'mp4' if has_ffmpeg else None
            }
            with ytdlp_api.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        else:
            if shutil.which('yt-dlp') is None:
                raise RuntimeError("'yt-dlp' not found. Install backend package or ensure yt-dlp is on PATH.")
            
            # Check for ffmpeg for CLI as well
            has_ffmpeg = shutil.which('ffmpeg') is not None
            if has_ffmpeg:
                 format_spec = 'bestvideo[height<=720]+bestaudio/best[height<=720]'
                 cmd = ['yt-dlp', '-f', format_spec, '--merge-output-format', 'mp4', '-o', output_template, url]
            else:
                 print("⚠️ ffmpeg not found. Falling back to single file download.")
                 format_spec = 'best[ext=mp4]/best'
                 cmd = ['yt-dlp', '-f', format_spec, '-o', output_template, url]
                 
            subprocess.run(cmd, check=True)

        # Locate file
        matches = glob.glob(os.path.join(self.save_dir, f"{video_id}.*"))
        if matches:
            preferred_exts = ['.mp4', '.mkv', '.webm', '.mov', '.mp3', '.m4a', '.wav', '.aac', '.flac', '.ogg', '.opus']
            chosen = None
            for ext in preferred_exts:
                for m in matches:
                    if m.lower().endswith(ext):
                        chosen = m
                        break
                if chosen:
                    break
            
            if not chosen:
                for m in matches:
                    if not m.lower().endswith(('.vtt', '.srt', '.ass', '.sub')):
                        chosen = m
                        break
            
            if chosen:
                return os.path.abspath(chosen)
        
        raise FileNotFoundError("File not found after download.")
