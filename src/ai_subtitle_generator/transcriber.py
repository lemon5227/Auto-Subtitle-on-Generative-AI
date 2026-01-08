import os
import glob
import subprocess
import shutil
import whisper
try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    FASTER_AVAILABLE = True
except ImportError:
    FASTER_AVAILABLE = False
import torch
from typing import Tuple
from .utils import format_time
from .gpu_detector import get_optimal_device

class SubtitleGenerator:
    def __init__(self, device=None):
        if not device:
            self.device, _ = get_optimal_device()
        else:
            self.device = device
            
    def generate_subtitles(self, video_path: str, model_name: str = 'tiny', segment_time: int = 300, 
                          use_faster: bool = False, language: str = None, 
                          progress_callback: callable = None) -> Tuple[str, str]:
        """
        Generate subtitles for a video file.
        
        Args:
            video_path: Path to video file
            model_name: Whisper model size
            segment_time: Segment duration in seconds
            use_faster: Use faster-whisper if available
            language: Language code (optional)
            progress_callback: Function(status_msg, progress_percent)
            
        Returns:
            Tuple(subtitles_path, vtt_content)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        base = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.dirname(video_path)
        
        # Temporary chunks directory
        import uuid
        job_id = str(uuid.uuid4())[:8]
        chunks_dir = os.path.join(output_dir, f'{job_id}_chunks')
        os.makedirs(chunks_dir, exist_ok=True)
        chunk_template = os.path.join(chunks_dir, 'chunk%03d.wav')
        
        try:
            # 1. Estimate chunks
            if progress_callback:
                progress_callback("Analyzing video duration...", 0)
                
            try:
                ffprobe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
                             '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
                res = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=True)
                duration = float(res.stdout.strip())
                estimated_total = max(1, int((duration + segment_time - 1) // segment_time))
            except Exception:
                estimated_total = 0

            # 2. Slice Audio
            if progress_callback:
                progress_callback(f"Creating chunks (approx {estimated_total})...", 5)
            
            cmd = ['ffmpeg', '-i', video_path, '-vn', '-ar', '44100', '-ac', '2', '-f', 'segment', 
                   '-segment_time', str(segment_time), '-reset_timestamps', '1', chunk_template]
            subprocess.run(cmd, check=True)
            
            chunks = sorted(glob.glob(os.path.join(chunks_dir, 'chunk*.wav')))
            total = len(chunks)
            if total == 0:
                raise RuntimeError('No audio chunks produced')

            # 3. Transcribe
            all_segments = []
            
            if use_faster and FASTER_AVAILABLE:
                # faster-whisper
                device_str = "cuda" if self.device == "cuda" else "cpu" # faster-whisper doesn't support 'mps' properly universally yet?
                # Actually app.py said: device="cuda" if torch.cuda.is_available() else "cpu" for faster-whisper (L974)
                # But inside extract_job_worker it used DEVICE. L1689: device=DEVICE
                # Let's trust DEVICE passed in, but faster-whisper might not support 'mps'
                
                fw_device = self.device
                if fw_device == 'mps':
                    fw_device = 'cpu' # faster-whisper does not support MPS currently
                
                compute_type = "float16" if fw_device == "cuda" else "int8"
                
                fw_model = FasterWhisperModel(model_name, device=fw_device, compute_type=compute_type)
                
                for idx, chunk in enumerate(chunks):
                    if progress_callback:
                        percent = int(5 + ((idx) / total) * 90)
                        progress_callback(f'Transcribing chunk {idx+1}/{total} (faster-whisper)', percent)

                    fw_lang = language if language else None
                    segments, info = fw_model.transcribe(chunk, beam_size=5, language=fw_lang, word_timestamps=False)
                    
                    offset = idx * segment_time
                    
                    for seg in segments:
                        # Normalize segment object
                        if hasattr(seg, 'start'):
                            start = seg.start
                            end = seg.end
                            text = seg.text
                        else:
                            # fallback
                            continue
                            
                        seg_start = start + offset
                        seg_end = end + offset
                        all_segments.append({'start': seg_start, 'end': seg_end, 'text': text})
            else:
                # openai-whisper
                try:
                    model = whisper.load_model(model_name, device=self.device)
                except TypeError:
                    model = whisper.load_model(model_name)
                    # Try moving to device
                    if hasattr(model, 'to'):
                         try:
                             model.to(self.device)
                         except:
                             pass
                             
                for idx, chunk in enumerate(chunks):
                    if progress_callback:
                        percent = int(5 + ((idx) / total) * 90)
                        progress_callback(f'Transcribing chunk {idx+1}/{total}', percent)
                    
                    res = model.transcribe(chunk, language=language) if language else model.transcribe(chunk)
                    offset = idx * segment_time
                    
                    for seg in res.get('segments', []):
                        seg_start = seg['start'] + offset
                        seg_end = seg['end'] + offset
                        all_segments.append({'start': seg_start, 'end': seg_end, 'text': seg['text']})

            # 4. Save VTT
            if progress_callback:
                 progress_callback("Finalizing subtitles...", 98)
                 
            vtt_content = 'WEBVTT\n\n'
            for i, seg in enumerate(all_segments, start=1):
                start = format_time(seg['start'])
                end = format_time(seg['end'])
                text = seg['text'].strip()
                vtt_content += f"{i}\n{start} --> {end}\n{text}\n\n"
            
            subtitles_filename = f'{base}.vtt'
            subtitles_path = os.path.join(output_dir, subtitles_filename)
            with open(subtitles_path, 'w', encoding='utf-8') as f:
                f.write(vtt_content)
                
            return subtitles_path, vtt_content

        finally:
            # Cleanup
            try:
                if os.path.exists(chunks_dir):
                    shutil.rmtree(chunks_dir)
            except Exception:
                pass
