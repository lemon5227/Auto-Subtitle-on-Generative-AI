"""
Public API for AI Subtitle Generator.

This module provides a clean interface for external projects to generate subtitles.
It abstracts away the internal implementation details and provides both sync and async APIs.

Usage:
    from ai_subtitle_generator.api import generate_subtitles
    
    # Simple usage
    result = generate_subtitles("https://www.youtube.com/watch?v=...")
    print(result['original_vtt'])
    
    # With translation
    result = generate_subtitles(
        "https://www.bilibili.com/video/BV...",
        translate_to="zh",
        model="base"
    )
    print(result['translated_vtt'])
"""
import os
import uuid
import threading
from typing import Dict, Optional, Literal
from dataclasses import dataclass, asdict

from .downloader import VideoDownloader
from .transcriber import SubtitleGenerator
from .translator import SubtitleTranslator
from .utils import format_time


# Job storage for async API
_jobs: Dict[str, dict] = {}
_jobs_lock = threading.Lock()


@dataclass
class SubtitleResult:
    """Result of subtitle generation."""
    video_id: str
    video_path: str
    original_vtt: str
    translated_vtt: Optional[str] = None
    source_language: Optional[str] = None
    target_language: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


def generate_subtitles(
    url: str,
    model: str = "base",
    source_lang: Optional[str] = None,
    translate_to: Optional[str] = None,
    use_faster_whisper: bool = False,
    output_dir: str = "./save",
    backend: Literal['local', 'openai'] = 'local',
    api_key: Optional[str] = None,
    progress_callback: Optional[callable] = None
) -> SubtitleResult:
    """
    Generate subtitles for a video URL (sync API).
    
    Args:
        url: Video URL (YouTube, Bilibili, or local file path)
        model: Whisper model size (tiny/base/small/medium/large)
        source_lang: Source language code (None for auto-detect)
        translate_to: Target language for translation (None to skip translation)
        use_faster_whisper: Use faster-whisper if available
        output_dir: Directory to save output files
        backend: 'local' for local models, 'openai' for OpenAI API
        api_key: API key (required if backend='openai')
        progress_callback: Optional callback(status_msg, percent)
        
    Returns:
        SubtitleResult with VTT content
        
    Example:
        result = generate_subtitles(
            "https://www.youtube.com/watch?v=abc123",
            model="base",
            translate_to="zh"
        )
        with open("subtitles.vtt", "w") as f:
            f.write(result.original_vtt)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Download video
    if progress_callback:
        progress_callback("Downloading video...", 5)
    
    downloader = VideoDownloader(output_dir=output_dir)
    
    # Check if it's a URL or local file
    if os.path.isfile(url):
        video_path = url
        video_id = os.path.splitext(os.path.basename(url))[0]
    else:
        video_id = downloader.get_video_id(url)
        video_path = downloader.download_video(url, video_id)
    
    if progress_callback:
        progress_callback("Download complete", 15)
    
    # 2. Transcribe
    if backend == 'openai' and api_key:
        # Use OpenAI API backend
        from .backends.openai_api import OpenAIWhisperBackend
        transcriber = OpenAIWhisperBackend(api_key=api_key, model=model)
        segments = transcriber.transcribe(video_path, language=source_lang, progress_callback=progress_callback)
        
        # Convert segments to VTT
        original_vtt = _segments_to_vtt(segments)
    else:
        # Use local Whisper
        generator = SubtitleGenerator()
        _, original_vtt = generator.generate_subtitles(
            video_path=video_path,
            model_name=model,
            use_faster=use_faster_whisper,
            language=source_lang,
            progress_callback=progress_callback
        )
    
    if progress_callback:
        progress_callback("Transcription complete", 80)
    
    # 3. Translate (optional)
    translated_vtt = None
    if translate_to:
        if progress_callback:
            progress_callback(f"Translating to {translate_to}...", 85)
        
        if backend == 'openai' and api_key:
            from .backends.openai_api import OpenAITranslationBackend
            translator_backend = OpenAITranslationBackend(api_key=api_key)
            translated_vtt = _translate_vtt(original_vtt, source_lang or 'en', translate_to, translator_backend)
        else:
            translator = SubtitleTranslator()
            translated_vtt = _translate_vtt_local(original_vtt, source_lang or 'en', translate_to, translator)
        
        if progress_callback:
            progress_callback("Translation complete", 95)
    
    if progress_callback:
        progress_callback("Done", 100)
    
    return SubtitleResult(
        video_id=video_id,
        video_path=video_path,
        original_vtt=original_vtt,
        translated_vtt=translated_vtt,
        source_language=source_lang,
        target_language=translate_to
    )


def start_subtitle_job(
    url: str,
    **kwargs
) -> str:
    """
    Start an async subtitle generation job.
    
    Args:
        url: Video URL
        **kwargs: Same as generate_subtitles
        
    Returns:
        job_id for tracking progress
        
    Example:
        job_id = start_subtitle_job("https://youtube.com/...")
        while True:
            status = get_job_status(job_id)
            if status['status'] == 'done':
                print(status['result']['original_vtt'])
                break
            time.sleep(2)
    """
    job_id = str(uuid.uuid4())
    
    with _jobs_lock:
        _jobs[job_id] = {
            'status': 'pending',
            'progress': 0,
            'message': 'Starting...',
            'result': None,
            'error': None
        }
    
    def run_job():
        try:
            def progress_cb(msg, pct):
                with _jobs_lock:
                    _jobs[job_id]['status'] = 'processing'
                    _jobs[job_id]['progress'] = pct
                    _jobs[job_id]['message'] = msg
            
            result = generate_subtitles(url, progress_callback=progress_cb, **kwargs)
            
            with _jobs_lock:
                _jobs[job_id]['status'] = 'done'
                _jobs[job_id]['progress'] = 100
                _jobs[job_id]['message'] = 'Complete'
                _jobs[job_id]['result'] = result.to_dict()
        except Exception as e:
            with _jobs_lock:
                _jobs[job_id]['status'] = 'error'
                _jobs[job_id]['error'] = str(e)
    
    thread = threading.Thread(target=run_job, daemon=True)
    thread.start()
    
    return job_id


def get_job_status(job_id: str) -> dict:
    """
    Get status of an async job.
    
    Args:
        job_id: Job ID from start_subtitle_job
        
    Returns:
        Dict with status, progress, message, result/error
    """
    with _jobs_lock:
        if job_id not in _jobs:
            return {'status': 'not_found', 'error': 'Job not found'}
        return dict(_jobs[job_id])


# --- Helper functions ---

def _segments_to_vtt(segments) -> str:
    """Convert TranscriptionSegment list to VTT string."""
    vtt = "WEBVTT\n\n"
    for i, seg in enumerate(segments, 1):
        start = format_time(seg.start)
        end = format_time(seg.end)
        vtt += f"{i}\n{start} --> {end}\n{seg.text.strip()}\n\n"
    return vtt


def _translate_vtt_local(vtt_content: str, source_lang: str, target_lang: str, translator: SubtitleTranslator) -> str:
    """Translate VTT content using local translator."""
    lines = vtt_content.split('\n')
    result = []
    
    for line in lines:
        # Keep non-text lines as-is (WEBVTT, timestamps, numbers, empty)
        if not line.strip() or line.startswith('WEBVTT') or '-->' in line or line.strip().isdigit():
            result.append(line)
        else:
            # Translate text lines
            translated = translator.translate_text(line, source_lang, target_lang)
            result.append(translated)
    
    return '\n'.join(result)


def _translate_vtt(vtt_content: str, source_lang: str, target_lang: str, backend) -> str:
    """Translate VTT content using a TranslationBackend."""
    lines = vtt_content.split('\n')
    result = []
    
    for line in lines:
        if not line.strip() or line.startswith('WEBVTT') or '-->' in line or line.strip().isdigit():
            result.append(line)
        else:
            translated = backend.translate(line, source_lang, target_lang)
            result.append(translated)
    
    return '\n'.join(result)


# Convenience exports
__all__ = [
    'generate_subtitles',
    'start_subtitle_job', 
    'get_job_status',
    'SubtitleResult'
]
