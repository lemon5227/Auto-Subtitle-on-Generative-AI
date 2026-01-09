"""
Local backend implementations using Whisper and transformers.
Wraps existing transcriber.py and translator.py without modifying them.
"""
from typing import List, Dict, Optional
from .base import TranscriptionBackend, TranslationBackend, TranscriptionSegment

# Import existing implementations
from ..transcriber import SubtitleGenerator
from ..translator import SubtitleTranslator, SUPPORTED_TRANSLATION_PAIRS


class LocalWhisperBackend(TranscriptionBackend):
    """
    Local Whisper-based transcription backend.
    Uses existing SubtitleGenerator implementation.
    """
    
    def __init__(self, model_name: str = "base", use_faster: bool = False, device: str = None):
        """
        Initialize local Whisper backend.
        
        Args:
            model_name: Whisper model size (tiny/base/small/medium/large)
            use_faster: Use faster-whisper if available
            device: Device to use (cuda/cpu/mps) or None for auto-detect
        """
        self.model_name = model_name
        self.use_faster = use_faster
        self._generator = SubtitleGenerator(device=device)
    
    def transcribe(
        self, 
        audio_path: str, 
        language: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> List[TranscriptionSegment]:
        """Transcribe using local Whisper model."""
        
        # Use existing generator, get VTT content
        _, vtt_content = self._generator.generate_subtitles(
            video_path=audio_path,
            model_name=self.model_name,
            use_faster=self.use_faster,
            language=language,
            progress_callback=progress_callback
        )
        
        # Parse VTT to segments
        return self._parse_vtt_to_segments(vtt_content)
    
    def _parse_vtt_to_segments(self, vtt_content: str) -> List[TranscriptionSegment]:
        """Parse VTT content to TranscriptionSegment list."""
        segments = []
        lines = vtt_content.strip().split('\n')
        
        i = 0
        # Skip WEBVTT header
        if lines and lines[0].startswith('WEBVTT'):
            i = 1
        
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            
            # Skip numeric cue ID
            if line.isdigit():
                i += 1
                continue
            
            # Check for timestamp line
            if '-->' in line:
                times = line.split('-->')
                if len(times) == 2:
                    start = self._parse_time(times[0].strip())
                    end = self._parse_time(times[1].strip())
                    
                    # Collect text lines
                    i += 1
                    text_lines = []
                    while i < len(lines) and lines[i].strip() and '-->' not in lines[i]:
                        text_lines.append(lines[i].strip())
                        i += 1
                    
                    if text_lines:
                        segments.append(TranscriptionSegment(
                            start=start,
                            end=end,
                            text='\n'.join(text_lines)
                        ))
                    continue
            
            i += 1
        
        return segments
    
    def _parse_time(self, time_str: str) -> float:
        """Parse VTT time string to seconds."""
        parts = time_str.replace(',', '.').split(':')
        if len(parts) == 3:
            h, m, s = parts
            return float(h) * 3600 + float(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return float(m) * 60 + float(s)
        return float(time_str)
    
    @property
    def name(self) -> str:
        return f"Local Whisper ({self.model_name})"


class LocalTranslationBackend(TranslationBackend):
    """
    Local translation backend using Helsinki-NLP models.
    Uses existing SubtitleTranslator implementation.
    """
    
    def __init__(self, device: str = None):
        """
        Initialize local translation backend.
        
        Args:
            device: Device to use (cuda/cpu/mps) or None for auto-detect
        """
        self._translator = SubtitleTranslator(device=device)
    
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate using local Helsinki-NLP models."""
        return self._translator.translate_text(text, source_lang, target_lang)
    
    def get_supported_pairs(self) -> List[Dict]:
        """Get supported translation pairs."""
        return SUPPORTED_TRANSLATION_PAIRS
    
    @property
    def name(self) -> str:
        return "Local Helsinki-NLP"
