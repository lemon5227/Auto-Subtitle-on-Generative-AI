"""
Abstract base classes for transcription and translation backends.
Allows swapping between local models and cloud APIs.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class TranscriptionSegment:
    """A single transcription segment with timing info."""
    start: float  # seconds
    end: float    # seconds
    text: str


class TranscriptionBackend(ABC):
    """Abstract interface for speech-to-text backends."""
    
    @abstractmethod
    def transcribe(
        self, 
        audio_path: str, 
        language: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> List[TranscriptionSegment]:
        """
        Transcribe audio file to text segments.
        
        Args:
            audio_path: Path to audio/video file
            language: Language code (e.g., 'en', 'zh') or None for auto-detect
            progress_callback: Optional callback(status_msg, percent)
            
        Returns:
            List of TranscriptionSegment objects
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this backend."""
        pass
    
    @property
    def supports_streaming(self) -> bool:
        """Whether this backend supports streaming transcription."""
        return False


class TranslationBackend(ABC):
    """Abstract interface for text translation backends."""
    
    @abstractmethod
    def translate(
        self, 
        text: str, 
        source_lang: str, 
        target_lang: str
    ) -> str:
        """
        Translate text from source to target language.
        
        Args:
            text: Text to translate
            source_lang: Source language code (e.g., 'en')
            target_lang: Target language code (e.g., 'zh')
            
        Returns:
            Translated text
        """
        pass
    
    @abstractmethod
    def get_supported_pairs(self) -> List[Dict]:
        """
        Get list of supported translation pairs.
        
        Returns:
            List of dicts with 'source', 'target', 'name' keys
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this backend."""
        pass
