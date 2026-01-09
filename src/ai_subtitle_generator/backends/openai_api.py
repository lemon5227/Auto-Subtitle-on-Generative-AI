"""
OpenAI API backend implementations (placeholder for future use).
"""
from typing import List, Dict, Optional
from .base import TranscriptionBackend, TranslationBackend, TranscriptionSegment


class OpenAIWhisperBackend(TranscriptionBackend):
    """
    OpenAI Whisper API backend for cloud-based transcription.
    
    Requires: pip install openai
    """
    
    def __init__(self, api_key: str, model: str = "whisper-1"):
        """
        Initialize OpenAI Whisper API backend.
        
        Args:
            api_key: OpenAI API key
            model: Model name (default: whisper-1)
        """
        self.api_key = api_key
        self.model = model
        self._client = None
    
    def _get_client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("OpenAI package not installed. Run: pip install openai")
        return self._client
    
    def transcribe(
        self, 
        audio_path: str, 
        language: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> List[TranscriptionSegment]:
        """Transcribe using OpenAI Whisper API."""
        
        if progress_callback:
            progress_callback("Uploading to OpenAI API...", 10)
        
        client = self._get_client()
        
        with open(audio_path, 'rb') as audio_file:
            kwargs = {
                "model": self.model,
                "file": audio_file,
                "response_format": "verbose_json"
            }
            if language:
                kwargs["language"] = language
            
            response = client.audio.transcriptions.create(**kwargs)
        
        if progress_callback:
            progress_callback("Processing response...", 90)
        
        # Parse response segments
        segments = []
        for seg in getattr(response, 'segments', []):
            segments.append(TranscriptionSegment(
                start=seg.get('start', 0),
                end=seg.get('end', 0),
                text=seg.get('text', '')
            ))
        
        if progress_callback:
            progress_callback("Done", 100)
        
        return segments
    
    @property
    def name(self) -> str:
        return f"OpenAI Whisper API ({self.model})"


class OpenAITranslationBackend(TranslationBackend):
    """
    OpenAI GPT-based translation backend.
    
    Requires: pip install openai
    """
    
    SUPPORTED_PAIRS = [
        {"source": "en", "target": "zh", "name": "English to Chinese"},
        {"source": "zh", "target": "en", "name": "Chinese to English"},
        {"source": "en", "target": "ja", "name": "English to Japanese"},
        {"source": "en", "target": "ko", "name": "English to Korean"},
        {"source": "en", "target": "fr", "name": "English to French"},
        {"source": "en", "target": "de", "name": "English to German"},
        {"source": "en", "target": "es", "name": "English to Spanish"},
        # Add more as needed - GPT can translate to/from most languages
    ]
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """
        Initialize OpenAI translation backend.
        
        Args:
            api_key: OpenAI API key
            model: GPT model to use (default: gpt-4o-mini)
        """
        self.api_key = api_key
        self.model = model
        self._client = None
    
    def _get_client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("OpenAI package not installed. Run: pip install openai")
        return self._client
    
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate using OpenAI GPT."""
        client = self._get_client()
        
        lang_names = {
            'en': 'English', 'zh': 'Chinese', 'ja': 'Japanese',
            'ko': 'Korean', 'fr': 'French', 'de': 'German', 'es': 'Spanish'
        }
        
        src_name = lang_names.get(source_lang, source_lang)
        tgt_name = lang_names.get(target_lang, target_lang)
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": f"You are a professional translator. Translate the following {src_name} text to {tgt_name}. Output only the translation, nothing else."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
    
    def get_supported_pairs(self) -> List[Dict]:
        """Get supported translation pairs (GPT supports most pairs)."""
        return self.SUPPORTED_PAIRS
    
    @property
    def name(self) -> str:
        return f"OpenAI GPT Translation ({self.model})"
