"""
Backend factory for creating transcription and translation backends.
"""
from typing import Optional, Literal
from .base import TranscriptionBackend, TranslationBackend


BackendType = Literal['local', 'openai']


def create_transcription_backend(
    backend: BackendType = 'local',
    model_name: str = 'base',
    use_faster: bool = False,
    device: str = None,
    api_key: str = None,
    **kwargs
) -> TranscriptionBackend:
    """
    Factory function to create a transcription backend.
    
    Args:
        backend: 'local' for Whisper, 'openai' for OpenAI API
        model_name: Model name/size
        use_faster: Use faster-whisper (local only)
        device: Device to use (local only)
        api_key: API key (openai only)
        
    Returns:
        TranscriptionBackend instance
    """
    if backend == 'local':
        from .local import LocalWhisperBackend
        return LocalWhisperBackend(
            model_name=model_name,
            use_faster=use_faster,
            device=device
        )
    elif backend == 'openai':
        if not api_key:
            raise ValueError("api_key required for OpenAI backend")
        from .openai_api import OpenAIWhisperBackend
        return OpenAIWhisperBackend(api_key=api_key, model=model_name)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def create_translation_backend(
    backend: BackendType = 'local',
    device: str = None,
    api_key: str = None,
    model_name: str = None,
    **kwargs
) -> TranslationBackend:
    """
    Factory function to create a translation backend.
    
    Args:
        backend: 'local' for Helsinki-NLP, 'openai' for GPT
        device: Device to use (local only)
        api_key: API key (openai only)
        model_name: Model name (openai only)
        
    Returns:
        TranslationBackend instance
    """
    if backend == 'local':
        from .local import LocalTranslationBackend
        return LocalTranslationBackend(device=device)
    elif backend == 'openai':
        if not api_key:
            raise ValueError("api_key required for OpenAI backend")
        from .openai_api import OpenAITranslationBackend
        return OpenAITranslationBackend(
            api_key=api_key,
            model=model_name or "gpt-4o-mini"
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")
