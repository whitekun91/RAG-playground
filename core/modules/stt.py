import io
from typing import Union, Any, Dict

from openai import OpenAI

from setting import (
    OPENAI_API_KEY, STT_PROVIDER_DEFAULT, OPENAI_STT_MODEL
)


class STTManager:
    """
    A class to manage Speech-to-Text (STT) operations.
    Supports multiple providers, including OpenAI.
    """

    def __init__(self, provider: str | None = None):
        self.provider = (provider or STT_PROVIDER_DEFAULT).lower()
        self.client = OpenAI(api_key=OPENAI_API_KEY) if self.provider == "openai" else None
        self.model = OPENAI_STT_MODEL or "gpt-4o-mini-transcribe"

    def transcribe(self, audio: Union[str, bytes, io.BufferedIOBase]) -> Dict[str, Any]:
        """
        Perform speech-to-text transcription.

        Args:
            audio (Union[str, bytes, io.BufferedIOBase]): Audio input (file path, bytes, or file-like object).

        Returns:
            Dict[str, Any]: Transcription result.
        """
        if self.provider == "openai":
            if not OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY is not set. (STT_PROVIDER=openai)")

            # Handle different audio input types
            if isinstance(audio, (bytes, bytearray)):
                f = io.BytesIO(audio)
                f.name = "audio.webm"
                close_after = True
            elif isinstance(audio, str):
                f = open(audio, "rb")
                close_after = True
            else:
                f = audio
                close_after = False

            try:
                response = self.client.audio.transcriptions.create(model=self.model, file=f)
                text = getattr(response, "text", None) or str(response)
                return {"text": text.strip() if isinstance(text, str) else text}
            finally:
                if close_after:
                    try:
                        f.close()
                    except Exception:
                        pass
        else:
            raise NotImplementedError(f"Provider '{self.provider}' is not supported.")

# Example usage:
# stt_manager = STTManager(provider="openai")
# result = stt_manager.transcribe("path/to/audio/file.webm")
