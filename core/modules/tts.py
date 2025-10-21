import os
from typing import Optional, Dict, Any

import numpy as np
import soundfile as sf
from openai import OpenAI
from transformers import BarkModel, AutoProcessor as BarkProcessor

from setting import (
    BARK_MODEL_PATH,
    BARK_VOICE_SEMANTIC_PROMPT,
    BARK_VOICE_COARSE_PROMPT,
    BARK_VOICE_FINE_PROMPT,
    TTS_PROVIDER_DEFAULT,
    OPENAI_API_KEY,
    OPENAI_TTS_VOICE,
)


class TTSManager:
    """
    A class to manage Text-to-Speech (TTS) operations.
    Supports multiple providers, including OpenAI.
    """

    def __init__(self, provider: Optional[str] = None):
        self.provider = (provider or TTS_PROVIDER_DEFAULT or "local").lower()
        self.client = OpenAI(api_key=OPENAI_API_KEY) if self.provider == "openai" else None
        self.default_voice = (OPENAI_TTS_VOICE or "alloy").strip()
        self.default_format = os.getenv("OPENAI_TTS_FORMAT", "mp3").lower()

    def tts(self, text: str, outfile_path: Optional[str] = None, format: Optional[str] = None, voice: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Perform text-to-speech conversion.

        Args:
            text (str): The text to convert to speech.
            outfile_path (Optional[str]): Path to save the output audio file.
            format (Optional[str]): Audio format (e.g., mp3, wav).
            voice (Optional[str]): Voice model to use.

        Returns:
            Dict[str, Any]: Metadata about the generated audio.
        """
        if self.provider == "openai":
            if not OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY is not set. (TTS_PROVIDER=openai)")

            fmt = (format or self.default_format).lower()
            v = (voice or self.default_voice)

            # Streaming TTS (latest SDK style)
            if outfile_path:
                with self.client.audio.speech.with_streaming_response.create(
                    model=model,
                    voice=v,
                    input=text,
                    # format=fmt,  # Removed: not supported by OpenAI SDK
                    **kwargs,
                ) as resp:
                    resp.stream_to_file(outfile_path)
                return {"path": outfile_path, "bytes": None, "format": fmt, "sample_rate": None}

            # If you want to receive audio in memory: stream to a temp file, then read and return
            from tempfile import NamedTemporaryFile
            with NamedTemporaryFile(delete=False, suffix=f".{fmt}") as tmpf:
                tmp_path = tmpf.name
            try:
                with self.client.audio.speech.with_streaming_response.create(
                    model=model,
                    voice=v,
                    input=text,
                    # format=fmt,  # Removed: not supported by OpenAI SDK
                    **kwargs,
                ) as resp:
                    resp.stream_to_file(tmp_path)
                with open(tmp_path, "rb") as f:
                    data = f.read()
                return {"path": None, "bytes": data, "format": fmt, "sample_rate": None}
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        # ───────────────────── Local Bark ─────────────────────
        tts_processor = BarkProcessor.from_pretrained(BARK_MODEL_PATH)
        tts_model = BarkModel.from_pretrained(BARK_MODEL_PATH, torch_dtype=torch.float16).to(device)
        hist = {
            "semantic_prompt": torch.tensor(np.load(BARK_VOICE_SEMANTIC_PROMPT)).to(device),
            "coarse_prompt":   torch.tensor(np.load(BARK_VOICE_COARSE_PROMPT)).to(device),
            "fine_prompt":     torch.tensor(np.load(BARK_VOICE_FINE_PROMPT)).to(device),
        }
        SAMPLE_RATE = 24_000

        def tts(text: str, outfile_path: Optional[str] = None, format: str = "wav", **kwargs) -> Dict[str, Any]:
            # Prepare input for Bark TTS
            inputs = tts_processor(text, return_tensors="pt").to(device)
            with torch.no_grad():
                audio_pt = tts_model.generate(
                    **inputs,
                    semantic_history_prompt=hist["semantic_prompt"],
                    coarse_history_prompt=hist["coarse_prompt"],
                    fine_history_prompt=hist["fine_prompt"],
                )
            audio = audio_pt.squeeze().detach().cpu().numpy().astype(np.float32)

            # Save to file if outfile_path is provided
            if outfile_path:
                sf.write(outfile_path, audio, SAMPLE_RATE)
                return {"path": outfile_path, "bytes": None, "format": "wav", "sample_rate": SAMPLE_RATE}

            # Otherwise, return audio as bytes
            import io as _io
            buf = _io.BytesIO()
            sf.write(buf, audio, SAMPLE_RATE, format="WAV")
            return {"path": None, "bytes": buf.getvalue(), "format": "wav", "sample_rate": SAMPLE_RATE}

        return tts
