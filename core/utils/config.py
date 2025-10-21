import os
from dataclasses import dataclass


@dataclass
class AppConfig:
    tts_provider_default: str
    openai_tts_format: str

    @classmethod
    def from_env(cls, tts_provider_default: str):
        return cls(
            tts_provider_default=tts_provider_default,
            openai_tts_format=os.getenv("OPENAI_TTS_FORMAT", "mp3").lower(),
        )


