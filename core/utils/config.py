import os
from typing import Optional

class AppConfig:
    def __init__(self, tts_provider: str = "local"):
        self.tts_provider = tts_provider
    
    @classmethod
    def from_env(cls, tts_provider_default: str = "local"):
        tts_provider = os.getenv("TTS_PROVIDER", tts_provider_default)
        return cls(tts_provider=tts_provider)
