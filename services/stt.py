from typing import Union, Any, Dict
import io


class STTService:
    def __init__(self, manager):
        self.manager = manager

    def transcribe(self, audio: Union[str, bytes, io.BufferedIOBase]) -> Dict[str, Any]:
        return self.manager.transcribe(audio)

    def __call__(self, audio: Union[str, bytes, io.BufferedIOBase]) -> Dict[str, Any]:
        return self.transcribe(audio)


