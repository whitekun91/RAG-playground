from typing import Optional, Dict, Any


class TTSService:
    def __init__(self, manager):
        self.manager = manager

    def synthesize(self, text: str, outfile_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        return self.manager.tts(text, outfile_path=outfile_path, **kwargs)

    def __call__(self, text: str, outfile_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        return self.synthesize(text, outfile_path=outfile_path, **kwargs)


