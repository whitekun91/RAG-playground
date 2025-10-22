import torch
from ..tts_core import load_tts_model

class TTSManager:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = load_tts_model(device=device, torch=torch)
    
    def synthesize(self, text: str, output_path: str = None, **kwargs):
        return self.model(text, outfile_path=output_path, **kwargs)

def create_tts_manager():
    return TTSManager()
