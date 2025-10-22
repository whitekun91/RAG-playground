import torch
from ..stt_core import load_stt_model

class STTManager:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = load_stt_model(device=device, torch_dtype=torch_dtype)
    
    def transcribe(self, audio_path: str):
        return self.model(audio_path)

def create_stt_manager():
    return STTManager()
