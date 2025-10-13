import numpy as np
from transformers import (
    BarkModel,
    AutoProcessor as BarkProcessor,
)

from settings import (
    BARK_MODEL_PATH,
    BARK_VOICE_SEMANTIC_PROMPT,
    BARK_VOICE_COARSE_PROMPT,
    BARK_VOICE_FINE_PROMPT,
)


def load_tts_model(device, torch):
    # TTS - Bark 모델
    tts_processor = BarkProcessor.from_pretrained(BARK_MODEL_PATH)
    tts_model = BarkModel.from_pretrained(BARK_MODEL_PATH, torch_dtype=torch.float16).to(device)
    tts_history_prompt = {
        "semantic_prompt": torch.tensor(np.load(BARK_VOICE_SEMANTIC_PROMPT)).to(device),
        "coarse_prompt": torch.tensor(np.load(BARK_VOICE_COARSE_PROMPT)).to(device),
        "fine_prompt": torch.tensor(np.load(BARK_VOICE_FINE_PROMPT)).to(device),
    }
    return tts_processor, tts_model, tts_history_prompt
