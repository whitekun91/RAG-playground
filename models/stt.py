import io, os
from typing import Union, Any, Dict
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from openai import OpenAI
from setting import (
    STT_MODEL_PATH, OPENAI_API_KEY, STT_PROVIDER_DEFAULT, OPENAI_STT_MODEL
)

_client = OpenAI(api_key=OPENAI_API_KEY)


def load_stt_model(device, torch_dtype, provider: str | None = None):
    selected_provider = (provider or STT_PROVIDER_DEFAULT).lower()

    if selected_provider == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다. (STT_PROVIDER=openai)")
        model = OPENAI_STT_MODEL or "gpt-4o-mini-transcribe"  # whisper-1 / gpt-4o-transcribe도 가능

        def stt(audio: Union[str, bytes, io.BufferedIOBase]) -> Dict[str, Any]:
            # audio: 경로/bytes/파일객체 모두 지원
            if isinstance(audio, (bytes, bytearray)):
                f = io.BytesIO(audio);
                f.name = "audio.webm";
                close_after = True
            elif isinstance(audio, str):
                f = open(audio, "rb");
                close_after = True
            else:
                f = audio;
                close_after = False

            try:
                resp = _client.audio.transcriptions.create(model=model, file=f)
                text = getattr(resp, "text", None) or str(resp)
                return {"text": text.strip() if isinstance(text, str) else text}
            finally:
                if close_after:
                    try:
                        f.close()
                    except:
                        pass

        return stt

    # ── Local (HF Whisper 등) ──
    stt_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        STT_MODEL_PATH, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    ).to(device)
    stt_processor = AutoProcessor.from_pretrained(STT_MODEL_PATH)
    stt_pipe = pipeline(
        "automatic-speech-recognition",
        model=stt_model,
        tokenizer=stt_processor.tokenizer,
        feature_extractor=stt_processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        generate_kwargs={"language": "ko", "task": "transcribe"},
    )

    def stt(audio):
        out = stt_pipe(audio)
        return {"text": out["text"]} if isinstance(out, dict) else out

    return stt
