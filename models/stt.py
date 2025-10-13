from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)

from settings import STT_MODEL_PATH


def load_stt_model(device, torch_dtype):
    # STT
    stt_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        STT_MODEL_PATH,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    ).to(device)
    stt_processor = AutoProcessor.from_pretrained(STT_MODEL_PATH)
    stt_pipe = pipeline(
        "automatic-speech-recognition",
        model=stt_model,
        tokenizer=stt_processor.tokenizer,
        feature_extractor=stt_processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        generate_kwargs={"language": "ko", "task": "transcribe"}
    )

    return stt_pipe
