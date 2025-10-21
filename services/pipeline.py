import os
from typing import Dict, Any


class PipelineService:
    def __init__(self, query_service, stt_service, tts_service, tts_provider_default: str):
        self.query_service = query_service
        self.stt_service = stt_service
        self.tts_service = tts_service
        self.tts_provider_default = tts_provider_default

    def ask_text(self, text: str, return_audio: bool = False) -> Dict[str, Any]:
        result = self.query_service.handle(text)
        response: Dict[str, Any] = {
            "question": text,
            "rag_answer": result.get("answer_text"),
            "image_urls": result.get("image_urls"),
            "download_url": None,
        }
        if return_audio:
            response["download_url"] = self._tts_to_file(result.get("answer_text"))
        return response

    def ask_audio(self, audio_path: str, return_audio: bool = False) -> Dict[str, Any]:
        stt = self.stt_service(audio_path)
        text = stt.get("text", "") if isinstance(stt, dict) else ""
        answer = self.ask_text(text, return_audio=return_audio)
        answer["stt_text"] = text
        return answer

    def _tts_to_file(self, text: str | None) -> str | None:
        if not text:
            return None
        from uuid import uuid4
        provider = (os.getenv("TTS_PROVIDER", self.tts_provider_default) or "local").lower()
        audio_format = (os.getenv("OPENAI_TTS_FORMAT", "mp3").lower()
                        if provider == "openai" else "wav")

        os.makedirs("outputs", exist_ok=True)
        filename = f"{uuid4().hex}.{audio_format}"
        output_path = os.path.join("outputs", filename)

        try:
            self.tts_service(text, outfile_path=output_path, instructions="Speak in a clear, friendly tone.")
            import os as _os
            if _os.path.isfile(output_path) and _os.path.getsize(output_path) > 0:
                return f"/download/{filename}"
        except Exception:
            return None
        return None


