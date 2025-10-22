import os
from uuid import uuid4
from ..rag.query_service import QueryService

class PipelineService:
    def __init__(self, query_service, stt_manager, tts_manager, tts_provider_default):
        self.query_service = query_service
        self.stt_manager = stt_manager
        self.tts_manager = tts_manager
        self.tts_provider_default = tts_provider_default
    
    def ask_text(self, question_text: str, return_audio: bool = False):
        """Handle text query"""
        result = self.query_service.retrieve_and_rerank(question_text)
        
        if return_audio:
            # Generate TTS audio
            os.makedirs("outputs", exist_ok=True)
            filename = f"{uuid4().hex}.wav"
            output_path = os.path.join("outputs", filename)
            
            self.tts_manager.synthesize(result["rag_answer"], output_path)
            result["download_url"] = f"/download/{filename}"
        
        return result
    
    def ask_audio(self, audio_path: str, return_audio: bool = False):
        """Handle audio query"""
        # Transcribe audio to text
        stt_result = self.stt_manager.transcribe(audio_path)
        question_text = stt_result["text"]
        
        # Process as text query
        result = self.ask_text(question_text, return_audio)
        result["stt_text"] = question_text
        
        return result
