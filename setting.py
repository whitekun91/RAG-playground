from dotenv import load_dotenv
import os

load_dotenv()

# VECTOR DB
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH")

# Huggingface model download token
HF_TOKEN = os.getenv('HF_TOKEN', '')

# RAG MODEL PATH
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "../models/embeddings/ko-sbert-sts")
RERANKER_MODEL_PATH = os.getenv("RERANKER_MODEL_PATH", "../models/embeddings/bge-reranker-v2-m3")

# STT MODEL (Local & OpenAI)
STT_MODEL_PATH = os.getenv('STT_MODEL_PATH', '../models/stt_models/whisper-large-v3-turbo')
OPENAI_STT_MODEL = os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")

# 로컬 / OpenAPI 선택
CHAT_PROVIDER_DEFAULT = os.getenv("CHAT_PROVIDER", "local").lower()  # local / openai
STT_PROVIDER_DEFAULT = os.getenv("STT_PROVIDER", "local").lower()  # local / openai
TTS_PROVIDER_DEFAULT = os.getenv("TTS_PROVIDER", "local").lower()  # local / openai

# OPenAI API 파라미터
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")

# 로컬 LLM 혹은 OpenAI API 파라미터
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "sk-samples")
VLLM_MODEL = os.getenv("VLLM_MODEL", "gemma-3-12b-it")
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.5"))
DEFAULT_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
DEFAULT_TOP_P = float(os.getenv("LLM_TOP_P", "0.95"))
DEFAULT_TIMEOUT = int(os.getenv("LLM_TIMEOUT_SEC", "45"))
DEFAULT_STOP = [os.getenv("LLM_STOP", "<end_of_turn>").split("|") if os.getenv("LLM_STOP") else ["<end_of_turn>"]]


# TTS MODEL
BARK_MODEL_PATH = os.getenv("BARK_MODEL_PATH", '../models/tts_models/bark')
BARK_VOICE_SEMANTIC_PROMPT = os.getenv("BARK_VOICE_SEMANTIC_PROMPT", '../models/tts_models/bark/speaker_embeddings/v2/ko_speaker_0_semantic_prompt.npy')
BARK_VOICE_COARSE_PROMPT = os.getenv("BARK_VOICE_COARSE_PROMPT", '../models/tts_models/bark/speaker_embeddings/v2/ko_speaker_0_coarse_prompt.npy')
BARK_VOICE_FINE_PROMPT = os.getenv("BARK_VOICE_FINE_PROMPT", '../models/tts_models/bark/speaker_embeddings/v2/ko_speaker_0_fine_prompt.npy')


# OPENAI TTS MODEL
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", 'gpt-4o-mini-tts')
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", 'alloy')
OPENAI_TTS_FORMAT = os.getenv("OPENAI_TTS_FORMAT", 'mp3')
