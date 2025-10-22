# 🧠 RAG Playground

A **FastAPI-based Retrieval-Augmented Generation (RAG)** system with modern UI and comprehensive AI services.  
This project provides a complete RAG solution with STT, TTS, and document processing capabilities using both local and cloud AI models.

---

## 🚀 Features

✅ **Modern Web Interface** - Beautiful, responsive UI with glassmorphism design  
✅ **FastAPI Backend** - High-performance API with comprehensive endpoints  
✅ **RAG System** - Document retrieval and question answering  
✅ **Speech-to-Text (STT)** - Voice input with Whisper support  
✅ **Text-to-Speech (TTS)** - Audio output with multiple providers  
✅ **Multi-modal Support** - Text, audio, and image processing  
✅ **Vector Database** - ChromaDB with embedding models  
✅ **Document Reranking** - CrossEncoder for improved accuracy  
✅ **Clean Architecture** - Modular services and core separation  
✅ **SSL Support** - HTTPS with custom certificates  
✅ **Virtual Environment** - Python venv for dependency management  

---

## 🏗️ System Architecture

```
📄 Documents (.pdf, .docx, .pptx, .xlsx)
       ↓ (text/image extraction)
🧩 Chroma VectorDB + CrossEncoder Reranking
       ↓ (similarity search + reranking)
🧠 LLM Engine: Local (vLLM) | Cloud (OpenAI)
       ↓
🎯 Answer with Evidence
       ↓ (optional)
🎤 STT (Audio Input) / 🔊 TTS (Audio Output)
```

---

## 🧰 Tech Stack

| Component | Description |
|-----------|-------------|
| **Backend** | FastAPI with uvicorn |
| **Frontend** | Modern HTML5 + CSS3 with glassmorphism |
| **Vector DB** | ChromaDB |
| **Embeddings** | Hugging Face (ko-sbert-sts) |
| **Reranker** | bge-reranker-v2-m3 |
| **Local LLM** | vLLM (gemma-3-12b-it) |
| **Cloud LLM** | OpenAI GPT family |
| **STT** | Whisper (local) / OpenAI Whisper API |
| **TTS** | Bark (local) / OpenAI TTS API |
| **Architecture** | Services/Core separation |
| **Virtual Env** | Python venv |

---

## 📦 Installation

### 1) Clone Repository
```bash
git clone https://github.com/whitekun91/RAG-playground.git
cd RAG-playground
```

### 2) Create Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

### 3) Install Dependencies
```bash
pip install -U pip
pip install -r requirements.txt
```

### 4) Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your settings
# Required: Set your API keys and model paths
```

---

## ⚙️ Environment Configuration (`.env`)

```bash
# VectorDB 경로
VECTOR_DB_PATH='./documents/vector_db/'

# AI 모델 경로
EMBEDDING_MODEL_PATH='./models/embeddings/ko-sbert-sts'
RERANKER_MODEL_PATH='./models/embeddings/bge-reranker-v2-m3'

# 허깅페이스 토큰 (필요시)
# HF_TOKEN = "your_huggingface_token_here"

# 텍스트 토크나이징 하이퍼파라미터
CHUNK_SIZE=600
CHUNK_OVERLAP=0

# AI 서비스 제공자 (local / openai)
STT_PROVIDER=openai
CHAT_PROVIDER=openai
TTS_PROVIDER=openai

# 로컬 LLM 설정
LM_MODEL_PATH='./models/LM_Models/gemma-3-12b-it'
VLLM_BASE_URL=http://localhost:8000
VLLM_MODEL=gemma-3-12b-it
VLLM_API_KEY=sk-samples

# OpenAI API 설정 (필요시)
# OPENAI_API_KEY='your_openai_api_key_here'
OPENAI_MODEL=gpt-4.1

# LLM 하이퍼파라미터
LLM_TEMPERATURE=0.5
LLM_MAX_TOKENS=1024
LLM_TOP_P=0.95
LLM_STOP=<end_of_turn>

# TTS 설정
BARK_MODEL_PATH='./models/TTS_Models/bark'
BARK_VOICE_SEMANTIC_PROMPT='./models/TTS_Models/bark/speaker_embeddings/v2/ko_speaker_0_semantic_prompt.npy'
BARK_VOICE_COARSE_PROMPT='./models/TTS_Models/bark/speaker_embeddings/v2/ko_speaker_0_coarse_prompt.npy'
BARK_VOICE_FINE_PROMPT='./models/TTS_Models/bark/speaker_embeddings/v2/ko_speaker_0_fine_prompt.npy'

# STT 모델 경로
STT_MODEL_PATH='./models/STT_Models/whisper-large-v3-turbo'

# OpenAI STT/TTS 모델
OPENAI_STT_MODEL=gpt-4o-mini-transcribe
OPENAI_TTS_MODEL=gpt-4o-mini-tts
OPENAI_TTS_VOICE=alloy
OPENAI_TTS_FORMAT=mp3
```

---

## ▶️ Quick Start

### 1) Start vLLM Server (Optional - for local LLM)
```bash
# GPU 설정 (다중 GPU 사용 시)
export CUDA_VISIBLE_DEVICES=0,1

# vLLM 서버 시작
python -m vllm.entrypoints.openai.api_server \
    --model "your-model-path" \
    --served-model-name "gemma-3-12b-it" \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.65 \
    --tensor-parallel-size 2 \
    --max-num-seqs 16 \
    --swap-space 8 \
    --enable-log-requests \
    --port 8000
```

### 2) Start RAG Playground Server
```bash
# 가상환경 활성화 (이미 활성화된 경우 생략)
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 서버 시작
python run_main.py --reload

# SSL로 실행 (HTTPS)
python run_main.py --ssl --reload

# 포트 변경
python run_main.py --port 8080 --reload

# 환경 확인만
python run_main.py --check-only
```

### 3) Access Web Interface
- **HTTP**: http://localhost:5001
- **HTTPS**: https://localhost:5001 (SSL 인증서 필요)

---

## 💬 API Endpoints

### 🎯 Main Endpoints

#### `POST /ask-text` - Text Query
```json
{
  "question": "What is the main topic of the document?",
  "return_audio": false
}
```

#### `POST /ask-audio-tts` - Audio Query with TTS
```json
{
  "file": "audio_file.webm",
  "return_audio": true
}
```

### 📊 System Endpoints

#### `GET /` - Web Interface
Returns the modern HTML interface with glassmorphism design

#### `GET /status` - System Status
```json
{
  "status": "ok",
  "gpu_available": true,
  "device": "NVIDIA GeForce RTX 4090",
  "memory_allocated_MB": 2048,
  "memory_reserved_MB": 4096,
  "uptime_sec": 3600
}
```

#### `GET /metrics` - Performance Metrics
Returns Prometheus-style metrics

#### `GET /download/{filename}` - Download Audio
Download generated TTS audio files

#### `GET /pdf-images/{pdf_name}` - PDF Images
Get extracted images from PDF documents

### 🔧 Example Usage

#### Text Query
```bash
curl -X POST http://localhost:5001/ask-text \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the main topic?",
    "return_audio": false
  }'
```

#### Audio Query with TTS Response
```bash
curl -X POST http://localhost:5001/ask-audio-tts \
  -F "file=@audio.wav" \
  -F "return_audio=true"
```

### 📝 Response Format
```json
{
  "question": "What is the main topic?",
  "rag_answer": "The main topic is...",
  "download_url": "/download/audio_file.wav",
  "image_urls": ["/images/doc1/page1.png"],
  "elapsed_time": "2.5s"
}
```

---

## 🧩 Directory Structure

```
RAG-playground/
│
├── main.py                          # FastAPI entry point
├── run_main.py                      # Server startup script
├── setting.py                       # Configuration & environment
│
├── core/                            # Core business logic
│   ├── components/                  # Core components
│   │   ├── image_loader.py          # Image processing
│   │   ├── vector_selector.py       # Vector DB selection
│   │   └── korean_splitter.py       # Korean text splitting
│   ├── interface/                   # Interface layer
│   │   └── prompts.py               # Prompt templates
│   ├── models/                      # Model files & utilities
│   │   ├── embeddings/              # Embedding models
│   │   ├── stt_models/              # Speech-to-text models
│   │   ├── tts_models/              # Text-to-speech models
│   │   └── model_downloader.py      # Model download utility
│   ├── prompts/                     # Prompt templates
│   │   ├── db_select_prompt.py      # DB selection prompts
│   │   ├── image_detect_prompt.py   # Image detection prompts
│   │   └── origin_prompt.py         # Base RAG prompts
│   └── utils/                       # Utility functions
│       ├── config.py                # Configuration management
│       ├── errors.py                # Error handling
│       ├── metrics.py               # Performance metrics
│       └── observability.py         # Logging & monitoring
│
├── services/                        # Service layer (client-core bridge)
│   ├── llm_core.py                  # LLM core functions
│   ├── stt_core.py                  # STT core functions
│   ├── tts_core.py                  # TTS core functions
│   ├── rag/                         # RAG services
│   │   ├── chains.py                # Chain services
│   │   ├── vector_store.py          # Vector store services
│   │   ├── reranker.py              # Document reranking
│   │   ├── query_service.py         # Query processing
│   │   ├── composer.py              # RAG response composer
│   │   ├── image_builder.py         # Image link builder
│   │   └── db_classifier.py         # Database classification
│   ├── pipeline/                    # Pipeline orchestration
│   │   └── orchestrator.py          # Main pipeline orchestrator
│   └── providers/                   # Provider managers
│       ├── llm_manager.py           # LLM provider management
│       ├── stt_manager.py           # STT provider management
│       └── tts_manager.py           # TTS provider management
│
├── documents/                       # Document storage
│   ├── vector_db/                   # Chroma vector database
│   ├── extracted_images/            # Extracted images from PDFs
│   └── raw/                         # Raw document files
│
├── static/                          # Frontend files
│   ├── index.html                   # Modern HTML interface
│   └── style.css                    # Glassmorphism CSS styles
│
├── outputs/                         # Generated outputs
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment template
├── .env                            # Environment variables (create from .env.example)
├── key.pem                         # SSL private key (for HTTPS)
├── cert.pem                        # SSL certificate (for HTTPS)
└── README.md                       # This file
```

---

## 🎨 UI Features

- **Modern Design** - Glassmorphism with gradient backgrounds
- **Responsive Layout** - Mobile and desktop optimized
- **Interactive Elements** - Smooth animations and transitions
- **Voice Input** - Real-time speech recognition
- **Audio Output** - Text-to-speech with play controls
- **Image Gallery** - Document image display with modal view
- **Real-time Chat** - Live conversation interface

---

## 🧪 System Requirements

- **Python**: 3.9+
- **OS**: Windows 10+, Ubuntu 20.04+, macOS 10.15+
- **GPU**: NVIDIA GPU with CUDA support (optional)
- **RAM**: 8GB+ recommended
- **Storage**: 10GB+ for models and documents

---

## 📘 Roadmap

### ✅ Completed Features
- [x] **Modern UI** - Glassmorphism design with responsive layout
- [x] **STT/TTS Integration** - Whisper STT + Bark/OpenAI TTS
- [x] **Clean Architecture** - Services/Core separation
- [x] **Pipeline Orchestration** - Complex workflow management
- [x] **Multi-modal Support** - Text, audio, and image processing
- [x] **SSL Support** - HTTPS with custom certificates
- [x] **Virtual Environment** - Python venv management

### 🚧 In Progress
- [ ] **Performance Optimization** - GPU memory management
- [ ] **Error Handling** - Robust error recovery
- [ ] **Document Upload** - Web interface for document management

### 🔮 Future Features
- [ ] **Multi-turn Memory** - Conversation context retention
- [ ] **Smart Document Routing** - Automatic PDF/DOCX/PPTX classification
- [ ] **Evidence Visualization** - Enhanced document evidence display
- [ ] **Real-time Streaming** - Live audio processing
- [ ] **Multi-language Support** - Internationalization
- [ ] **User Authentication** - Login and user management
- [ ] **API Rate Limiting** - Request throttling and quotas

---

## 🧑‍💻 Author

**Minwoo Baek**  
Senior AI Engineer | PhD Candidate (Big Data Applications)  
💼 Hanwha Momentum / AI Manufacturing R&D  
📧 minwoo713@gmail.com  
🔗 LinkedIn: https://www.linkedin.com/in/bjh713/

---

## 📜 License

MIT