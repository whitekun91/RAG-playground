# 🧠 RAG-playground

A **FastAPI-based Retrieval-Augmented Generation (RAG)** template for hybrid local and cloud LLM inference.  
This project automatically downloads Hugging Face embedding models, builds Chroma vector stores, and performs local inference via **vLLM**.  
> 🔀 **Engine switch is manual**: choose between **local (vLLM)** and **openai** per request with a simple parameter (no automatic fallback).

---

## 🚀 Features

✅ **FastAPI backend** for RAG  
✅ **vLLM local inference** (e.g., Gemma, Mistral, Llama)  
✅ **Manual engine switch** — `engine: "local" | "openai"` per request  
✅ **Chroma vector database** for retrieval  
✅ **CrossEncoder reranking** for better context  
✅ **LangChain orchestration**  
✅ Clean, modular codebase ready for on-prem or hybrid

---

## 🏗️ System Architecture

```
📄 Documents (.pdf, .docx, .pptx, .xlsx)
       ↓ (text/image extraction)
🧩 LangChain + Chroma VectorDB
       ↓ (similarity search + reranking)
🧠 Inference Engine (manual): vLLM (local)  |  OpenAI API (cloud)
       ↓
🎯 Answer with Evidence
```

---

## 🧰 Tech Stack

| Component | Description |
|------------|-------------|
| **Backend** | FastAPI |
| **Vector DB** | Chroma |
| **Embeddings** | Hugging Face (e.g. `jhgan/ko-sbert-sts`) |
| **Reranker** | `bge-reranker-v2-m3` |
| **Local LLM** | vLLM (e.g. `gemma-3-12b-it`) |
| **Cloud LLM** | OpenAI GPT family |
| **Orchestration** | LangChain |
| **Hardware** | Multi-GPU (CUDA_VISIBLE_DEVICES=0,1) supported |

---

## 📦 Installation

### 1) Clone
```bash
git clone https://github.com/whitekun91/RAG-playground.git
cd RAG-playground
```

### 2) Virtual Env
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

### 3) Install
```bash
pip install -U pip
pip install -r requirements.txt
```

> ⚠️ Ensure your `torch`/`transformers` CUDA wheels match your system (e.g., cu121/cu126).

---

## ⚙️ Environment (`.env`)

```bash
# Embeddings & Reranker
EMB_MODEL_PATH=../models/embeddings/ko-sbert-sts
RERANKER_PATH=../models/reranker/bge-reranker-v2-m3

# Local LLM (vLLM OpenAI-compatible server)
VLLM_MODEL=gemma-3-12b-it
VLLM_URL=http://0.0.0.0:8000/v1/completions

# OpenAI (cloud)
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
OPENAI_MODEL=gpt-5
```
---

## ▶️ Run

### Start vLLM (Local)
```bash
CUDA_VISIBLE_DEVICES=0,1 
python -m vllm.entrypoints.openai.api_server \
    --model "./models/LM_Models/gemma-3-12b-it" \
    --served-model-name "gemma-3-12b-it" \
    --tokenizer "./models/LM_Models/gemma-3-12b-it" \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --chat-template-content-format auto \
    --gpu-memory-utilization 0.65 \
    --tensor-parallel-size 2 \
    --max-num-seqs 16 \
    --swap-space 8 \
    --enable-log-requests \
    --port 8000

```

### Start FastAPI
```bash
uvicorn main:app --host 0.0.0.0 --port 5001 --reload
```

---

## 💬 API — Manual Engine Switch

### Endpoint
`POST /ask`

### Request Body (common)
```json
{
  "question": "~~~~~~~",
  "engine": "local",
  "model": "gemma-3-12b-it",
  "top_p": 0.9,
  "temperature": 0.3,
  "max_tokens": 512
}
```

- `engine`: `"local"` | `"openai"` (required)
- `model`: For local, use the vLLM model name; for openai, use the OpenAI model name (if not entered, the default value from `.env` is used)
- Other sample parameters are optional

### cURL — Local (vLLM)
```bash
curl -X POST http://localhost:5001/ask   -H "Content-Type: application/json"   -d '{
    "question": "~~~~~~~~~~~",
    "engine": "local",
    "model": "gemma-3-12b-it",
    "temperature": 0.2,
    "top_p": 0.9,
    "max_tokens": 512
  }'
```

### cURL — OpenAI
```bash
curl -X POST http://localhost:5001/ask   -H "Content-Type: application/json"   -d '{
    "question": "~~~~~~~~~",
    "engine": "openai",
    "model": "gpt-5",
    "temperature": 0.2,
    "top_p": 0.9,
    "max_tokens": 512
  }'
```

### Response (example)
```json
{
  "answer": "~~~~~~~~~~",
  "evidence": [
    "documents/~~~~~~~~~~~~~.pdf"
  ],
  "engine": "local",
  "model_used": "gemma-3-12b-it"
}
```
---

## 🧩 Directory

```
RAG-playground/
│
├── app.py                           # FastAPI entry
├── settings.py                      # Paths & env
│
├── components/                      # Core components (image, vectorDB, split)
│   ├── image_load.py                # Image selector
│   ├── select_vectordb.py           # VectorDB select
│   ├── split_korean.py              # Korean text splitter
│
├── interface/                       # LLM/RAG chain interface
│   ├── create_chain.py              # LLM/RAG chains
│   ├── load_vector.py               # Chroma loaders (multi-DB)
│   ├── rag_reranker.py              # RAG Reranker
│
├── documents/                       # Source files
│   ├── vector_db/                   # Chroma (PDF)
│   ├── create_vectordb_by_pdf.py    # Build Chroma from unstructured data (PDF/image extraction)
│   ├── extracted_images/            # Extracted images from PDFs
│   ├── raw/                         # Raw document files
│
├── models/                          # Model files
│   ├── embeddings/                  # Embedding models
│   ├── LM_Models/                   # Local LLM models
│   ├── STT_Models/                  # Speech-to-text models
│   ├── TTS_Models/                  # Text-to-speech models
│   ├── llm.py                       # LLM logic
│   ├── stt.py                       # STT logic
│   ├── tts.py                       # TTS logic
│
├── prompts/                         # Prompt templates
│   ├── db_select_prompt.py          # VectorDB select prompt
│   ├── image_detect_prompt.py       # Image call prompt
│   ├── origin_prompt.py             # Base prompt
│
├── static/                          # Static files (frontend)
│   ├── index.html                   # Main HTML page
│   ├── style.css                    # CSS styles
│
├── outputs/                         # Output results
│
├── requirements.txt
├── .env
└── README.md
```

---

## 🧪 Notes

- Ubuntu 22.04 / CUDA 12.x / Python 3.10  
- vLLM 0.6+ / LangChain 0.2+  
- Multi-GPU and long-context friendly  
- Designed for **RAG** with verifiable evidence

---

## 📘 Roadmap


- [ ] OpenAI Whisper STT + Bark TTS (integrated voice input/output)
- [ ] Smart document routing (automatic classification of PDF / DOCX / PPTX)
- [ ] Hybrid Engine Enhancement  
  → Support hybrid RAG with selectable inference engines (Local vLLM / OpenAI API)  
  → Add simple “toggle” parameter or UI for engine switching  
  → Compare OpenAI model quality & latency for hybrid benchmarking
- [ ] Add multi-turn memory module (context retention based on conversation)
- [ ] Add retrieval-evidence visualization on frontend (document evidence visualization)

---

## 🧑‍💻 Author

**Minwoo Baek  **  
Senior AI Engineer | PhD Candidate (Big Data Applications)  
💼 Hanwha Momentum / AI Manufacturing R&D  
📧 minwoo713@gmail.com  
🔗 LinkedIn: https://www.linkedin.com/in/bjh713/

---

## 📜 License

MIT
