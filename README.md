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
🎯 Answer with Evidence (문서/도면/로그 근거 포함)
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
python -m vllm.entrypoints.openai.api_server   
   --model ../models/LM_Models/gemma-3-12b-it  
   --dtype bfloat16  
   --max-model-len 4096
```

### Start FastAPI
```bash
uvicorn app:app --host 0.0.0.0 --port 5001 --reload
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
- `model`: local일 때는 vLLM 모델명, openai일 때는 OpenAI 모델명(미입력 시 `.env` 기본값 사용)
- 기타 샘플 파라미터는 선택

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

> 📌 URS 준수: “사용자가 질문하면 문서/도면/로그/도메인 지식 근거와 함께 답변 제공”을 위해 `evidence` 필드를 포함합니다.

---

## 🧩 Directory

```
RAG-playground/
│
├── app.py                           # FastAPI entry
├── settings.py                      # Paths & env
│
├── components
├── components/image_load.py         # Image select[interface](..%2FManufacturing_LLM%2FRAG%2Finterface)
├── components/select_vectordb.py    # VectorDB select
├── components/split_korean.py       # Korean split
│
├── interface
├── interface/create_chain.py        # LLM/RAG chains
├── interface/load_vector.py         # Chroma loaders (multi-DB)
├── interface/rag_reranker.py        # RAG Reranker
│
├── documents/                       # Source files
├── documents/vector_db/             # Chroma (PDF)
├── documents/create_vectordb.py     # Build Chroma from Unstructure data (Include PDF image extraction)
│
├── models/                        
├── models/llm                       # Local LLM/OpenAI
├── models/stt                       # STT
├── models/tts                       # TTS
│
├── prompts/                        
├── prompts/db_select_prompt.py      # VectorDB select               
├── prompts/image_detect_prompt.py   # Image call                 
├── prompts/origin_prompt.py         # Base prompt           
│
├── requirements.txt
└── .env
```

---

## 🧪 Notes

- Ubuntu 22.04 / CUDA 12.x / Python 3.10  
- vLLM 0.6+ / LangChain 0.2+  
- Multi-GPU and long-context friendly  
- Designed for **RAG** with verifiable evidence

---

## 📘 Roadmap


- [ ] OpenAI Whisper STT + Bark TTS (음성 입출력 통합)
- [ ] Smart document routing (PDF / DOCX / PPTX 자동 분류)
- [ ] Hybrid Engine Enhancement  
  → Support hybrid RAG with selectable inference engines (Local vLLM / OpenAI API)  
  → Add simple “toggle” parameter or UI for engine switching  
  → Compare OpenAI model quality & latency for hybrid benchmarking
- [ ] Add multi-turn memory module (대화 기반 문맥 유지)
- [ ] Add retrieval-evidence visualization on frontend (문서 근거 시각화)

---

## 🧑‍💻 Author

**Minwoo Baek (백민우)**  
Senior AI Engineer | PhD Candidate (Big Data Applications)  
💼 Hanwha Momentum / AI Manufacturing R&D  
📧 miwnoo.baek@gmail.com  
🔗 LinkedIn: https://www.linkedin.com/in/bjh713/

---

## 📜 License

MIT
