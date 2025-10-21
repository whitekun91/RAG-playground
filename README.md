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

---

## 🛠️ Refactored Class-Based Structure

The project has been refactored to use a class-based structure for better modularity and maintainability. Below are the key classes and their responsibilities:

### `TTSManager`
Manages Text-to-Speech (TTS) operations.
```python
from core.modules.tts import TTSManager

tts_manager = TTSManager(provider="openai")
result = tts_manager.tts("Hello, world!", outfile_path="output.mp3")
```

### `STTManager`
Manages Speech-to-Text (STT) operations.
```python
from core.modules.stt import STTManager

stt_manager = STTManager(provider="openai")
result = stt_manager.transcribe("path/to/audio/file.webm")
```

### `LLMManager`
Handles interactions with Language Models (LLMs).
```python
from core.modules.llm import LLMManager

llm_manager = LLMManager(provider="openai")
response = llm_manager.call_llm("What is the capital of France?")
```

### `PDFProcessor`
Processes PDF files, including image extraction and vector database creation.
```python
from documents.create_vectordb_by_pdf import PDFProcessor

pdf_processor = PDFProcessor()
pdf_processor.extract_images_from_pdf("example.pdf", "./output_images")
pdf_processor.process_pdf_to_vectordb("example.pdf")
```

### `Reranker`
Reranks documents based on their relevance to a query.
```python
from core.services.reranker import RerankService

reranker = RerankService()
reranked_docs = reranker.rerank_documents(documents, query)
```

### `VectorStoreService`
Manages the setup and configuration of vector stores.
```python
from core.services.vector_store import VectorStoreService

vector_service = VectorStoreService()
vector_store = vector_service.create()
```

### `ChainService`
Creates chains for database selection and image detection.
```python
from core.services.chains import ChainService

chain_service = ChainService()
db_chain = chain_service.create_db_chain(db_select_prompt, call_llm)
```

---

This refactored structure ensures that the codebase is modular, maintainable, and easy to extend.
