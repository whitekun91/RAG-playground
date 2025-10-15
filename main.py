import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import os
import shutil
import tempfile
import time
import traceback
import warnings
from pathlib import Path

import psutil
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from pydantic import BaseModel

from components.select_vectordb import classify_question_to_db_type
from interface.create_chain import create_db_chain
from interface.load_vector import setup_vector_store
from interface.rag_reranker import get_reranked_documents
from models.llm import call_llm
from models.stt import load_stt_model
from models.tts import load_tts_model
from prompts import (
    origin_prompt,
    db_select_prompt,
    image_detect_prompt
)

from setting import TTS_PROVIDER_DEFAULT

warnings.simplefilter(action='ignore', category=FutureWarning)

import transformers
transformers.logging.set_verbosity_error()

import posthog
posthog.disabled = True


class TextQuery(BaseModel):
    question: str
    return_audio: bool = False


# ──────────────── FastAPI ──────────────── #
app = FastAPI(title="RAG + STT + TTS")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/images", StaticFiles(directory="./documents/extracted_images"), name="images")

# ──────────────── Prompt Setting ──────────────── #
prompt = origin_prompt.prompt
db_select_prompt = db_select_prompt.prompt
image_detect_prompt = image_detect_prompt.prompt

# ──────────────── LLM 호출 및 체인 구성 ──────────────── #
db_selector_chain = create_db_chain(db_select_prompt, call_llm)
# image_request_chain = create_image_chain(image_detect_prompt, call_vllm)

# ────────────────────── Load Components ────────────────────── #
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load vectorDB
vector_store = setup_vector_store()

# STT
stt_pipe = load_stt_model(device=device, torch_dtype=torch_dtype)

# TTS
tts = load_tts_model(device=device, torch=torch)


# ────────────────────── Main Handling Query ────────────────────── #
def handle_query(question_text: str, return_audio: bool = False):
    image_urls = []  # ← (추가)
    db_type = classify_question_to_db_type(db_selector_chain, question_text)
    retriever = vector_store.as_retriever()
    reranked_docs = get_reranked_documents(retriever, question_text)
    rag_chain = (
            {"context": lambda _: reranked_docs, "question": RunnablePassthrough()}
            | prompt
            | RunnableLambda(lambda x: call_llm(x.to_string()))
            | StrOutputParser()
    )
    answer_text = rag_chain.invoke(question_text)

    # ✅ Image link addition for image requests
    if db_type == "pdf":
        image_refs = []
        for doc in reranked_docs:
            try:
                img_list = json.loads(doc.metadata.get("image_refs", "[]"))
                image_refs.extend(img_list)
            except Exception:
                continue
        if image_refs:
            image_refs_dedup = list(dict.fromkeys(image_refs))[:5]
            # Convert to actual URL (adjust as needed for your server/static path)
            for ref in image_refs_dedup:
                # Example: ref = "extracted_images/manual_1/image_page2_1.png"
                filename = ref.split("/")[-1]
                pdf_base = ref.split("/")[-2]
                url = f"/images/{pdf_base}/{filename}"  # Must match FastAPI mount path
                image_urls.append(url)
            # answer_text += "\n\n📷 Related images:\n" + "\n".join(image_urls)

    response = {
        "question": question_text,
        "rag_answer": answer_text,
        "download_url": None,
        "image_urls": image_urls if image_urls else None,  # (added)
    }

    # ✅ TTS processing (regardless of OPC)
    if return_audio:
        import os
        from uuid import uuid4
        import logging

        provider = (os.getenv("TTS_PROVIDER", TTS_PROVIDER_DEFAULT) or "local").lower()
        audio_format = (os.getenv("OPENAI_TTS_FORMAT", "mp3").lower()
                        if provider == "openai" else "wav")

        os.makedirs("outputs", exist_ok=True)
        filename = f"{uuid4().hex}.{audio_format}"
        output_path = os.path.join("outputs", filename)

        try:
            # Generate TTS audio file
            # Removed 'format' argument because Speech.create() does not support it
            tts(
                answer_text,
                outfile_path=output_path,
                instructions="Speak in a clear, friendly tone."
            )
            # Check if file was created and is not empty
            if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
                response["download_url"] = f"/download/{filename}"
                print(f"[TTS SUCCESS] Audio file created: {output_path}")
            else:
                print(f"[TTS ERROR] Audio file not created or empty: {output_path}")
                response["download_url"] = None
        except Exception as e:
            # Log error and show reason in UI (optional)
            print("[TTS ERROR]", e)
            response["download_url"] = None

    return response

# ────────────────────── FastAPI 라우팅 ────────────────────── #
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = Path("static/index.html")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

@app.get("/download/{filename}")
async def download_audio(filename: str):
    path = os.path.join("outputs", filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found")

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "wav"
    media_type = "audio/mpeg" if ext == "mp3" else "audio/wav"
    return FileResponse(path, media_type=media_type, filename=filename)

@app.get("/status")
async def server_status():
    try:
        gpu_available = torch.cuda.is_available()
        status = {
            "status": "ok",
            "gpu_available": gpu_available,
            "device": torch.cuda.get_device_name(0) if gpu_available else "cpu",
            "memory_allocated_MB": torch.cuda.memory_allocated() / 1024 / 1024 if gpu_available else 0,
            "memory_reserved_MB": torch.cuda.memory_reserved() / 1024 / 1024 if gpu_available else 0,
            "uptime_sec": round(time.time() - psutil.boot_time(), 2)
        }
        return status
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "trace": traceback.format_exc()
            }
        )

@app.get("/pdf-images/{pdf_name}")
async def list_extracted_images(pdf_name: str):
    image_folder = Path(f"static/extracted_images/{pdf_name}")
    if not image_folder.exists():
        return JSONResponse(status_code=404, content={"error": "No images found for this PDF."})

    image_files = sorted([
        f"/static/extracted_images/{pdf_name}/{img.name}"
        for img in image_folder.glob("*.png")
    ])
    return {"image_count": len(image_files), "images": image_files}


@app.post("/ask-audio-tts")
async def ask_audio_tts(file: UploadFile = File(...), return_audio: bool = Form(False)):
    start = time.time()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_audio_path = tmp_file.name

        stt_result = stt_pipe(tmp_audio_path)
        question_text = stt_result["text"]
        os.remove(tmp_audio_path)

        result = handle_query(question_text, return_audio)
        result["stt_text"] = question_text
        result["elapsed_time"] = f"{round(time.time() - start, 2)}s"
        return result

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})


@app.post("/ask-text")
async def ask_text_tts(query: TextQuery):
    start = time.time()
    try:
        result = handle_query(query.question, query.return_audio)
        result["input_text"] = query.question
        result["elapsed_time"] = f"{round(time.time() - start, 2)}s"
        return result

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})