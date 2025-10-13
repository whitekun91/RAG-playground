import json
import os
import shutil
import tempfile
import time
import traceback
import warnings
from pathlib import Path
from uuid import uuid4

import numpy as np
import psutil
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from pydantic import BaseModel

from components.select_vectordb import classify_question_to_db_type
from components.split_korean import split_korean_sentences
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

warnings.simplefilter(action='ignore', category=FutureWarning)

import transformers
transformers.logging.set_verbosity_error()

import posthog
posthog.disabled = True


class TextQuery(BaseModel):
    question: str
    return_audio: bool = False


# ──────────────── FastAPI ──────────────── #
app = FastAPI(title="RAG + STT + TTS + OPC-UA")
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
vector_store, coater_store = setup_vector_store()

# STT
stt_pipe = load_stt_model(device=device, torch_dtype=torch_dtype)

# TTS
tts_processor, tts_model, tts_history_prompt = load_tts_model(device=device, torch=torch)


# ────────────────────── Main Handling Query ────────────────────── #
def handle_query(question_text: str, return_audio: bool = False):
    image_urls = []  # ← (추가)
    db_type = classify_question_to_db_type(db_selector_chain, question_text)
    retriever = (vector_store if db_type == "docx" else coater_store).as_retriever()
    reranked_docs = get_reranked_documents(retriever, question_text)

    rag_chain = (
            {"context": lambda _: reranked_docs, "question": RunnablePassthrough()}
            | prompt
            | RunnableLambda(lambda x: call_llm(x.to_string()))
            | StrOutputParser()
    )
    answer_text = rag_chain.invoke(question_text)

    # ✅ 이미지 요청 시 이미지 링크 추가
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
            # 실제 URL로 변환 (아래처럼 고쳐줘!)
            for ref in image_refs_dedup:
                # ref는 예시로 "extracted_images/manual_1/image_page2_1.png"
                # 이미지 저장 경로/서버 static 경로에 따라 조정 가능
                filename = ref.split("/")[-1]
                pdf_base = ref.split("/")[-2]
                url = f"/images/{pdf_base}/{filename}"  # FastAPI에 mount된 경로와 일치!
                image_urls.append(url)
            # answer_text += "\n\n 관련 이미지:\n" + "\n".join(image_urls)

    response = {
        "question": question_text,
        "rag_answer": answer_text,
        "download_url": None,
        "image_urls": image_urls if image_urls else None,  # ← (추가)
    }

    # ✅ TTS 처리 (OPC 여부와 무관하게)
    if return_audio:
        sentences = split_korean_sentences(answer_text)
        all_audio = []
        for sentence in sentences:
            inputs = tts_processor(text=[sentence], return_tensors="pt").to(device)
            pad_token_id = tts_processor.tokenizer.pad_token_id or tts_processor.tokenizer.eos_token_id
            with torch.no_grad():
                audio_array = tts_model.generate(
                    **inputs,
                    pad_token_id=pad_token_id,
                    do_sample=True,
                    history_prompt=tts_history_prompt,
                )
                audio_np = audio_array.cpu().numpy().squeeze()
                audio_np = np.clip(audio_np, -1.0, 1.0)
                all_audio.append(audio_np)
            del inputs, audio_array
            torch.cuda.empty_cache()

        silence = np.zeros(int(0.25 * 24000), dtype=np.float32)
        combined_audio = np.concatenate([np.concatenate([a, silence]) for a in all_audio])
        filename = f"{uuid4().hex}.wav"
        output_path = os.path.join("outputs", filename)
        os.makedirs("outputs", exist_ok=True)
        sf.write(output_path, combined_audio, samplerate=24000)

        response["download_url"] = f"/download/{filename}"

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
    return FileResponse(path, media_type="audio/wav", filename=filename)

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