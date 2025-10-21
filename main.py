import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

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
from fastapi.responses import PlainTextResponse
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.services.chains import ChainService
from core.services.vector_store import VectorStoreService
from core.services.reranker import RerankService
from services.providers.llm_manager import create_llm_manager
from services.llm import LLMService
from services.providers.stt_manager import create_stt_manager
from services.providers.tts_manager import create_tts_manager
from services.stt import STTService
from services.tts import TTSService
from services.pipeline import PipelineService
from core.services.query_service import QueryService
from core.services.db_classifier import build_db_classifier
from core.interface.prompts import (
    origin_prompt,
    db_select_prompt,
    image_detect_prompt
)

from setting import TTS_PROVIDER_DEFAULT
from core.utils.config import AppConfig
from core.utils.observability import SimpleLogger
from core.utils.metrics import Metrics
from core.utils.errors import AppError, error_response

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
# ────────────────────── Exception Handlers ────────────────────── #
@app.exception_handler(AppError)
async def handle_app_error(_, exc: AppError):
    return JSONResponse(status_code=exc.status, content=error_response(exc))

@app.exception_handler(Exception)
async def handle_unexpected_error(_, exc: Exception):
    return JSONResponse(status_code=500, content=error_response(exc))

# ──────────────── Prompt Setting ──────────────── #
prompt = origin_prompt.prompt
db_select_prompt = db_select_prompt.prompt
image_detect_prompt = image_detect_prompt.prompt

# ──────────────── LLM 호출 및 체인 구성 ──────────────── #
# image_request_chain = create_image_chain(image_detect_prompt, call_vllm)

# ────────────────────── Load Components ────────────────────── #
# Config & Observability
config = AppConfig.from_env(TTS_PROVIDER_DEFAULT)
logger = SimpleLogger()
metrics = Metrics()
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Instantiate VectorStoreService
vector_service = VectorStoreService()

# Load vectorDB
vector_store = vector_service.create()

# STT
stt_service = STTService(create_stt_manager())

# TTS
tts_service = TTSService(create_tts_manager())

# Instantiate ChainService
chain_service = ChainService()

# Instantiate RerankService
reranker = RerankService()

# Instantiate LLMManager and wrap with LLMService (retry/backoff)
llm_manager = create_llm_manager()
llm_service = LLMService(llm_manager.call_llm)

# Build db selector chain via ChainService and LLM service
db_selector_chain = chain_service.create_db_chain(db_select_prompt, llm_service.call)

# Build QueryService which encapsulates full RAG flow
_db_classifier_fn = build_db_classifier(db_selector_chain)

query_service = QueryService(
    db_classifier=_db_classifier_fn,
    vector_store=vector_store,
    prompt=prompt,
    call_llm=llm_service.call,
    reranker=reranker,
)

# Pipeline orchestrator
pipeline = PipelineService(
    query_service=query_service,
    stt_service=stt_service,
    tts_service=tts_service,
    tts_provider_default=TTS_PROVIDER_DEFAULT,
)


# ────────────────────── Main Handling Query ────────────────────── #
def handle_query(question_text: str, return_audio: bool = False):
    return pipeline.ask_text(question_text, return_audio=return_audio)

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
        metrics.inc("status.requests")
        finish = metrics.time("status.latency")
        gpu_available = torch.cuda.is_available()
        status = {
            "status": "ok",
            "gpu_available": gpu_available,
            "device": torch.cuda.get_device_name(0) if gpu_available else "cpu",
            "memory_allocated_MB": torch.cuda.memory_allocated() / 1024 / 1024 if gpu_available else 0,
            "memory_reserved_MB": torch.cuda.memory_reserved() / 1024 / 1024 if gpu_available else 0,
            "uptime_sec": round(time.time() - psutil.boot_time(), 2)
        }
        finish()
        return status
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=error_response(e) | {"trace": traceback.format_exc()}
        )

@app.get("/metrics", response_class=PlainTextResponse)
async def metrics_endpoint():
    try:
        # Very basic exposition format
        lines = []
        for k, v in metrics.counters.items():
            lines.append(f"app_counter{{name=\"{k}\"}} {v}")
        for k, total in metrics.timers_total.items():
            cnt = metrics.timers_count.get(k, 0)
            avg = metrics.avg(k)
            lines.append(f"app_timer_total_seconds{{name=\"{k}\"}} {total}")
            lines.append(f"app_timer_count{{name=\"{k}\"}} {cnt}")
            lines.append(f"app_timer_avg_seconds{{name=\"{k}\"}} {avg}")
        return "\n".join(lines) + "\n"
    except Exception as e:
        return PlainTextResponse(str(e), status_code=500)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=error_response(e) | {"trace": traceback.format_exc()}
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

        wrapped = logger.wrap("ask-audio-tts", pipeline.ask_audio)
        metrics.inc("ask_audio.requests")
        finish = metrics.time("ask_audio.latency")
        result = wrapped(tmp_audio_path, return_audio)
        os.remove(tmp_audio_path)
        finish()
        result["elapsed_time"] = f"{round(time.time() - start, 2)}s"
        return result

    except Exception as e:
        return JSONResponse(status_code=500, content=error_response(e) | {"trace": traceback.format_exc()})


@app.post("/ask-text")
async def ask_text_tts(query: TextQuery):
    start = time.time()
    try:
        wrapped = logger.wrap("ask-text", handle_query)
        metrics.inc("ask_text.requests")
        finish = metrics.time("ask_text.latency")
        print(query.question)
        result = wrapped(query.question, query.return_audio)
        finish()
        result["input_text"] = query.question
        result["elapsed_time"] = f"{round(time.time() - start, 2)}s"
        return result

    except Exception as e:
        return JSONResponse(status_code=500, content=error_response(e) | {"trace": traceback.format_exc()})
