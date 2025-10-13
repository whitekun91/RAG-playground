import os
import aiohttp
import asyncio
from huggingface_hub import HfApi, login
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeRemainingColumn,
    DownloadColumn, TransferSpeedColumn
)

# ----------------------- 설정 -----------------------
HF_TOKEN = ""
repo_id = ""
repo_type = "model"
output_dir = "TTS_Models/higgs-audio-v2-tokenizer"
CHUNK_SIZE = 65536  # 64KB
MAX_RETRIES = 100
BACKOFF_FACTOR = 2

# 로그인 및 디렉토리 생성
login(token=HF_TOKEN)
os.makedirs(output_dir, exist_ok=True)

# ------------------- rich progress -------------------
progress = Progress(
    TextColumn("[bold blue]{task.fields[filename]}"),
    BarColumn(),
    DownloadColumn(),
    TransferSpeedColumn(),
    TimeRemainingColumn(),
)


# ------------------ 파일 다운로드 함수 ------------------
async def download_file(session, file_path, output_directory, retries=MAX_RETRIES, backoff_factor=BACKOFF_FACTOR):
    destination_path = os.path.join(output_directory, file_path)
    destination_dir = os.path.dirname(destination_path)

    if os.path.exists(destination_path):
        return f"⚠️ 건너뜀: {file_path}"

    os.makedirs(destination_dir, exist_ok=True)
    url = f"https://huggingface.co/{repo_id}/resolve/main/{file_path}"

    for attempt in range(1, retries + 1):
        try:
            headers = {"Authorization": f"Bearer {HF_TOKEN}"}
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=900)) as response:
                if response.status != 200:
                    raise Exception(f"HTTP 오류 코드: {response.status}")

                total = int(response.headers.get("Content-Length", 0))
                task_id = progress.add_task("다운로드 중", filename=file_path, total=total)

                with open(destination_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                        f.write(chunk)
                        progress.update(task_id, advance=len(chunk))

                progress.remove_task(task_id)
                return f"✅ 완료: {file_path}"

        except Exception as e:
            wait = min(backoff_factor ** attempt, 60)
            error_type = type(e).__name__
            error_message = str(e).strip()

            progress.console.print(
                f"[yellow]🔁 재시도 {attempt}/{retries} - {file_path}[/yellow]\n"
                f"[red]⛔ 오류유형: {error_type} | 메시지: {error_message or '알 수 없는 오류'}[/red]\n"
                f"[green]⏳ {wait}초 후 다시 시도합니다...[/green]"
            )
            await asyncio.sleep(wait)

    return f"❌ 실패: {file_path} (최대 재시도 초과)"


# ------------------- 메인 비동기 함수 -------------------
async def main():
    print("🔍 리포지토리 파일 목록을 가져오는 중...")
    api = HfApi()
    try:
        repo_info = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
        print(f"📄 총 {len(repo_info)}개 파일 발견")
    except Exception as e:
        print(f"❌ 파일 목록 가져오기 실패: {e}")
        return

    connector = aiohttp.TCPConnector(ssl=False)

    async with aiohttp.ClientSession(connector=connector) as session:
        with progress:
            tasks = [download_file(session, file_path, output_dir) for file_path in repo_info]
            results = await asyncio.gather(*tasks)
            for r in results:
                progress.console.print(r)


# ------------------------ 실행 ------------------------
if __name__ == "__main__":
    asyncio.run(main())
