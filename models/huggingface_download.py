import os
import aiohttp
import asyncio
from huggingface_hub import HfApi, login
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeRemainingColumn,
    DownloadColumn, TransferSpeedColumn
)

# ----------------------- CONFIGURATION -----------------------
# Hugging Face access token (replace with your own if needed)
HF_TOKEN = "HF_ACCESS_TOKEN"
# Model repository to download from
repo_id = "suno/bark"
repo_type = "model"
# Output directory for downloaded files
output_dir = "TTS_Models/bark"
# Download chunk size (in bytes)
CHUNK_SIZE = 65536  # 64KB
# Maximum number of download retries
MAX_RETRIES = 100
# Exponential backoff factor for retries
BACKOFF_FACTOR = 2

# Login to Hugging Face and create output directory if it doesn't exist
login(token=HF_TOKEN)
os.makedirs(output_dir, exist_ok=True)

# ------------------- RICH PROGRESS BAR SETUP -------------------
progress = Progress(
    TextColumn("[bold blue]{task.fields[filename]}"),
    BarColumn(),
    DownloadColumn(),
    TransferSpeedColumn(),
    TimeRemainingColumn(),
)

# ------------------ FILE DOWNLOAD FUNCTION ------------------
async def download_file(session, file_path, output_directory, retries=MAX_RETRIES, backoff_factor=BACKOFF_FACTOR):
    """
    Download a single file from the Hugging Face repository with retry and progress bar support.
    Skips download if the file already exists.
    """
    destination_path = os.path.join(output_directory, file_path)
    destination_dir = os.path.dirname(destination_path)

    if os.path.exists(destination_path):
        return f"⚠️ Skipped: {file_path} (already exists)"

    os.makedirs(destination_dir, exist_ok=True)
    url = f"https://huggingface.co/{repo_id}/resolve/main/{file_path}"

    for attempt in range(1, retries + 1):
        try:
            headers = {"Authorization": f"Bearer {HF_TOKEN}"}
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=900)) as response:
                if response.status != 200:
                    raise Exception(f"HTTP error code: {response.status}")

                total = int(response.headers.get("Content-Length", 0))
                task_id = progress.add_task("Downloading", filename=file_path, total=total)

                with open(destination_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                        f.write(chunk)
                        progress.update(task_id, advance=len(chunk))

                progress.remove_task(task_id)
                return f"✅ Completed: {file_path}"

        except Exception as e:
            wait = min(backoff_factor ** attempt, 60)
            error_type = type(e).__name__
            error_message = str(e).strip()

            progress.console.print(
                f"[yellow]🔁 Retry {attempt}/{retries} - {file_path}[/yellow]\n"
                f"[red]⛔ Error type: {error_type} | Message: {error_message or 'Unknown error'}[/red]\n"
                f"[green]⏳ Retrying in {wait} seconds...[/green]"
            )
            await asyncio.sleep(wait)

    return f"❌ Failed: {file_path} (max retries exceeded)"

# ------------------- MAIN ASYNC FUNCTION -------------------
async def main():
    """
    Main function to fetch the file list from the Hugging Face repo and download all files in parallel.
    """
    print("🔍 Fetching repository file list...")
    api = HfApi()
    try:
        repo_info = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
        print(f"📄 Found {len(repo_info)} files in the repository.")
    except Exception as e:
        print(f"❌ Failed to fetch file list: {e}")
        return

    connector = aiohttp.TCPConnector(ssl=False)

    async with aiohttp.ClientSession(connector=connector) as session:
        with progress:
            tasks = [download_file(session, file_path, output_dir) for file_path in repo_info]
            results = await asyncio.gather(*tasks)
            for r in results:
                progress.console.print(r)

# ------------------------ SCRIPT ENTRY POINT ------------------------
if __name__ == "__main__":
    # Run the main async function
    asyncio.run(main())
