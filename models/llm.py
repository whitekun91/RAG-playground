import os
import requests


def call_llm(
    prompt: str,
    engine: str = "local",  # "local" (vLLM) or "openai"
    model: str = None,
    temperature: float = 0.5,
    top_p: float = 0.95,
    max_tokens: int = 1024,
):
    """
    Unified LLM caller.
    Supports:
      - local vLLM (OpenAI-compatible server)
      - OpenAI API (cloud)
    """

    if engine == "openai":
        # ✅ OpenAI API 호출
        openai_api_key = os.getenv("OPENAI_API_KEY", "sk-yourkey")
        openai_model = model or os.getenv("OPENAI_MODEL", "gpt-5")

        payload = {
            "model": openai_model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }

        res = requests.post(
            "https://api.openai.com/v1/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai_api_key}"
            },
            json=payload,
            timeout=120
        )
        res.raise_for_status()
        return res.json()["choices"][0]["text"].strip()

    else:
        # ✅ Local vLLM 호출
        vllm_url = os.getenv("VLLM_URL", "http://localhost:8000/v1/completions")
        vllm_model = model or os.getenv("VLLM_MODEL", "gemma-3-12b-it")

        payload = {
            "model": vllm_model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stop": ["<end_of_turn>"]
        }

        res = requests.post(
            vllm_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer sk-sample"
            },
            json=payload,
            timeout=120
        )
        res.raise_for_status()
        return res.json()["choices"][0]["text"].strip()
