import requests
from openai import OpenAI
from setting import (CHAT_PROVIDER_DEFAULT, OPENAI_API_KEY, OPENAI_MODEL,
                                                            VLLM_BASE_URL, VLLM_API_KEY,
                                                            VLLM_MODEL, DEFAULT_TEMPERATURE,
                                                            DEFAULT_MAX_TOKENS, DEFAULT_TOP_P,
                                                            DEFAULT_STOP, DEFAULT_TIMEOUT)

_client = OpenAI(  # Customize base_url if needed
    api_key=OPENAI_API_KEY,
)

def call_llm(prompt: str, provider: str | None = None) -> str:
    """
    Call OpenAI API or local vLLM server with a single function.
    If provider argument is omitted, follows PROVIDER setting in .env.
    """
    selected_provider = (provider or CHAT_PROVIDER_DEFAULT).lower()

    # ── OpenAI ─────────────────────────────────────
    if selected_provider == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        # Compose system/user messages (system_input is optional)
        system_input = "You are a helpful assistant."
        messages = [
            {"role": "system", "content": system_input},
            {"role": "user", "content": prompt}
        ]

        try:
            resp = _client.chat.completions.create(
                model=OPENAI_MODEL,  # e.g., "gpt-4o-mini"
                messages=messages,
                temperature=DEFAULT_TEMPERATURE,
                max_tokens=DEFAULT_MAX_TOKENS,
                top_p=DEFAULT_TOP_P,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            # Raise error to upper chain for retry handler
            raise RuntimeError(f"OpenAI chat.completions error: {e}") from e

    # ── Local vLLM ─────────────────────────────────────
    elif selected_provider == "local":
        headers = {"Content-Type": "application/json"}
        if VLLM_API_KEY:
            headers["Authorization"] = f"Bearer {VLLM_API_KEY}"

        res = requests.post(
            f"{VLLM_BASE_URL.rstrip('/')}/v1/completions",
            headers=headers,
            json={
                "model": VLLM_MODEL,
                "prompt": prompt,
                "temperature": DEFAULT_TEMPERATURE,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "top_p": DEFAULT_TOP_P,
                "stop": DEFAULT_STOP,
            },
            timeout=DEFAULT_TIMEOUT,
        )
        res.raise_for_status()
        return res.json()["choices"][0]["text"].strip()

    else:
        raise ValueError(f"Unsupported PROVIDER value: {selected_provider}")