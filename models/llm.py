import requests
from openai import OpenAI
from setting import (CHAT_PROVIDER_DEFAULT, OPENAI_API_KEY, OPENAI_MODEL,
                                                            VLLM_BASE_URL, VLLM_API_KEY,
                                                            VLLM_MODEL, DEFAULT_TEMPERATURE,
                                                            DEFAULT_MAX_TOKENS, DEFAULT_TOP_P,
                                                            DEFAULT_STOP, DEFAULT_TIMEOUT)

_client = OpenAI(  # 필요하면 base_url 커스터마이즈 가능
    api_key=OPENAI_API_KEY,
)

def call_llm(prompt: str, provider: str | None = None) -> str:
    """
    하나의 함수로 OpenAI API 또는 로컬 vLLM 서버 호출.
    provider 인자를 생략하면 .env의 PROVIDER 설정값을 따름.
    """
    selected_provider = (provider or CHAT_PROVIDER_DEFAULT).lower()

    # ── OpenAI ─────────────────────────────────────
    if selected_provider == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")

        # system/user 메시지 구성 (system_input은 선택)
        system_input = "You are a helpful assistant."
        messages = [
            {"role": "system", "content": system_input},
            {"role": "user", "content": prompt}
        ]

        try:
            resp = _client.chat.completions.create(
                model=OPENAI_MODEL,  # 예: "gpt-5-chat"
                messages=messages,
                temperature=DEFAULT_TEMPERATURE,
                max_tokens=DEFAULT_MAX_TOKENS,
                top_p=DEFAULT_TOP_P,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            # 체인 상위로 에러를 올리면 재시도 핸들러에서 잡기 쉬움
            raise RuntimeError(f"OpenAI chat.completions error: {e}") from e

    # ── 로컬 vLLM ─────────────────────────────────────
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
        raise ValueError(f"지원되지 않는 PROVIDER 값: {selected_provider}")