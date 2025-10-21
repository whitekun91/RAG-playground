import requests
from openai import OpenAI
from setting import (
    CHAT_PROVIDER_DEFAULT, OPENAI_API_KEY, OPENAI_MODEL,
    VLLM_BASE_URL, VLLM_API_KEY, VLLM_MODEL, DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS, DEFAULT_TOP_P, DEFAULT_STOP, DEFAULT_TIMEOUT
)
from langchain.schema.runnable import Runnable

class LLMManager(Runnable):
    """
    A class to manage interactions with Language Models (LLMs).
    Supports multiple providers, including OpenAI and vLLM.
    Implements Runnable for compatibility with LangChain.
    """

    def __init__(self, provider: str | None = None):
        self.provider = (provider or CHAT_PROVIDER_DEFAULT).lower()
        self.client = OpenAI(api_key=OPENAI_API_KEY) if self.provider == "openai" else None

    def call_llm(self, prompt: str) -> str:
        """
        Call the LLM with the given prompt.

        Args:
            prompt (str): The input prompt for the LLM.

        Returns:
            str: The response from the LLM.
        """
        if self.provider == "openai":
            if not OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY is not set.")

            # Construct system and user messages
            system_input = "You are a helpful assistant."
            messages = [
                {"role": "system", "content": system_input},
                {"role": "user", "content": prompt}
            ]

            try:
                response = self.client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=messages,
                    temperature=DEFAULT_TEMPERATURE,
                    max_tokens=DEFAULT_MAX_TOKENS,
                    top_p=DEFAULT_TOP_P,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                raise RuntimeError(f"Error calling OpenAI API: {e}")

        elif self.provider == "vllm":
            # Call vLLM server
            url = f"{VLLM_BASE_URL}/v1/completions"
            payload = {
                "model": VLLM_MODEL,
                "prompt": prompt,
                "temperature": DEFAULT_TEMPERATURE,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "top_p": DEFAULT_TOP_P,
                "stop": DEFAULT_STOP
            }
            headers = {"Authorization": f"Bearer {VLLM_API_KEY}"}

            try:
                response = requests.post(url, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)
                response.raise_for_status()
                return response.json().get("text", "").strip()
            except Exception as e:
                raise RuntimeError(f"Error calling vLLM API: {e}")

        else:
            raise NotImplementedError(f"Provider '{self.provider}' is not supported.")

    def invoke(self, input: str) -> str:
        """
        Invoke the LLM with the given input.

        Args:
            input (str): The input prompt for the LLM.

        Returns:
            str: The response from the LLM.
        """
        return self.call_llm(input)

# Example usage:
# llm_manager = LLMManager(provider="openai")
# result = llm_manager.call_llm("What is the capital of France?")
