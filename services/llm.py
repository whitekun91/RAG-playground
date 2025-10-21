import time
from typing import Callable, Optional


class LLMService:
    def __init__(self, call_fn: Callable[[str], str], retries: int = 2, backoff_sec: float = 0.8):
        self._call_fn = call_fn
        self._retries = max(0, retries)
        self._backoff_sec = max(0.0, backoff_sec)

    def call(self, prompt: str) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(self._retries + 1):
            try:
                return self._call_fn(prompt)
            except Exception as e:
                last_err = e
                if attempt < self._retries:
                    time.sleep(self._backoff_sec * (attempt + 1))
                else:
                    raise

    def invoke(self, input: str) -> str:
        return self.call(input)


