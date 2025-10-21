import time
from typing import Callable, Any, Dict


class SimpleLogger:
    def log_request(self, route: str, payload: Dict[str, Any]):
        print(f"[REQ] {route} payload_keys={list(payload.keys())}")

    def log_response(self, route: str, elapsed_sec: float, status: str = "ok"):
        print(f"[RES] {route} status={status} elapsed={elapsed_sec:.2f}s")

    def wrap(self, route: str, fn: Callable[..., Dict[str, Any]]):
        def _inner(*args, **kwargs):
            start = time.time()
            try:
                self.log_request(route, kwargs if kwargs else {})
                out = fn(*args, **kwargs)
                self.log_response(route, time.time() - start, status="ok")
                return out
            except Exception:
                self.log_response(route, time.time() - start, status="error")
                raise
        return _inner


