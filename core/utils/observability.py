import time
from typing import Callable, Any

class SimpleLogger:
    def __init__(self):
        pass
    
    def wrap(self, operation_name: str, func: Callable):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                print(f"[{operation_name}] Completed in {time.time() - start_time:.2f}s")
                return result
            except Exception as e:
                print(f"[{operation_name}] Failed after {time.time() - start_time:.2f}s: {e}")
                raise
        return wrapper
