import time
from typing import Dict

class Metrics:
    def __init__(self):
        self.counters: Dict[str, int] = {}
        self.timers_total: Dict[str, float] = {}
        self.timers_count: Dict[str, int] = {}
        self._active_timers: Dict[str, float] = {}
    
    def inc(self, name: str, value: int = 1):
        self.counters[name] = self.counters.get(name, 0) + value
    
    def time(self, name: str):
        start_time = time.time()
        self._active_timers[name] = start_time
        
        def finish():
            if name in self._active_timers:
                elapsed = time.time() - self._active_timers[name]
                self.timers_total[name] = self.timers_total.get(name, 0) + elapsed
                self.timers_count[name] = self.timers_count.get(name, 0) + 1
                del self._active_timers[name]
        
        return finish
    
    def avg(self, name: str) -> float:
        if name not in self.timers_count or self.timers_count[name] == 0:
            return 0.0
        return self.timers_total[name] / self.timers_count[name]
