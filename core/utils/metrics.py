import time
from collections import defaultdict
from typing import Dict


class Metrics:
    def __init__(self):
        self.counters: Dict[str, int] = defaultdict(int)
        self.timers_total: Dict[str, float] = defaultdict(float)
        self.timers_count: Dict[str, int] = defaultdict(int)

    def inc(self, name: str, value: int = 1):
        self.counters[name] += value

    def time(self, name: str):
        start = time.time()
        def _finish():
            elapsed = time.time() - start
            self.timers_total[name] += elapsed
            self.timers_count[name] += 1
            return elapsed
        return _finish

    def avg(self, name: str) -> float:
        cnt = self.timers_count.get(name, 0)
        return (self.timers_total.get(name, 0.0) / cnt) if cnt else 0.0



