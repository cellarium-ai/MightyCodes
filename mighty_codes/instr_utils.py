import time


class Timer:
    def __init__(self):
        self.total = 0.
    
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.total += (self.end - self.start)
