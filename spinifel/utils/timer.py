import time


class Timer():
    def __init__(self):
        self.start_time = time.perf_counter()
        self.prev_time = self.start_time

    def lap(self):
        curr_time = time.perf_counter()
        lap_time = curr_time - self.prev_time
        self.prev_time = curr_time
        return lap_time

    def total(self):
        return time.perf_counter() - self.start_time

