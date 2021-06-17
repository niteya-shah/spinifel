import time


def prod(iterable):
    """Return product of elements of iterable."""
    if not iterable:
        return 0
    accumulator = 1
    for element in iterable:
        accumulator *= element
    return accumulator


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


class Logger():
    def __init__(self, active):
        self.active = active

    def log(self, msg):
        if self.active:
            print(msg, flush=True)


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
