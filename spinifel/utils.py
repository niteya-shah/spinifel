import numpy as np
import time
import PyNVTX as nvtx


@nvtx.annotate("utils.py", is_prefix=True)
def prod(iterable):
    """Return product of elements of iterable."""
    if not iterable:
        return 0
    accumulator = 1
    for element in iterable:
        accumulator *= element
    return accumulator


@nvtx.annotate("utils.py", is_prefix=True)
def getMyUnfairShare(numJobs, numWorkers, rank):
    """Returns number of events assigned to the slave calling this function."""
    assert numJobs >= numWorkers
    try:
        allJobs = np.arange(numJobs)
        jobChunks = np.array_split(allJobs, numWorkers)
        myChunk = jobChunks[rank]
        myJobs = allJobs[myChunk[0] : myChunk[-1] + 1]
        return myJobs
    except BaseException:
        return None


class Timer:
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


class Logger:
    def __init__(self, active, settings, myrank=None):
        self.active = active
        self.myrank = myrank
        self.settings = settings

    def log(self, msg, level=0):
        if self.active:
            if self.myrank is None:
                if level <= self.settings.verbosity:
                    print(f"{msg}", flush=True)
            else:
                if level <= self.settings.verbosity:
                    print(f"rank:{self.myrank} {msg}", flush=True)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
