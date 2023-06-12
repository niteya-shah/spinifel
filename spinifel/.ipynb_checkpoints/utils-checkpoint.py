import numpy as np
import time
import PyNVTX as nvtx
import os


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
        self.messages = []
        self.fsc = []

    def log(self, msg, level=0):
        if self.active:
            if self.myrank is None:
                if level <= self.settings.verbosity:
                    print(f"{msg}", flush=True)
            else:
                if level <= self.settings.verbosity:
                    print(f"rank:{self.myrank} {msg}", flush=True)
            self.messages.append(msg)
    
    def log_fsc(self, fsc):
        if self.active:
            self.fsc.append(fsc)
            
    def get_unique_name(self, path, prefix, suffix):
        counter = 0
        while os.path.isfile(os.path.join(path, f'{prefix}-{counter}.{suffix}')):
            counter += 1
        fname = f'{prefix}-{counter}.{suffix}'
        return fname
    
    def save(self, path):
        log_fname = os.path.join(path, 'log.txt')
        fsc_fname = os.path.join(path, 'fsc.txt')
        with open(os.path.join(path, log_fname), 'w') as f:
            for line in self.messages:
                f.write(f"{line}\n")
        with open(os.path.join(path, fsc_fname), 'w') as f:
            for line in self.fsc:
                f.write(f"{line}\n")


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
