import PyNVTX as nvtx
import numpy  as np


@nvtx.annotate("utils.py", is_prefix=True)
def getMyUnfairShare(numJobs, numWorkers, rank):
    """Returns number of events assigned to the slave calling this function."""
    assert(numJobs >= numWorkers)
    try:
        allJobs = np.arange(numJobs)
        jobChunks = np.array_split(allJobs, numWorkers)
        myChunk = jobChunks[rank]
        myJobs = allJobs[myChunk[0]:myChunk[-1]+1]
        return myJobs
    except:
        return None