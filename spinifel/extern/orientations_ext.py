from importlib.metadata import version
import numpy as np
import PyNVTX as nvtx

from spinifel import settings, contexts, Profiler

if settings.use_cupy:
    import cupy as cp
    import cupyx as cpx

from mpi4py.util import dtlib
from mpi4py import MPI

from threading import Thread
from itertools import chain
from math import ceil

class TransferBufferGPU:
    def __init__(self, shape, dtype):
        """
        Initialise our Transfer Buffer class for the GPU. This class wraps the CPU host and GPU for the CPU target->CPU host->GPU transfer
        Uses the context from spinifel context

        :param shape -- shape of the buffer
        :param dtype -- dtype of the buffers
        """
        contexts.ctx.push()

        self.stream = cp.cuda.Stream()
        self.cpu_buf = cpx.empty_pinned(shape, dtype)
        self.gpu_buf = cp.empty(shape, dtype=dtype)

        contexts.ctx.pop()

    def set_data(self):
        """
        Require the correct context ctx to be pushed
        
        Sends the entire cpu_buf to the gpu_buf and assumes that H, K, L are set correctly
        """
        self.gpu_buf.set(self.cpu_buf, self.stream)
        self.stream.synchronize()
        
    def set_data_local(self, arr, index):
        """
        Require the correct context ctx to be pushed. Since the memory is local, H, K, L can
        be set independently.
        """
        self.gpu_buf[index].set(arr, self.stream)
        self.stream.synchronize()

    def get_HKL(self, **kwargs):
        """
        Unused keyword argument kwargs to keep API uniform between CPU and GPU mode
        """
        return self.gpu_buf[0], self.gpu_buf[1], self.gpu_buf[2]

class TransferBufferCPU:
    def __init__(self, shape, dtype):
        """
        Initialise our Transfer Buffer class for the CPU. We can directly load 
        either from local shared memory or just use the cpu_buf for MPI loads

        :param shape -- shape of the buffer
        :param dtype -- dtype of the buffers
        """
        self.cpu_buf = np.empty(shape, dtype)
        self.local_buf = {}

    def set_data(self):
        """
        Data already in buffer so this function is not used
        """
        pass

    def set_data_local(self, arr, index):
        """
        Store address of local data array for ease of access and unified API
        """
        self.local_buf[index] = arr

    def get_HKL(self, shared=True):
        """
        if shared use local_buf, else cpu_buf
        """
        if shared:
            return self.local_buf[0], self.local_buf[1], self.local_buf[2]
        else:
            return self.cpu_buf[0], self.cpu_buf[1], self.cpu_buf[2]

class SharedMemory:
    
    def __init__(self, shape, dtype, split=True, pinned=True):
        """
        Initialise our Shared Memory class. This class wraps a MPI shared Window
        :param shape -- shape of the buffer
        :param dtype -- dtype of the buffer
        :param split(optional) -- if split is true, every rank in the node creates its own buffer. Else, only rank 0 creates a buffer
        """

        self.np_dtype = dtype
        self.mpi_dtype = dtlib.from_numpy_dtype(dtype)
        self.itemsize = self.mpi_dtype.Get_size()
        self.shape = shape

        if split or contexts.rank_shared == 0:
            self.nbytes = np.prod(self.shape) * self.itemsize
        else:
            self.nbytes = 0
        self.win_shared = MPI.Win.Allocate_shared(
            size=self.nbytes,disp_unit=self.itemsize, comm=contexts.comm_compute_shared)

        buf, itemsize = self.win_shared.Shared_query(contexts.rank_shared if split else 0)
        self.local_buf = np.ndarray(buffer=buf, dtype=self.np_dtype, shape=self.shape)

        if split or (contexts.rank_shared == 0) and pinned:
            cp.cuda.runtime.hostRegister(self.local_buf.ctypes.data, self.local_buf.nbytes, 0x2)
        contexts.comm_compute_shared.Barrier()

    def __getitem__(self, k):
        return self.local_buf[k]

    def __setitem__(self, k, v):
        self.local_buf[k] = v

class WindowManager:

    def __init__(self, shape, dtype, pinned=True):
        """
        Initialise our Window Manager class. This class wraps the global MPI Windows
        :param shape -- shape of all data 
        :param dtype(optional) -- dtype of the buffer
        :param split(optional) -- if split is true, every rank in the node creates its own buffer. Else, only rank 0 creates a buffer
        """
        if settings.split_type == 'even':
            self.rank_shape, self.splits = self.get_even_rank_shape(shape)
        elif settings.split_type == 'balanced':
            self.rank_shape, self.splits = self.get_balanced_rank_shape(shape)
        else:
            raise ValueError("Unknown split type")

        self.dtype = dtype
                
        self.shared_memory = SharedMemory(self.rank_shape, dtype, pinned=pinned)
        self.win = MPI.Win.Create(self.shared_memory.local_buf, comm=contexts.comm_compute)

    def lock(self, target_rank):
        """
        Helper to begin transfers from the window at target rank
        """
        self.win.Lock(rank=target_rank, lock_type=MPI.LOCK_SHARED)

    def unlock(self, target_rank):
        """
        Helper to end transfers from the window at target rank
        """
        self.win.Unlock(rank=target_rank)

    @staticmethod
    def get_even_rank_shape(shape):
        """
        Heurestic to split the data. Currently, splits so that data is split evenly among the ranks. 
        """
        assert shape[1] % (contexts.size_compute) == 0, "Cannot evenly split orientations between ranks"
        return [shape[0], shape[1]//(contexts.size_compute), *shape[2:]], 1
    
    @staticmethod
    def get_balanced_rank_shape(shape):
        """
        Heurestic to split the data. Currently, splits so that data is balanced over the ranks to maximize local data.
        """
        # Computation is always local
        if shape[1] <= settings.split_size:
            return [
                shape[0],
                shape[1] // contexts.size_compute_shared,
                *shape[2:],
            ], contexts.size_compute // contexts.size_compute_shared
        # We need to distribute over nodes
        else:
            num_nodes_in_split = ceil(shape[1] / settings.split_size)
            assert (
                shape[1] % num_nodes_in_split == 0
            ), "Cannot split orientations over ranks in a balanced manner"
            assert (
                contexts.size_compute // contexts.size_compute_shared
            ) >= num_nodes_in_split, "Not enough nodes to split in balanced manner"
            assert (
                contexts.size_compute // contexts.size_compute_shared
            ) % num_nodes_in_split == 0, (
                "Cannot split orientations over nodes in balanced manner"
            )
            return [shape[0], shape[1] // (num_nodes_in_split * contexts.size_compute_shared), *shape[2:]], (
                contexts.size_compute // contexts.size_compute_shared
            ) // num_nodes_in_split

    def get_win(self, target_rank, target, transfer_buf):
        """
        Helper Function to perform the Get MPI operation for the MPI window
        target_rank : Rank to fetch data from
        target : MPI Window Get information for offset, dtype and strides
        transfer_buf : Buffer to put the data into
        """
        self.win.Get(
            transfer_buf,
            target_rank=target_rank,
            target=(*target, self.shared_memory.mpi_dtype))
    
    def get_win_local(self, rank):
        """
        Get the data from a shared memory region as an numpy array
        """
        buf, itemsize = self.shared_memory.win_shared.Shared_query(rank)
        return np.ndarray(buffer=buf, dtype=self.dtype, shape=self.rank_shape)
    
    def set_win(self, arr):
        """
        Set values for the rank's shared memory region
        """
        self.shared_memory[:] = arr
    
    def get_strides(self):
        """
        Return strides for the shared memory region
        """
        return self.shared_memory.local_buf.strides

    def flush(self, target_rank):
        """
        Flush all local operations for the Window
        """
        self.win.Flush_local(target_rank)

def rank_generator(shared_comm_size, comm_size, rank, stream_id, num_streams, splits=1):
    """
    Yields target rank for a symmetric All-to-All exchange.
    stream_id : Ranks for the stream
    splits : Number of times to split the nodes
    """
    assert (
        comm_size % shared_comm_size == 0
    ), "shared comm size is not divisible comm size"
    assert rank < comm_size, "rank larger than comm size"
    assert stream_id < num_streams, "stream id larger than number of streams"

    num_nodes = comm_size // shared_comm_size
    local_rank = rank % shared_comm_size
    node_id = rank // shared_comm_size

    assert num_nodes % splits == 0, "Cannot split data evenly between the nodes"

    num_nodes_in_split = num_nodes // splits
    offset_node_id = node_id % num_nodes_in_split
    offset_in_split = node_id // num_nodes_in_split

    for idx, i in enumerate(
        chain(range(local_rank, shared_comm_size), range(local_rank))
    ):
        for jdx, j in enumerate(
            chain(range(offset_node_id, num_nodes_in_split), range(offset_node_id))
        ):
            target_rank = (i + j * shared_comm_size) % (
                shared_comm_size * num_nodes_in_split
            )
            stream_id_target = (idx * num_nodes_in_split + jdx) % num_streams
            if stream_id == stream_id_target:
                yield target_rank + offset_in_split * num_nodes_in_split * shared_comm_size
