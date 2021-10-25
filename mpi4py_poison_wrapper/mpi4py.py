def rc(*args, **kwargs):
    pass

class MPI:
    def Is_initialized():
        return False
    def Init():
        assert False, "Attempting to initialize mpi4py in Legion mode"

# assert False, "Attempting to import mpi4py in a Legion run"
