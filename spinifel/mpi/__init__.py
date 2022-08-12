# Ensure that the MPI contexts have been initialized
from .main import main
from .. import contexts
contexts.init_mpi()
