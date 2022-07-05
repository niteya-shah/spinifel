# Ensure that the MPI contexts have been initialized
from .. import contexts
contexts.init_mpi()

from .main import main
from .main_psana2 import main as main_psana2
