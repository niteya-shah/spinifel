# Ensure that the MPI contexts have been initialized => skopi needs MPI to be
# initialized
from .. import contexts, settings
if settings.mode == "mpi":
    contexts.init_mpi()

from .main import main