# Somehow mona needs to initialize mpi for the test.
# We may need to look into this.
from spinifel import contexts
contexts.init_mpi()
import os

# Import all the main tests here
#from .orientation_matching import run_match
from spinifel.mpi import main as mpi_main


# Call your tests here
def run_tests():
    test_module = os.environ.get('SPINIFEL_TEST_MODULE', '')
    if test_module == 'ORIENTATION_MATCHING':
        print('TEST ORIENTATION MATCHING')
        run_match()
    elif test_module == 'MAIN_PSANA2':
        print('TEST MPI(PSANA2) MAIN')
        mpi_main()
