from mpi4py import MPI

from spinifel import parms, utils


def main():
    print("In MPI main", flush=True)

    timer = utils.Timer()

    print(f"Total: {timer.total():.2f}s.")
