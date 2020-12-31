from spinifel.mpi import main
from spinifel     import SpinifelSettings, Profiler



if __name__ == '__main__':
    settings = SpinifelSettings()
    profiler = Profiler()

    if settings.verbose:
        print(settings)

    # Configure profiler
    profiler.callmonitor_enabled = settings.use_callmonitor

    main()
