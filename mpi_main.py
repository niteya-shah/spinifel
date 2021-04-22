from spinifel.mpi import main
from spinifel     import SpinifelSettings, Profiler
import logging

#logger = logging.getLogger('spinifel.sequential.orientation_matching')
#logger = logging.getLogger('spinifel.autocorrelation')
#logger.setLevel(logging.DEBUG)
#ch = logging.StreamHandler()
#ch.setLevel(logging.DEBUG)
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#ch.setFormatter(formatter)
#logger.addHandler(ch)


if __name__ == '__main__':
    settings = SpinifelSettings()
    profiler = Profiler()

    if settings.verbose:
        print(settings)

    # Configure profiler
    profiler.callmonitor_enabled = settings.use_callmonitor

    main()
