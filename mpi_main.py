from spinifel.mpi import main
from spinifel     import settings

if __name__ == '__main__':
    if settings.verbose:
        print(settings)


    main()
