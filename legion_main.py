from spinifel.legion import main
from spinifel        import SpinifelSettings



if __name__ == '__main__':
    settings = SpinifelSettings()
    if settings.verbose:
        print(settings)

    main()
