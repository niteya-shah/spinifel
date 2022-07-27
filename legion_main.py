from spinifel        import settings
from spinifel.legion import main



if __name__ == '__main__':
    if settings.verbose:
        print(settings)
    if settings.mode == "legion_psana2":
        from spinifel.legion import main_psana2
        main_psana2()
    else:
        main()
