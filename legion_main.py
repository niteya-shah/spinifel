from spinifel        import settings
from spinifel.legion import main



if __name__ == '__main__':
    # Get settings from CLI
    settings.from_cli()

    if settings.verbosity > 0 :
        print(settings)
    if settings.mode == "legion_psana2":
        from spinifel.legion import main_psana2
        main_psana2()
    else:
        assert settings.mode == "legion"
        main()
