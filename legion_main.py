from spinifel import settings
# Get settings from CLI -- this needs to happen before any other spinifel import
settings.from_cli()

from spinifel.legion import main



if __name__ == '__main__':
    if settings.verbosity > 0 :
        print(settings)
    if settings.mode == "legion_psana2":
        from spinifel.legion import main_psana2
        main_psana2()
    else:
        assert settings.mode == "legion"
        main()
