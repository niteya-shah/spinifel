import logging

from .fqmn      import fully_qualified_module_name
from .singleton import Singleton


class LoggerSettings(metaclass=Singleton):
    """
    Stores logger settings as global singleton class.
    """

    def __init__(self):
        self.level = logging.DEBUG


def getLogger(name):
    """
    getLogger(name)

    Get logger (create it if need be) with `name`. Loggers are configured using
    the `LoggerSettings` singleton class.
    """
    logger = logging.getLogger(name)
    logger_settings = LoggerSettings()
    logger.setLevel(logger_settings.level)

    # Set up custom stream handler for logger
    ch = logging.StreamHandler()
    ch.setLevel(logger_settings.level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def get_loggers():
    """
    get_loggers()

    Get all loggers that are currently in use.
    """
    return (
        logging.getLogger(name) for name in logging.root.manager.loggerDict
    )


def setLevel(level):
    """
    setLevel(level)

    Set all currently used loggers to `level` and update the setting in the
    LoggerSettings singleton class (i.e. all fututre loggers created with
    `getLogger` will use this level).
    """
    # Update existing loggers
    for logger in get_loggers():
        logger.setLevel(level)

    # Update settings for future loggers
    logger_settings = LoggerSettings()
    logger_settings.level = level




class Logger():
    def __init__(self, active):
        self.active = active

    def log(self, msg):
        if self.active:
            print(msg, flush=True)