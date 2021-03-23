from confi import BaseEnvironConfig, ConfigField, ConfigError, BooleanConfig
from distutils.util import strtobool
import logging


def true_if_modin_installed():
    try:
        import modin
        logging.info(
            "Detected Modin and an explicit USE_MODIN value was not provided. Modin will be used for dfsql operations.")
        return True
    except ImportError:
        return False


class Configuration(BaseEnvironConfig):
    USE_MODIN = BooleanConfig(default=true_if_modin_installed)
