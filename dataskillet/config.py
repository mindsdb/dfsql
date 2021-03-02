from confi import BaseEnvironConfig, ConfigField, ConfigError, BooleanConfig
from distutils.util import strtobool
import logging

engine_options = ('ray', 'dask')


def process_engine_options(value):
    value = value.strip().lower()
    if value not in engine_options:
        raise ConfigError(f'MODIN_ENGINE must be one of {str(engine_options)}, got: {value}')
    return value


def true_if_modin_installed():
    try:
        import modin
        logging.info(
            "Detected installed modin and an explicit USE_MODIN value was not provided. Modin will be used for dataskillet operations.")
        return True
    except ImportError:
        return False


class Configuration(BaseEnvironConfig):
    USE_MODIN = BooleanConfig(default=true_if_modin_installed)
    MODIN_ENGINE = ConfigField(processor=process_engine_options, default='dask')
