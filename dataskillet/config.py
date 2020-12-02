from confi import BaseEnvironConfig, ConfigField, ConfigError

engine_options = ('ray', 'dask')


def process_engine_options(value):
    value = value.strip().lower()
    if value not in engine_options:
        raise ConfigError(f'MODIN_ENGINE must be one of {str(engine_options)}, got: {value}')
    return value


class Configuration(BaseEnvironConfig):
    MODIN_ENGINE = ConfigField(processor=process_engine_options, default='dask')
