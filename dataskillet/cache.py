from functools import lru_cache


class BaseCache:
    def get(self, table):
        pass

    def clear(self):
        pass


class DoNothingCache(BaseCache):
    pass


class MemoryCache(BaseCache):
    def __init__(self, maxsize=None):
        decorated_get = lru_cache(maxsize=maxsize)(self.get)
        setattr(self, 'get', decorated_get)

    def clear(self):
        self.get.cache_clear()

    def get(self, table):
        df = table.fetch_and_preprocess()
        return df
