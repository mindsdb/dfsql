from dataskillet.cache import MemoryCache


class TestCache:
    def test_cache(self, data_source):
        cache = MemoryCache()
        data_source.set_cache(cache)
        assert data_source.cache is cache

        cache_info = cache.get.cache_info()
        assert cache_info.hits == 0
        assert cache_info.misses == 0
        assert cache_info.currsize == 0

        sql = "SELECT * FROM titanic"
        data_source.query(sql)

        cache_info = cache.get.cache_info()
        assert cache_info.currsize > 0
        assert cache_info.hits == 0
        assert cache_info.misses == 1

        data_source.query(sql)
        cache_info = cache.get.cache_info()
        assert cache_info.currsize > 0
        assert cache_info.hits == 1
        assert cache_info.misses == 1

        data_source.query(sql)
        cache_info = cache.get.cache_info()
        assert cache_info.currsize > 0
        assert cache_info.hits == 2
        assert cache_info.misses == 1

    def test_maxsize(self, data_source):
        cache = MemoryCache(maxsize=1)
        data_source.set_cache(cache)
        assert data_source.cache is cache

        cache_info = cache.get.cache_info()
        assert cache_info.hits == 0
        assert cache_info.misses == 0
        assert cache_info.currsize == 0

        sql = "SELECT * FROM titanic"
        data_source.query(sql)

        assert cache_info.hits == 0
        assert cache_info.misses == 0
        assert cache_info.currsize == 0

        cache = MemoryCache(maxsize=None)
        data_source.set_cache(cache)
        assert data_source.cache is cache

        sql = "SELECT * FROM titanic"
        data_source.query(sql)
        cache_info = cache.get.cache_info()
        assert cache_info.hits == 0
        assert cache_info.misses == 1
        assert cache_info.currsize > 0
