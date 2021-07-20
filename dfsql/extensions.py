import logging
import re
from dfsql import sql_query
from dfsql.engine import pd as pd_engine
import warnings
import pandas
from pandas.core.accessor import CachedAccessor


@pandas.api.extensions.register_dataframe_accessor("sql")
class SQLAccessor:
    def __init__(self, pandas_obj):
        self._obj = pd_engine.DataFrame(pandas_obj)

    def maybe_add_from_to_query(self, sql_query, table_name):
        """Inserts "FROM temp" into every SELECT clause in query that does not have a FROM clause."""
        sql_query = sql_query.lower()
        sql_query = sql_query.replace("(", " ( ").replace(")", " ) ")
        insert_positions = []
        for m in re.finditer('select', sql_query):
            select_pos = m.start()

            str_after_select = sql_query[select_pos:]
            words_after_select = str_after_select.split(' ')

            keywords = ['where', 'group', 'having', 'order', 'limit', 'offset']
            need_to_insert_from = True
            insert_pos = len(str_after_select)

            parentheses_count = 0
            for word in words_after_select:
                if word == 'from':
                    need_to_insert_from = False
                    break

                if word == '(':
                    parentheses_count += 1
                elif word == ')':
                    if parentheses_count == 0:
                        insert_pos = str_after_select.find(word)
                        break
                    else:
                        parentheses_count -= 1
                if word in keywords:
                    insert_pos = str_after_select.find(word)
                    break
            if not need_to_insert_from:
                continue
            insert_pos = select_pos + insert_pos

            insert_positions.append(insert_pos)
        insert_text = f' from {table_name} '
        new_query = ''
        last_pos = None
        for pos in insert_positions:
            new_query += sql_query[last_pos:pos] + insert_text
            last_pos = pos

        new_query += sql_query[last_pos:]
        return new_query

    def __call__(self, sql, *args, **kwargs):
        table_name = 'temp'
        sql = self.maybe_add_from_to_query(sql, table_name=table_name)
        kwargs.update({table_name: self._obj})
        return sql_query(sql, *args, **kwargs)

try:
    import modin.pandas as mpd

    def register_modin_accessor(name, cls):
        def decorator(accessor):
            if hasattr(cls, name):
                warnings.warn(
                    f"registration of accessor {repr(accessor)} under name "
                    f"{repr(name)} for type {repr(cls)} is overriding a preexisting "
                    f"attribute with the same name.",
                    UserWarning,
                    stacklevel=2,
                )

            setattr(cls, name, CachedAccessor(name, accessor))
            return accessor

        return decorator


    def register_modin_dataframe_accessor(name):
        from modin.pandas import DataFrame
        return register_modin_accessor(name, DataFrame)

    register_modin_dataframe_accessor("sql")(SQLAccessor)
except ImportError:
    warnings.warn('Modin not found, dfsql Modin dataframe extensions not loaded', UserWarning)
