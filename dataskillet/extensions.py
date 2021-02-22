import tempfile
import os
import shutil
import pandas as pd
import time
from dataskillet import DataSource
from dataskillet.exceptions import DataskilletException


def sql_query(sql, from_tables):
    if not from_tables or not isinstance(from_tables, dict):
        raise DataskilletException(f"Wrong from_tables value. Expected to be a dict of table names and dataframes, got: {str(from_tables)}")

    tmpdir = tempfile.gettempdir() + '/dataskillet_temp_' + time.ctime()
    ds = DataSource(metadata_dir=str(tmpdir))
    try:
        for table_name, dataframe in from_tables.items():
            if table_name not in sql:
                raise DataskilletException(f"Table {table_name} found in from_tables, but not in the SQL query.")
            tmp_fpath = os.path.join(tmpdir, f'{table_name}.csv')
            dataframe.to_csv(tmp_fpath, index=False)
            ds.add_table_from_file(tmp_fpath)

        result = ds.query(sql)
        return result
    finally:
        ds.clear_metadata(ds.metadata_dir)
        shutil.rmtree(tmpdir)


@pd.api.extensions.register_dataframe_accessor("sql")
class SQLAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def maybe_add_from_to_query(self, sql_query, table_name):
        """Inserts "FROM temp" into every SELECT clause in query that does not have a FROM clause."""
        sql_query = sql_query.lower()
        sql_query = sql_query.replace("(", " ( ").replace(")", " ) ")
        insert_positions = []
        for index, value in enumerate(sql_query):
            if sql_query[index:index + (len("select"))] == "select":
                select_pos = index

                str_after_select = sql_query[select_pos:]
                words_after_select = str_after_select.split(' ')

                keywords = ['where', 'group', 'having', 'order', 'limit', 'offset', ')']
                need_to_insert_from = True
                insert_pos = len(str_after_select)
                for word in words_after_select:
                    if word == 'from':
                        need_to_insert_from = False
                        break

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

    def __call__(self, sql):
        table_name = 'temp'
        sql = self.maybe_add_from_to_query(sql, table_name=table_name)

        return sql_query(sql, from_tables={table_name: self._obj})
