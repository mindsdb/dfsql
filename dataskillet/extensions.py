import tempfile
import os
import pandas as pd
from dataskillet import DataSource


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
        print(new_query)
        return new_query

    def __call__(self, sql_query):
        tmpdir = tempfile.gettempdir()
        ds = DataSource(metadata_dir=str(tmpdir))
        assert not ds.tables
        table_name = 'temp'
        tmp_fpath = os.path.join(tmpdir, f'{table_name}.csv')

        try:
            self._obj.to_csv(tmp_fpath, index=False)
            ds.add_table_from_file(tmp_fpath)

            sql_query = self.maybe_add_from_to_query(sql_query, table_name=table_name)
            result = ds.query(sql_query)
        finally:
            ds.clear_metadata(ds.metadata_dir)
            os.remove(tmp_fpath)
        return result


# print('Registered accessor')