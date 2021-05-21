import tempfile
import os
import shutil
import time
from dfsql.config import Configuration
from dfsql.exceptions import dfsqlException
from dfsql.data_sources import DataSource
from pandas import DataFrame as PandasDataFrame


def sql_query(sql, *args, ds_kwargs=None, custom_functions=None, **kwargs):
    ds_args = ds_kwargs or {}
    custom_functions = custom_functions or {}
    from_tables = kwargs
    if not from_tables or not isinstance(from_tables, dict):
        raise dfsqlException(f"Wrong from_tables value. Expected to be a dict of table names and dataframes, got: {str(from_tables)}")

    tmpdir = os.path.join(tempfile.gettempdir(), 'dfsql_temp_' + str(round(time.time())))
    ds = DataSource(*args, metadata_dir=str(tmpdir), custom_functions=custom_functions, **ds_args)
    try:
        for table_name, dataframe in from_tables.items():
            if table_name not in sql:
                raise dfsqlException(f"Table {table_name} found in from_tables, but not in the SQL query.")
            tmp_fpath = os.path.join(tmpdir, f'{table_name}.csv')
            dataframe.to_csv(tmp_fpath, index=False)
            ds.add_table_from_file(tmp_fpath)

        result = ds.query(sql)
        return result
    finally:
        ds.clear_metadata(ds.metadata_dir)
        shutil.rmtree(tmpdir)

