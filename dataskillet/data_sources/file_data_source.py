import os
import modin.pandas as pd
from dataskillet.data_sources.base_data_source import DataSource
from dataskillet.table import FileTable


class FileSystemDataSource(DataSource):
    def __init__(self, tables=None):
        super().__init__(tables)

    def add_table_from_file(self, path, clean=True):
        table = FileTable.from_file(path, clean=clean)
        self.add_table(table)

    @staticmethod
    def from_dir(root_path):
        files = os.listdir(root_path)
        ds = FileSystemDataSource(tables={})
        for f in files:
            if f.endswith('.csv'):
                fpath = os.path.join(root_path, f)
                ds.add_table_from_file(fpath)

        if not ds.tables:
            raise(Exception(f'Directory {root_path} does not contain any spreadsheet files'))
        return ds
