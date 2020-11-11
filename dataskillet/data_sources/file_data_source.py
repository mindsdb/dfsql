import os
import modin.pandas as pd
from dataskillet.data_sources.base_data_source import DataSource
from dataskillet.table import FileTable


class FileSystemDataSource(DataSource):
    def add_table_from_file(self, path, clean=True):
        table = FileTable.from_file(path, clean=clean)
        self.add_table(table)

    @staticmethod
    def from_dir(metadata_dir, files_dir_path):
        files = os.listdir(files_dir_path)
        ds = FileSystemDataSource(metadata_dir=metadata_dir)
        for f in files:
            if f.endswith('.csv'):
                fpath = os.path.join(files_dir_path, f)
                ds.add_table_from_file(fpath)

        if not ds.tables:
            raise(Exception(f'Directory {files_dir_path} does not contain any spreadsheet files'))
        return ds
