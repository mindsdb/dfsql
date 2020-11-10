import os
import modin.pandas as pd
from dataskillet.data_sources.base_data_source import DataSource
from dataskillet.table import Table


class FileSystemDataSource(DataSource):
    def __init__(self, tables=None):
        super().__init__(tables)

    def add_table_from_url(self, path):
        pass

    def add_table_from_file(self, path, preprocess=True):
        fpath = os.path.join(path)
        df = pd.read_csv(fpath)

        fname = '.'.join(os.path.basename(fpath).split('.')[:-1])
        table = Table.from_dataframe(name=fname, df=df, preprocess=preprocess)
        self.add_table(table)

    def preprocess_dataframe(self, df):
        df.columns = [col.lower() for col in df.columns]
        return df

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
