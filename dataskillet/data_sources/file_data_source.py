import os
import modin.pandas as pd
from dataskillet.data_sources.base_data_source import DataSource
from dataskillet.table import Table


class FileSystemDataSource(DataSource):
    def __init__(self, tables, root_path):
        super().__init__(tables)
        self.root_path = root_path

    @staticmethod
    def from_dir(root_path):
        files = os.listdir(root_path)

        tables = []
        for f in files:
            if f.endswith('.csv'):
                fpath = os.path.join(root_path, f)
                df = pd.read_csv(fpath)

                df.columns = [col.lower() for col in df.columns]

                fname = '.'.join(f.split('.')[:-1])
                table = Table(name=fname, df=df)
                tables.append(table)

        if not tables:
            raise(Exception(f'Directory {root_path} does not contain any spreadsheet files'))

        return FileSystemDataSource(tables=tables,
                                    root_path=root_path)
