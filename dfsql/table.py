import re
from dfsql.engine import pd
import numpy as np
import os


def preprocess_dataframe(df):
    df.index = range(len(df))
    df = df.convert_dtypes()
    return df


class Table:
    def __init__(self, name, *args, cache=None, **kwargs):
        self.name = name
        self.cache = cache

    def __hash__(self):
        return hash(self.name)

    def fetch_dataframe(self):
        pass

    def fetch_and_preprocess(self):
        df = self.fetch_dataframe()
        df = preprocess_dataframe(df)
        return df

    @property
    def dataframe(self):
        if self.cache:
            return self.cache.get(self)

        return self.fetch_and_preprocess()

    def to_json(self):
        return dict(
            type=self.__class__.__name__,
            name=self.name,
        )

    @staticmethod
    def from_json(json):
        cls = {
            'Table': Table,
            'FileTable': FileTable
        }[json['type']]
        return cls(**json)


class FileTable(Table):
    def __init__(self, *args, fpath, **kwargs):
        super().__init__(*args, **kwargs)
        self.fpath = fpath

    def fetch_dataframe(self):
        return pd.read_csv(self.fpath)

    @classmethod
    def from_file(cls, path):
        fpath = os.path.join(path)
        fname = '.'.join(os.path.basename(fpath).split('.')[:-1])

        table = cls(name=fname, fpath=fpath)
        df = table.fetch_dataframe()

        return table

    def to_json(self):
        json = super().to_json()
        json['fpath'] = self.fpath
        return json
