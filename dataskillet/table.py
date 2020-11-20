import re
import modin.pandas as pd
import numpy as np
import os


def preprocess_column_name(text):
    text = re.sub(' +', ' ', text)
    text = "".join(c for c in text if c.isalnum() or c in (' ', '_', '-')).strip().replace('.', '_').replace(' ', '_').replace('-', '_').lower()
    return text


def make_preprocessing_dict(df):
    rename = {col: preprocess_column_name(col) for col in df.columns}
    empty_rows = df.index[df.isnull().all(axis=1)].values.tolist()
    drop_columns = df.columns[df.isnull().all(axis=0)].values.tolist()
    return dict(rename=rename, empty_rows=empty_rows, drop_columns=drop_columns)


def preprocess_dataframe(df, rename, empty_rows, drop_columns):
    df = df.rename(columns=rename)
    df = df.drop(drop_columns, axis=1)
    df = df.drop(empty_rows, axis=0)
    df.index = range(len(df))
    df = df.convert_dtypes()
    return df


class Table:
    def __init__(self, name, *args, preprocessing_dict=None, **kwargs):
        self.name = name
        self.preprocessing_dict = preprocessing_dict

        self._df_cache = None

    def clear_cache(self):
        self._df_cache = None

    def fetch_dataframe(self):
        pass

    def preprocess_dataframe(self, df):
        return preprocess_dataframe(df, **self.preprocessing_dict)

    @property
    def dataframe(self):
        if self._df_cache is None:
            self._df_cache = self.fetch_dataframe()
            if self.preprocessing_dict:
                self._df_cache = self.preprocess_dataframe(self._df_cache)
        return self._df_cache

    def to_json(self):
        return dict(
            type=self.__class__.__name__,
            name=self.name,
            preprocessing_dict=self.preprocessing_dict,
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
    def from_file(cls, path, clean=True):
        fpath = os.path.join(path)
        fname = '.'.join(os.path.basename(fpath).split('.')[:-1])

        table = cls(name=fname, fpath=fpath)
        df = table.fetch_dataframe()
        if clean:
            preprocessing_dict = make_preprocessing_dict(df)
            table.preprocessing_dict = preprocessing_dict

        return table

    def to_json(self):
        json = super().to_json()
        json['fpath'] = self.fpath
        return json
