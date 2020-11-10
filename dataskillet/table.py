import re
import pandas as pd
import numpy as np


def preprocess_column_name(text):
    text = re.sub(' +', ' ', text)
    text = "".join(c for c in text if c.isalnum()).strip().replace(' ', '_').lower()
    return text


def preprocess_dataframe(df):
    rename = {col: preprocess_column_name(col) for col in df.columns}
    df = df.rename(columns=rename)

    empty_rows = df.index[df.isnull().all(axis=1)].values

    drop_columns = df.columns[df.isnull().all(axis=0)].values
    df = df.drop(drop_columns, axis=1)

    duplicate_rows = df.reset_index().index.values[df.fillna('nan').duplicated()]
    drop_rows = list(set(empty_rows).union(duplicate_rows))
    df = df.drop(drop_rows, axis=0)

    return df


class Table:
    def __init__(self, name, df):
        self.name = name
        self.df = df

    @classmethod
    def from_dataframe(cls, name, df, preprocess=True):
        if preprocess:
            df = preprocess_dataframe(df)
        return cls(name=name, df=df)
