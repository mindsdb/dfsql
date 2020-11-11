import re
import pandas as pd
import numpy as np


def preprocess_column_name(text):
    text = re.sub(' +', ' ', text)
    text = "".join(c for c in text if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_').replace('-', '_').lower()
    return text


def make_preprocessing_dict(df):
    rename = {col: preprocess_column_name(col) for col in df.columns}
    empty_rows = list(df.index[df.isnull().all(axis=1)].values)
    drop_columns = list(df.columns[df.isnull().all(axis=0)].values)
    return dict(rename=rename, empty_rows=empty_rows, drop_columns=drop_columns)


def preprocess_dataframe(df, rename, empty_rows, drop_columns):
    df = df.rename(columns=rename)
    df = df.drop(drop_columns, axis=1)
    df = df.drop(empty_rows, axis=0)
    return df


class Table:
    def __init__(self, name, df, preprocessing_dict=None):
        self.name = name
        self.df = df
        self.preprocessing_dict = preprocessing_dict

    @classmethod
    def from_dataframe(cls, name, df, preprocess=True):
        preprocessing_dict = {}
        if preprocess:
            preprocessing_dict = make_preprocessing_dict(df)
            df = preprocess_dataframe(df, **preprocessing_dict)
        return cls(name=name, df=df, preprocessing_dict=preprocessing_dict)
