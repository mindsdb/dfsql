from pyparsing import basestring

from dfsql.engine import pd
from dfsql.exceptions import QueryExecutionException


def raise_bad_inputs(func):
    raise QueryExecutionException(f'Invalid inputs for function {func.name}')


def raise_bad_outputs(func):
    raise QueryExecutionException(f'Invalid outputs produced by function {func.name}')


def is_modin(thing):
    return (isinstance(thing, pd.Series) or isinstance(thing, pd.DataFrame))


def is_booly(thing):
    if ((is_modin(thing))
            or isinstance(thing, bool)
            or (int(thing) in (0, 1))):
        return True
    return False


def is_numeric(thing):
    if ((is_modin(thing) and thing.dtype.name != 'object')
            or isinstance(thing, int) or isinstance(thing, float)):
        return True
    return False


def is_stringy(thing):
    if ((is_modin(thing) and thing.dtype.name in ('string', 'object'))
            or isinstance(thing, str)):
        return True
    return False


class TwoArgsMixin:
    def assert_args(self, args):
        if len(args) != 2:
            raise_bad_inputs(self)


class OneArgMixin:
    def assert_args(self, args):
        if len(args) != 1:
            raise_bad_inputs(self)


class BoolInputMixin:
    def assert_args(self, args):
        if not all([is_booly(arg) for arg in args]):
            raise_bad_inputs(self)


class BoolOutputMixin:
    def assert_output(self, output):
        if not is_booly(output):
            raise_bad_outputs(self)


class NumericInputMixin:
    def assert_args(self, args):
        if not all([is_numeric(arg) for arg in args]):
            raise_bad_inputs(self)


class NumericOutputMixin:
    def assert_output(self, output):
        if not is_numeric(output):
            raise_bad_outputs(self)


class StringInputMixin:
    def assert_args(self, args):
        if not all([is_stringy(arg) for arg in args]):
            raise_bad_inputs(self)


class StringOutputMixin:
    def assert_output(self, output):
        if not is_stringy(output):
            raise_bad_outputs(self)


class CaseInsensitiveKey(str):
    def __init__(self, key):
        self.key = key

    def __hash__(self):
        return hash(self.key.lower())

    def __eq__(self, other):
        if isinstance(other, CaseInsensitiveKey):
            return self.key.lower() == other.key.lower()
        elif isinstance(other, str):
            return self.key.lower() == other.lower()

    def __str__(self):
        return self.key

    def __repr__(self):
        return self.key.__repr__()

"""https://stackoverflow.com/questions/2082152/case-insensitive-dictionary"""
class CaseInsensitiveDict(dict):
    @classmethod
    def _k(cls, key):
        return key.lower() if isinstance(key, basestring) else key

    def __init__(self, *args, **kwargs):
        super(CaseInsensitiveDict, self).__init__(*args, **kwargs)
        self._convert_keys()
    def __getitem__(self, key):
        return super(CaseInsensitiveDict, self).__getitem__(self.__class__._k(key))
    def __setitem__(self, key, value):
        super(CaseInsensitiveDict, self).__setitem__(self.__class__._k(key), value)
    def __delitem__(self, key):
        return super(CaseInsensitiveDict, self).__delitem__(self.__class__._k(key))
    def __contains__(self, key):
        return super(CaseInsensitiveDict, self).__contains__(self.__class__._k(key))
    def has_key(self, key):
        return super(CaseInsensitiveDict, self).has_key(self.__class__._k(key))
    def pop(self, key, *args, **kwargs):
        return super(CaseInsensitiveDict, self).pop(self.__class__._k(key), *args, **kwargs)
    def get(self, key, *args, **kwargs):
        return super(CaseInsensitiveDict, self).get(self.__class__._k(key), *args, **kwargs)
    def setdefault(self, key, *args, **kwargs):
        return super(CaseInsensitiveDict, self).setdefault(self.__class__._k(key), *args, **kwargs)
    def update(self, E={}, **F):
        super(CaseInsensitiveDict, self).update(self.__class__(E))
        super(CaseInsensitiveDict, self).update(self.__class__(**F))
    def _convert_keys(self):
        for k in list(self.keys()):
            v = super(CaseInsensitiveDict, self).pop(k)
            self.__setitem__(k, v)


def pd_get_column_case_insensitive(df, column):
    column_names = df.columns
    series = [df[c] for c in column_names]
    cols_dict = CaseInsensitiveDict(dict(zip(column_names, series)))
    return cols_dict.get(column)


def get_df_column(df, column, case_sensitive):
    if case_sensitive:
        if column in df.columns:
            return df[column]
    else:
        column = pd_get_column_case_insensitive(df, column)
        if column is not None:
            return column
