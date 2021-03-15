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

