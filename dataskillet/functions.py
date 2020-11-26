import re

import modin.pandas as pd
from pandas._testing import isiterable

from dataskillet.exceptions import QueryExecutionException


def is_modin(thing):
    return (isinstance(thing, pd.Series) or isinstance(thing, pd.DataFrame))


def raise_bad_inputs(func):
    raise QueryExecutionException(f'Invalid inputs for function {func.name}')


def raise_bad_outputs(func):
    raise QueryExecutionException(f'Invalid outputs produced by function {func.name}')


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


class BaseFunction:
    name = None

    def assert_args(self, args):
        pass

    def assert_output(self, out):
        pass

    def get_output(self, args):
        return None

    def __call__(self, *args):
        self.assert_args(args)
        output = self.get_output(args)
        self.assert_output(output)
        return output


class TwoArgsFunction(BaseFunction):
    def assert_args(self, args):
        if len(args) != 2:
            raise_bad_inputs(self)


class OneArgFunction(BaseFunction):
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


class And(TwoArgsFunction, BoolInputMixin, BoolOutputMixin):
    name = 'and'

    def get_output(self, args):
        if is_modin(args[0]) and is_modin(args[1]):
            return (args[0] * args[1]).astype(bool)
        return args[0] and args[1]


class Or(TwoArgsFunction, BoolInputMixin, BoolOutputMixin):
    name = 'or'

    def get_output(self, args):
        if is_modin(args[0]) and is_modin(args[1]):
            return (args[0] + args[1]).astype(bool)
        return args[0] or args[1]


class Not(BaseFunction, BoolOutputMixin):
    name = 'not'

    def assert_args(self, args):
        if len(args) != 1:
            raise_bad_inputs(self)

        if not (is_modin(args[0])
                or isinstance(args[0], bool)
                or (args[0] in (0, 1))):
            raise_bad_inputs(self)

    def get_output(self, args):
        if is_modin(args[0]):
            return ~args[0]
        return not args[0]


class Plus(TwoArgsFunction, NumericInputMixin, NumericOutputMixin):
    name = '+'

    def get_output(self, args):
        return args[0] + args[1]


class Minus(BaseFunction, NumericOutputMixin):
    name = '-'

    def assert_args(self, args):
        if not (len(args) == 1 or len(args) == 2):
            raise_bad_inputs(self)

        if len(args) == 2:
            if not ((is_modin(args[0]) and is_modin(args[1]))
                    or (is_numeric(args[0]) and is_numeric(args[1]))):
                raise_bad_inputs(self)

        if len(args) == 1:
            if not (is_modin(args[0]) or (is_numeric(args[0]))):
                raise_bad_inputs(self)

    def get_output(self, args):
        if len(args) == 1:
            return -args[0]
        return args[0] - args[1]


class Multiply(TwoArgsFunction, NumericInputMixin, NumericOutputMixin):
    name = '*'

    def get_output(self, args):
        return args[0] * args[1]


class Divide(TwoArgsFunction, NumericInputMixin, NumericOutputMixin):
    name = '/'

    def get_output(self, args):
        return args[0] / args[1]


class Modulo(TwoArgsFunction, NumericInputMixin, NumericOutputMixin):
    name = '%'

    def get_output(self, args):
        return args[0] % args[1]


class Power(TwoArgsFunction, NumericInputMixin, NumericOutputMixin):
    name = '^'

    def get_output(self, args):
        return args[0] ** args[1]


class Equals(TwoArgsFunction, BoolOutputMixin):
    name = '='

    def get_output(self, args):
        return args[0] == args[1]


class NotEquals(TwoArgsFunction, BoolOutputMixin):
    name = '!='

    def get_output(self, args):
        return args[0] != args[1]


class Greater(TwoArgsFunction, BoolOutputMixin):
    name = '>'

    def get_output(self, args):
        return args[0] > args[1]


class GreaterEqual(TwoArgsFunction, BoolOutputMixin):
    name = '>='

    def get_output(self, args):
        return args[0] >= args[1]


class Less(TwoArgsFunction, BoolOutputMixin):
    name = '<'

    def get_output(self, args):
        return args[0] < args[1]


class LessEqual(TwoArgsFunction, BoolOutputMixin):
    name = '<='

    def get_output(self, args):
        return args[0] <= args[1]


class In(BaseFunction, BoolOutputMixin):
    name = 'in'

    def assert_args(self, args):
        if not isiterable(args[1]):
            raise_bad_inputs(self)

    def get_output(self, args):
        if is_modin(args[0]):
            return args[0].isin(args[1].values)
        return args[0] in args[1]


class IsNull(OneArgFunction, BoolOutputMixin):
    name = 'is null'

    def get_output(self, args):
        return pd.isnull(args[0])


class IsNotNull(OneArgFunction, BoolOutputMixin):
    name = 'is not null'

    def get_output(self, args):
        return ~pd.isnull(args[0])


class IsTrue(OneArgFunction, BoolOutputMixin):
    name = 'is true'

    def get_output(self, args):
        if is_modin(args[0]):
            return args[0] == True
        return args[0] is True


class IsFalse(OneArgFunction, BoolOutputMixin):
    name = 'is false'

    def get_output(self, args):
        if is_modin(args[0]):
            return args[0] == False
        return args[0] is False

# String functions

class StringInputMixin:
    def assert_args(self, args):
        if not all([is_stringy(arg) for arg in args]):
            raise_bad_inputs(self)


class StringOutputMixin:
    def assert_output(self, output):
        if not is_stringy(output):
            raise_bad_outputs(self)


class StringConcat(TwoArgsFunction, StringInputMixin, StringOutputMixin):
    name = "||"

    def get_output(self, args):
        return args[0] + args[1]


class StringLower(OneArgFunction, StringInputMixin, StringOutputMixin):
    name = "lower"

    def get_output(self, args):
        if isinstance(args[0], str):
            return args[0].lower()
        return args[0].apply(lambda x: x.lower())


class StringUpper(OneArgFunction, StringInputMixin, StringOutputMixin):
    name = "upper"

    def get_output(self, args):
        if isinstance(args[0], str):
            return args[0].upper()
        return args[0].apply(lambda x: x.upper())


class Like(TwoArgsFunction, StringInputMixin, StringOutputMixin):
    name = "~~"

    def get_output(self, args):
        def matcher(inp, pattern):
            match = re.match(pattern, inp)
            return True if match else False

        if is_modin(args[0]):
            return args[0].apply(matcher, args=(args[1],))
        return matcher(args[0], args[1])


class AggregateFunction(BaseFunction):
    string_repr = None # for pandas group by

    @classmethod
    def string_or_callable(cls):
        if cls.string_repr:
            return cls.string_repr
        return cls()


class Mean(AggregateFunction):
    name = 'avg'
    string_repr = 'mean'


class Sum(AggregateFunction):
    name = 'sum'
    string_repr = 'sum'


class Count(AggregateFunction):
    name = 'count'
    string_repr = 'count'


class CountDistinct(AggregateFunction):
    name = 'count_distinct'
    string_repr = 'nunique'


AGGREGATE_FUNCTIONS = (
    Sum, Mean, Count, CountDistinct,
)

OPERATIONS = (
    And, Or, Not,

    Equals, NotEquals, Greater, GreaterEqual, Less, LessEqual,

    Plus, Minus, Multiply, Divide, Modulo, Power,

    StringConcat, StringLower, StringUpper, Like,

    In,

    IsNull, IsNotNull, IsTrue, IsFalse
)

AGGREGATE_MAPPING = {
    op.name: op for op in AGGREGATE_FUNCTIONS
}


OPERATION_MAPPING = {
    op.name: op for op in OPERATIONS
}
OPERATION_MAPPING['<>'] = Equals


def is_supported(op_name):
    return op_name.lower() in OPERATION_MAPPING or op_name.lower() in AGGREGATE_MAPPING


