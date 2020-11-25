import modin.pandas as pd
from pandas._testing import isiterable

from dataskillet.exceptions import QueryExecutionException


def is_modin(thing):
    return (isinstance(thing, pd.Series) or isinstance(thing, pd.DataFrame))


def is_number(thing):
    if isinstance(thing, pd.Series) or isinstance(thing, pd.DataFrame):
        return False

    if hasattr(thing, 'shape'):
        return False

    try:
        float(thing)
        return True
    except (TypeError, ValueError):
        return False


def raise_bad_inputs(func):
    raise QueryExecutionException(f'Invalid inputs for function {func.name}')


def raise_bad_outputs(func):
    raise QueryExecutionException(f'Invalid outputs produced by function {func.name}')


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
        super().assert_args(args)


class OneArgFunction(BaseFunction):
    def assert_args(self, args):
        if len(args) != 1:
            raise_bad_inputs(self)
        super().assert_args(args)


class BinaryBoolInputMixin:
    def assert_args(self, args):
        if not ((is_modin(args[0]) and is_modin(args[1]))
                or (isinstance(args[0], bool) and isinstance(args[1], bool))
                or (args[0] in (0, 1) and args[1] in (0, 1))):
            raise_bad_inputs(self)


class BoolOutputMixin:
    def assert_output(self, output):
        if not isinstance(output, bool) \
            and not ((output in (0, 1)) if is_number(output) else True) \
            and not (is_modin(output) and set(output.values.flatten().tolist()).union({True, False}) == {True, False}):
            raise_bad_outputs(self)


class BinaryNumericInputMixin:
    def assert_args(self, args):
        if not ((is_modin(args[0]) and is_modin(args[1]))
                or (is_number(args[0]) and is_number(args[1]))):
            raise_bad_inputs(self)


class NumericOutputMixin:
    def assert_output(self, output):
        if not is_number(output) and not (is_modin(output)):
            raise_bad_outputs(self)


class And(TwoArgsFunction, BinaryBoolInputMixin, BoolOutputMixin):
    name = 'and'

    def get_output(self, args):
        if is_modin(args[0]) and is_modin(args[1]):
            return (args[0] * args[1]).astype(bool)
        return args[0] and args[1]


class Or(TwoArgsFunction, BinaryBoolInputMixin, BoolOutputMixin):
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


class Plus(TwoArgsFunction, BinaryNumericInputMixin, NumericOutputMixin):
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
                    or (is_number(args[0]) and is_number(args[1]))):
                raise_bad_inputs(self)

        if len(args) == 1:
            if not (is_modin(args[0]) or (is_number(args[0]))):
                raise_bad_inputs(self)

    def get_output(self, args):
        if len(args) == 1:
            return -args[0]
        return args[0] - args[1]


class Multiply(TwoArgsFunction, BinaryNumericInputMixin, NumericOutputMixin):
    name = '*'

    def get_output(self, args):
        return args[0] * args[1]


class Divide(TwoArgsFunction, BinaryNumericInputMixin, NumericOutputMixin):
    name = '/'

    def get_output(self, args):
        return args[0] / args[1]


class Modulo(TwoArgsFunction, BinaryNumericInputMixin, NumericOutputMixin):
    name = '%'

    def get_output(self, args):
        return args[0] % args[1]


class Power(TwoArgsFunction, BinaryNumericInputMixin, NumericOutputMixin):
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


