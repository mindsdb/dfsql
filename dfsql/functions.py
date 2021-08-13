import re
from dfsql.engine import pd
from collections import Iterable

from dfsql.utils import (is_modin, is_numeric, is_booly, is_stringy, raise_bad_inputs, raise_bad_outputs,
                               TwoArgsMixin, OneArgMixin, StringInputMixin, NumericInputMixin, BoolInputMixin,
                               BoolOutputMixin, StringOutputMixin, NumericOutputMixin)


class BaseFunction:
    name = None

    # Fixes an issue with Modin internals trying to get the __name__ of aggregation functions
    @property
    def __name__(self):
        return self.name

    def assert_args(self, args):
        super().assert_args(args)

    def assert_output(self, out):
        super().assert_output(out)

    def get_output(self, args):
        return None

    def __call__(self, *args):
        self.assert_args(args)
        output = self.get_output(args)
        self.assert_output(output)
        return output

# Explanation on how this function definition works:
# https://stackoverflow.com/a/40187463/1571481

# Boolean functions


class And(BaseFunction, TwoArgsMixin, BoolInputMixin, BoolOutputMixin):
    name = 'and'

    def get_output(self, args):
        if is_modin(args[0]) and is_modin(args[1]):
            return (args[0] * args[1]).astype(bool)
        return args[0] and args[1]


class Or(BaseFunction, TwoArgsMixin, BoolInputMixin, BoolOutputMixin):
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


class Is(BaseFunction, TwoArgsMixin, BoolOutputMixin):
    name = 'is'

    def get_output(self, args):
        if is_modin(args[0]) or is_modin(args[1]):
            if args[0] is None or args[1] is None:
                # IS NULL
                target = args[0]
                if args[0] is None:
                    target = args[1]

                return pd.isnull(target)
            elif args[0] is True or args[0] is False or args[1] is True or args[1] is False and is_modin(args[0]):
                # IS [TRUE|FALSE]
                return args[0] == args[1]

        return args[0] is args[1]


class IsNot(Is):
    name = 'is not'

    def get_output(self, args):
        out = super().get_output(args)
        if is_modin(out):
            return ~out
        else:
            return not out


class Equals(BaseFunction, TwoArgsMixin, BoolOutputMixin):
    name = '='

    def get_output(self, args):
        return args[0] == args[1]


class NotEquals(BaseFunction, TwoArgsMixin, BoolOutputMixin):
    name = '!='

    def get_output(self, args):
        return args[0] != args[1]


class Greater(BaseFunction, TwoArgsMixin, BoolOutputMixin):
    name = '>'

    def get_output(self, args):
        return args[0] > args[1]


class GreaterEqual(BaseFunction, TwoArgsMixin, BoolOutputMixin):
    name = '>='

    def get_output(self, args):
        return args[0] >= args[1]


class Less(BaseFunction, TwoArgsMixin, BoolOutputMixin):
    name = '<'

    def get_output(self, args):
        return args[0] < args[1]


class LessEqual(BaseFunction, TwoArgsMixin, BoolOutputMixin):
    name = '<='

    def get_output(self, args):
        return args[0] <= args[1]


class In(BaseFunction, BoolOutputMixin):
    name = 'in'

    def assert_args(self, args):
        if not isinstance(args[1], Iterable):
            raise_bad_inputs(self)

    def get_output(self, args):
        if is_modin(args[0]):
            return args[0].isin(args[1].values)
        return args[0] in args[1]


# class IsNull(BaseFunction, OneArgMixin, BoolOutputMixin):
#     name = 'is null'
#
#     def get_output(self, args):
#         return pd.isnull(args[0])
#
#
# class IsNotNull(BaseFunction, OneArgMixin, BoolOutputMixin):
#     name = 'is not null'
#
#     def get_output(self, args):
#         return ~pd.isnull(args[0])
#
#
# class IsTrue(BaseFunction, OneArgMixin, BoolOutputMixin):
#     name = 'is true'
#
#     def get_output(self, args):
#         if is_modin(args[0]):
#             return args[0] == True
#         return args[0] is True
#
#
# class IsFalse(BaseFunction, OneArgMixin, BoolOutputMixin):
#     name = 'is false'
#
#     def get_output(self, args):
#         if is_modin(args[0]):
#             return args[0] == False
#         return args[0] is False

# Arithmetic functions


class Plus(BaseFunction, TwoArgsMixin, NumericInputMixin, NumericOutputMixin):
    name = '+'

    def get_output(self, args):
        return pd.to_numeric(args[0] + args[1])


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
            return pd.to_numeric(-args[0])
        return pd.to_numeric(args[0] - args[1])


class Multiply(BaseFunction, TwoArgsMixin, NumericInputMixin, NumericOutputMixin):
    name = '*'

    def get_output(self, args):
        return pd.to_numeric(args[0] * args[1])


class Divide(BaseFunction, TwoArgsMixin, NumericInputMixin, NumericOutputMixin):
    name = '/'

    def get_output(self, args):
        return pd.to_numeric(args[0] / args[1])


class Modulo(BaseFunction, TwoArgsMixin, NumericInputMixin, NumericOutputMixin):
    name = '%'

    def get_output(self, args):
        return pd.to_numeric(args[0] % args[1])


class Power(BaseFunction, TwoArgsMixin, NumericInputMixin, NumericOutputMixin):
    name = '^'

    def get_output(self, args):
        return pd.to_numeric(args[0] ** args[1])

# String functions


class StringConcat(BaseFunction, TwoArgsMixin, StringInputMixin, StringOutputMixin):
    name = "||"

    def get_output(self, args):
        return args[0] + args[1]


class StringLower(BaseFunction, OneArgMixin, StringInputMixin, StringOutputMixin):
    name = "lower"

    def get_output(self, args):
        if isinstance(args[0], str):
            return args[0].lower()
        return args[0].apply(lambda x: x.lower())


class StringUpper(BaseFunction, OneArgMixin, StringInputMixin, StringOutputMixin):
    name = "upper"

    def get_output(self, args):
        if isinstance(args[0], str):
            return args[0].upper()
        return args[0].apply(lambda x: x.upper())


class Like(BaseFunction, TwoArgsMixin, StringInputMixin, BoolOutputMixin):
    name = "like"

    def get_output(self, args):
        def matcher(inp, pattern):
            match = re.match(pattern, inp)
            return True if match else False

        if is_modin(args[0]):
            return args[0].apply(matcher, args=(args[1],))
        return matcher(args[0], args[1])

# Aggregate functions


class AggregateFunction(BaseFunction, OneArgMixin):
    string_repr = None # for pandas group by

    def assert_output(self, output):
        pass

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


class Max(AggregateFunction):
    name = 'max'
    string_repr = 'max'


class Min(AggregateFunction):
    name = 'min'
    string_repr = 'min'

OPERATIONS = (
    And, Or, Not,

    Equals, NotEquals, Greater, GreaterEqual, Less, LessEqual, Is, IsNot,

    Plus, Minus, Multiply, Divide, Modulo, Power,

    StringConcat, StringLower, StringUpper, Like,

    In,

    # IsNull, IsNotNull, IsTrue, IsFalse
)

OPERATION_MAPPING = {
    op.name: op for op in OPERATIONS
}
OPERATION_MAPPING['<>'] = NotEquals

AGGREGATE_FUNCTIONS = (
    Sum, Mean, Count, CountDistinct, Max, Min,
)

AGGREGATE_MAPPING = {
    op.name: op for op in AGGREGATE_FUNCTIONS
}


def is_supported(op_name):
    return op_name.lower() in OPERATION_MAPPING or op_name.lower() in AGGREGATE_MAPPING
