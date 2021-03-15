from dfsql.sql_parser.base import Statement
from dfsql.exceptions import SQLParsingException


LOOKUP_BOOL_OPERATION = {
    0: 'AND',
    1: 'OR',
    2: 'NOT'
}

LOOKUP_BOOL_TEST = {
    0: 'IS TRUE',
    2: 'IS FALSE',
}

LOOKUP_NULL_TEST = {
    0: "IS NULL",
    1: "IS NOT NULL",
}


class Operation(Statement):
    def __init__(self, op, args_, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.op = op
        self.args = args_
        self.assert_arguments()

    def assert_arguments(self):
        if not self.args:
            raise SQLParsingException(f'Expected arguments for operation "{self.op}"')

    def to_string(self, *args, **kwargs):
        args_str = ','.join([arg.to_string() for arg in self.args])
        return self.maybe_add_alias(f'{self.op}({args_str})')


class BinaryOperation(Operation):
    def to_string(self, *args, **kwargs):
        return self.maybe_add_alias(f'{self.args[0].to_string()} {self.op} {self.args[1].to_string()}')

    def assert_arguments(self):
        if len(self.args) != 2:
            raise SQLParsingException(f'Expected two arguments for operation "{self.op}"')


class UnaryOperation(Operation):
    def to_string(self, *args, **kwargs):
        return self.maybe_add_alias(f'{self.op} {self.args[0].to_string()}')

    def assert_arguments(self):
        if len(self.args) != 1:
            raise SQLParsingException(f'Expected one argument for operation "{self.op}"')


class ComparisonPredicate(UnaryOperation):
    def to_string(self, *args, **kwargs):
        return self.maybe_add_alias(f'{self.args[0].to_string()} {self.op}')


class Function(Operation):
    def to_string(self, *args, **kwargs):
        args_str = ', '.join([arg.to_string() for arg in self.args])
        return self.maybe_add_alias(f'{self.op}({args_str})')


class AggregateFunction(Function):
    pass


class InOperation(BinaryOperation):
    def __init__(self, *args, **kwargs):
        super().__init__(op='IN', *args, **kwargs)


def operation_factory(op, args, raw=None):
    if op == 'IN':
        return InOperation(args_=args)

    op_class = Operation
    if len(args) == 2:
        op_class = BinaryOperation
    elif len(args) == 1:
        op_class = UnaryOperation

    return op_class(op=op,
             args_=args,
             raw=raw)
