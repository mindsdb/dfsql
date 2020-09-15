from dataskillet.sql_parser.base import Statement


class Operation(Statement):
    def __init__(self, op, args_, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.op = op
        self.args = args_

    def to_string(self, *args, **kwargs):
        args_str = ','.join([arg.to_string() for arg in self.args])
        return self.maybe_add_alias(f'{self.op}({args_str})')


class BinaryOperation(Operation):
    def to_string(self, *args, **kwargs):
        return self.maybe_add_alias(f'{self.args[0].to_string()} {self.op} {self.args[1].to_string()}')


LOOKUP_BOOL_OPEARTION = {
    0: 'AND'
}


class BooleanOperation(Operation):
    def to_string(self, *args, **kwargs):
        return f'{self.args[0].to_string()} {self.op} {self.args[1].to_string()}'


class FunctionCall(Operation):
    def to_string(self, *args, **kwargs):
        args_str = ', '.join([arg.to_string() for arg in self.args])
        return self.maybe_add_alias(f'{self.op}({args_str})')


class InOperation(BinaryOperation):
    def __init__(self, *args, **kwargs):
        super().__init__(op='IN', *args, **kwargs)
