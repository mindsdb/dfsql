from dataskillet.sql_parser.base import Statement


class Operation(Statement):
    def __init__(self, op, args_, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.op = op
        self.args = args_

    def __str__(self):
        args_str = ','.join([str(arg) for arg in self.args])
        return self.maybe_add_alias(f'{self.op}({args_str})')


class BinaryOperation(Operation):
    def __str__(self):
        return self.maybe_add_alias(f'{str(self.args[0])} {self.op} {str(self.args[1])}')


class FunctionCall(Operation):
    def __str__(self):
        args_str = ', '.join([str(arg) for arg in self.args])
        return self.maybe_add_alias(f'{self.op}({args_str})')
