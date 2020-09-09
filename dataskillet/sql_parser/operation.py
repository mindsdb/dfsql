from dataskillet.sql_parser.base import Statement


class Operation(Statement):
    def __init__(self, op, args, raw=None):
        super().__init__(raw)
        self.op = op
        self.args = args

    def __str__(self):
        args_str = ','.join([str(arg) for arg in self.args])
        return f'{self.op}({args_str})'


class BinaryOperation(Operation):
    def __str__(self):
        return f'{str(self.args[0])} {self.op} {str(self.args[1])}'