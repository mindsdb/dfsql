from dataskillet.sql_parser.base import Statement


class Constant(Statement):
    def __init__(self, value, raw=None):
        super().__init__(raw)
        self.value = value

    def __str__(self):
        return str(self.value)
