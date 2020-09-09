from dataskillet.sql_parser.base import Statement


class Expression(Statement):
    def __init__(self, value, raw=None):
        super().__init__(raw)
        self.value = value

    def __str__(self):
        return str(self.value)


class Star(Expression):
    def __init__(self, raw=None):
        super().__init__(value='*', raw=raw)
