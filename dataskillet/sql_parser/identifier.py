from dataskillet.sql_parser.base import Statement


class Identifier(Statement):
    def __init__(self, value, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value = value

    def __str__(self):
        return self.maybe_add_alias(str(self.value))
