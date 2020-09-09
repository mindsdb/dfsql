from dataskillet.sql_parser.base import Statement


class Select(Statement):

    def __init__(self, targets, raw=None, from_table=None, where=None, group_by=None, order_by=None, limit=None):
        super().__init__(raw)
        self.targets = targets
        self.from_table = from_table
        self.where = where
        self.group_by = group_by
        self.order_by = order_by
        self.limit = limit

    def __str__(self):
        targets_str = ', '.join([str(out) for out in self.targets])

        out_str = f"""SELECT {targets_str}"""

        if self.from_table is not None:
            out_str += f' FROM {str(self.from_table)}'

        if self.where is not None:
            out_str += f' WHERE {str(self.where)}'
        return out_str

