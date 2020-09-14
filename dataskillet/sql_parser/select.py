from dataskillet.sql_parser.base import Statement


class Select(Statement):

    def __init__(self, targets, from_table=None, where=None, group_by=None, order_by=None, limit=None, offset=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.targets = targets
        self.from_table = from_table
        self.where = where
        self.group_by = group_by
        self.order_by = order_by
        self.limit = limit
        self.offset = offset

    def maybe_add_alias(self, some_str):
        if self.alias:
            return f'({some_str}) as {self.alias}'
        else:
            return some_str

    def __str__(self):
        targets_str = ', '.join([str(out) for out in self.targets])

        out_str = f"""SELECT {targets_str}"""

        if self.from_table is not None:
            from_table_str = ', '.join([str(out) for out in self.from_table])
            out_str += f' FROM {from_table_str}'

        if self.where is not None:
            out_str += f' WHERE {str(self.where)}'

        if self.group_by is not None:
            group_by_str = ', '.join([str(out) for out in self.group_by])
            out_str += f' GROUP BY {group_by_str}'

        if self.order_by is not None:
            order_by_str = ', '.join([str(out) for out in self.order_by])
            out_str += f' ORDER BY {order_by_str}'

        if self.limit is not None:
            out_str += f' LIMIT {str(self.limit)}'

        if self.offset is not None:
            out_str += f' OFFSET {str(self.offset)}'
        return self.maybe_add_alias(out_str)
