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

    # @staticmethod
    # def test():
    #     """
    #     Method to test this parser
    #     :return:
    #     """
    #
    #     queries = [
    #         (
    #             'select * from table_1',
    #             Select(columns=Columns(all=True), from_table=Table('table_1'))
    #         ),
    #         (
    #             'select sum(a, "all") as sumi from table_1',
    #             Select(columns=Columns([Column('sum(a, "all")','sumi')]), from_table=Table('table_1'))
    #         ),
    #         (
    #             'select a, b as d from table_1 ',
    #             Select(columns=Columns([Column('a'), Column('b', 'd')]), from_table=Table('table_1'))
    #         )
    #     ]
    #
    #     for q in queries:
    #
    #         q0 = Select.parse(q[0])
    #         print("=====")
    #         print(q[0])
    #         print("\n")
    #         print(q0)
    #         print("\n")
    #         print(q[1])
    #         print("\n")
    #         if q0 != q[1]:
    #             print('error queries dont match')
    #
    #         else:
    #             print('query matched')
    #
