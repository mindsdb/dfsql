from dataskillet.constants.query_parser import *
from dataskillet.classes.query_elements.helpers.parser_helpers import *

from dataskillet.classes.query_elements.columns import Columns, Column
from dataskillet.classes.query_elements.from_statement import FromStatement
from dataskillet.classes.query_elements.table import Table

class Select:

    def __init__(self, columns = ALL, from_statement = None, where = None, group_by = None, having = None, order_by = None, limit = None, offset = None):

        self.columns = columns
        self.from_statement = from_statement
        self.where = where
        self.group_by = group_by
        self.having = having
        self.order_by = order_by
        self.limit = limit
        self.offset = offset

    def __eq__(self, other):
        return to_dict(self) == to_dict(other)

    @staticmethod
    def parse_string(str):

        tokens = tokenize(str)

        columns = ALL
        from_statement = (None, None)
        where = None
        group_by = None
        having = None
        order_by = None
        limit = None
        offset = None

        end = False

        # ################
        # first there should be select
        # ################

        if tokens[0].lower() != 'select':
            raise Exception('select string must start with select command: {str}'.format(str=str))

        pointer = 1


        # ################
        # then parse the columns to select
        # ################

        columns_tokens, offset = get_tokens_until_word(tokens[pointer:], 'from')


        if columns_tokens is None:
            raise Exception('select expecting atleast one column name to select but from statement found instead: {str}'.format(str=str))

        pointer += offset

        columns_str = ' '.join(columns_tokens)
        columns = Columns.parse_string(columns_str)



        # ################
        # then parse the columns to from statement
        # ################

        if tokens[pointer] != 'from' or len(tokens) -1 <= pointer:
            raise Exception('FROM missing: {str}'.format(str=str))

        pointer += 1
        from_statement_tokens, offset, matched_token = get_tokens_until_words(tokens[pointer:], ['where'])


        if from_statement_tokens is None:
            from_statement_tokens = tokens[pointer:]
            end = True # no more in the query
        else:
            pointer += offset

        from_statement_str = ' '.join(from_statement_tokens)
        from_statement = FromStatement.parse_string(from_statement_str)

        return Select(columns=columns, from_statement = from_statement)

    def __str__(self):
        import pprint
        return pprint.pformat(to_dict(self), depth=3)

    @staticmethod
    def test():

        queries = [
            (
                'select * from table_1',
                Select(columns=Columns(all=True))
            ),
            (
                'select sum(a, "all") as sumi from table_1',
                Select(columns=Columns([Column('a')])) #, from_statement=Table('table_1'))
            ),
            (
                'select a, b as d from table_1',
                Select(columns=[Column('a'), Column('b', 'd')], from_statement=Table('table_1'))
            )
        ]

        for q in queries:

            q0 = Select.parse_string(q[0])
            print(q[0])
            print(q0)
            print(q[1])
            if q0 != q[1]:
                print('error queries dont match')

            else:
                print('query matched')


if __name__ == "__main__":
    Select.test()
