from dataskillet.constants.query_parser import *
from dataskillet.classes.query_elements.helpers.parser_helpers import *

from dataskillet.classes.query_elements.columns import Columns, Column
from dataskillet.classes.query_elements.table import Table
from dataskillet.classes.query_elements.where import Where
from dataskillet.classes.query_elements.limit import Limit
from dataskillet.classes.query_elements.order_by import OrderBy
from dataskillet.classes.query_elements.group_by import GroupBy


class Select:

    def __init__(self, columns = ALL, from_table = None, where = None, group_by = None,  order_by = None, limit = None):

        self.columns = columns
        self.from_table = from_table
        self.where = where
        self.group_by = group_by
        self.order_by = order_by
        self.limit = limit

    def __eq__(self, other):
        return to_dict(self) == to_dict(other)

    @staticmethod
    def parse(str_or_tokens):
        """
        Parse a string o a tokenized string

        :param str_or_tokens:
        :return:
        """

        if type(str_or_tokens) == type([]):
            tokens = str_or_tokens
            str = ' '.join(str_or_tokens)
        else:
            str = str_or_tokens
            tokens = tokenize(str_or_tokens)

        # ##############
        # Tokenize statement and get tokens per section on query
        # ##############

        statements = get_tokens_until_words(tokens,['select', 'from', 'where', 'group by', 'order by', 'limit'])
        statements_order = [s['word'] for s in statements]

        _index_of = lambda ix: statements_order.index(ix) if ix in statements_order else None
        _tokens_of = lambda ix: statements[_index_of(ix)]['tokens'] if _index_of(ix) is not None else None

        # Make sure that structure of query is valid
        if _index_of('select') != 0 or _index_of('from') != 1:
            raise Exception('select string must start with SELECT <what> FROM <from statement> ...: {str}'.format(str=str))

        for s in statements:
            if s['match_count'] > 1:
                raise Exception('{word} was found more than once in query: {str}'.format(word=s['word'].upper() ,str=str))
            if len(s['tokens']) == 0:
                raise Exception('{word} has no arguments in query: {str}'.format(word=s['word'].upper(), str=str))

        # ################
        # get the columns that were selected
        # ################

        columns = Columns.parse(_tokens_of('select'))

        # ################
        # get the tables we are querying to
        # ################

        from_table = Table.parse(_tokens_of('from'))

        # ################
        # get the where
        # ################
        where = None
        if _tokens_of('where'):
            where = Where.parse(_tokens_of('where'))

        # ################
        # get the group by
        # ################
        group_by = None
        if _tokens_of('group by'):
            group_by = GroupBy.parse(_tokens_of('group by'))

        # ################
        # get the order by
        # ################
        order_by = None
        if _tokens_of('order by'):
            order_by = OrderBy.parse(_tokens_of('order by'))

        # ################
        # get the limit
        # ################
        limit = None
        if _tokens_of('limit'):
            limit = Limit.parse(_tokens_of('limit'))

        return Select(columns=columns, from_table= from_table, where=where, group_by=group_by, order_by=order_by, limit=limit)

    def __str__(self):
        import pprint
        return pprint.pformat(to_dict(self), depth=None)



    @staticmethod
    def test():
        """
        Method to test this parser
        :return:
        """

        queries = [
            (
                'select * from table_1',
                Select(columns=Columns(all=True), from_table=Table('table_1'))
            ),
            (
                'select sum(a, "all") as sumi from table_1',
                Select(columns=Columns([Column('sum(a, "all")','sumi')]), from_table=Table('table_1'))
            ),
            (
                'select a, b as d from table_1 ',
                Select(columns=Columns([Column('a'), Column('b', 'd')]), from_table=Table('table_1'))
            )
        ]

        for q in queries:

            q0 = Select.parse(q[0])
            print("=====")
            print(q[0])
            print("\n")
            print(q0)
            print("\n")
            print(q[1])
            print("\n")
            if q0 != q[1]:
                print('error queries dont match')

            else:
                print('query matched')


if __name__ == "__main__":
    Select.test()
