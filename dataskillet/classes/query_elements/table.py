from dataskillet.classes.query_elements.helpers.parser_helpers import *

class JoinOn:

    def __init__(self, where_pairs):
        self.where_pairs = where_pairs

    @staticmethod
    def parse(str):
        pass

class Table:

    def __init__(self, table_name = None, alias = None, join_table=None, join_type = None, join_on = None):
        self.table_name = table_name
        self.alias = alias
        self.join_table = join_table
        self.join_type = join_type
        self.join_on = join_on

    @staticmethod
    def parse(str_or_tokens):

        if type(str_or_tokens) == type([]):
            tokens = str_or_tokens
            str = ' '.join(str_or_tokens)
        else:
            str = str_or_tokens
            tokens = tokenize(str_or_tokens)

        table_name = None
        alias = None
        join_table = None
        join_type = None
        join_on = None

        if len(tokens) == 0:
            raise Exception(
                'select expecting a table name after FROM: {str}'.format(
                    str=str))
        elif len(tokens) == 1:
            table_name = tokens[0]
        elif len(tokens) > 1 and tokens[1].lower() == 'as':
            if len(tokens) == 2:
                raise Exception(
                    'select expecting a table alias after AS: {str}'.format(
                        str=str))
            alias = tokens [2]


        return Table(table_name=table_name, alias = alias, join_table= join_table, join_type=join_type, join_on= join_on)