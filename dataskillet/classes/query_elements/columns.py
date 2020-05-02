from dataskillet.classes.query_elements.column import Column
from dataskillet.classes.query_elements.helpers.parser_helpers import *

class Columns:
    def __init__(self, columns=[], all = False, distinct= False):
        self.distinct = distinct
        self.columns = columns,
        self.all = all

    @staticmethod
    def parse(str_or_tokens):

        if type(str_or_tokens) == type([]):
            tokens = str_or_tokens
            str = ' '.join(str_or_tokens)
        else:
            str = str_or_tokens
            tokens = tokenize(str_or_tokens)

        distinct = False
        all = False
        columns = []

        if tokens[0].lower() == 'distinct':
            distinct = True
            tokens = tokens[1:]

        if len(tokens) == 0:
            raise Exception(
                'select expecting atleast one column name to select but from statement found instead: {str}'.format(
                    str=str))

        if tokens[0] == '*':
            all=True
        else:
            current_column_tokens = []

            for i, token in enumerate(tokens):

                if token[-1] != ',':
                    current_column_tokens += [token]

                if token[-1] == ',' or i == len(tokens) -1:

                    columns += [Column.parse(' '.join(current_column_tokens))]
                    current_column_tokens = []

        return Columns(columns=columns, distinct=distinct, all=all)



