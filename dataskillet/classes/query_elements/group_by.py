from dataskillet.classes.query_elements.helpers.parser_helpers import *


class GroupBy:

    def __init__(self, column_names = None, having = None):
        self.column_names = column_names
        self.having = having

    @staticmethod
    def parse(str_or_tokens):

        if type(str_or_tokens) == type([]):
            tokens = str_or_tokens
            str = ' '.join(str_or_tokens)
        else:
            str = str_or_tokens
            tokens = tokenize(str_or_tokens)


