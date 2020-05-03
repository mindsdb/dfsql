from dataskillet.classes.query_elements.helpers.parser_helpers import *


class Limit:

    def __init__(self, limit = None, offset = None):
        self.limit = limit
        self.offset = offset

    @staticmethod
    def parse(str_or_tokens):

        if type(str_or_tokens) == type([]):
            tokens = str_or_tokens
            str = ' '.join(str_or_tokens)
        else:
            str = str_or_tokens
            tokens = tokenize(str_or_tokens)


