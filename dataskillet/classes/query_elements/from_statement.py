from dataskillet.classes.query_elements.table import Table
from dataskillet.classes.query_elements.join import Join

from dataskillet.classes.query_elements.helpers.parser_helpers import *

class FromStatement:

    def __init__(self, table=None, join=None):
        self.table = table
        self.join = join

    @staticmethod
    def parse_string(str):
        tokens = tokenize(str)

        return FromStatement(table = Table(name=tokens[0]))

