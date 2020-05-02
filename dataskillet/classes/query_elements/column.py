from dataskillet.classes.query_elements.helpers.parser_helpers import to_dict, tokenize, get_tokens_until_word


class Column:

    def __init__(self, name = None, alias = None):
        self.name = name
        self.alias = alias

    @staticmethod
    def parse_string(str):

        tokens = tokenize(str)

        if len(tokens) >= 2:
            if tokens[1].lower() != 'as' or len(tokens)>3 or len(tokens) == 2:
                raise Exception('strange column definition, allowed format <col> [optional: as <col_alias>]'.format(str=str))
            name = tokens[0]
            alias = tokens[2]

        else:
            name = tokens[0]
            alias = None

        return Column(name = name, alias = alias)
