from dataskillet.classes.query_elements.helpers.parser_helpers import to_dict, tokenize, get_tokens_until_word


class Column:

    def __init__(self, name = None, alias = None):
        self.name = name
        self.alias = alias

    @staticmethod
    def parse(str_or_tokens):

        if type(str_or_tokens) == type([]):
            tokens = str_or_tokens
            str = ' '.join(str_or_tokens)
        else:
            str = str_or_tokens
            tokens = tokenize(str_or_tokens)


        if len(tokens) >= 2:
            if tokens[1].lower() != 'as' or len(tokens)>3 or len(tokens) == 2:
                raise Exception('strange column definition, allowed format <col> [optional: as <col_alias>]: {str}'.format(str=str))
            name = tokens[0]
            alias = tokens[2]

        else:
            name = tokens[0]
            alias = None

        return Column(name = name, alias = alias)
