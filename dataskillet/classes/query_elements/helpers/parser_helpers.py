
def get_tokens_until_word(tokens, until_word):
    """
    Get a list of tokens until the word until word is found and the index in which it was found
    :param tokens: a list of tokens
    :param until_word: stop workd
    :return: None if no match, else subtokens, match_index
    """
    ret = []

    for i,token in enumerate(tokens):
        if token.lower() == until_word:
            return ret, i
        ret += [token]

    return None, None

def get_tokens_until_words(tokens, until_words):
    """
    Get a list of tokens until the word until word is found and the index in which it was found
    :param tokens: a list of tokens
    :param until_words: stop word list
    :return: None if no match, else subtokens, match_index, matched word
    """
    ret = []

    for i,token in enumerate(tokens):
        if token.lower() in [until_words]:
            return ret, i, token
        ret += [token]

    return None, None, None

def tokenize(string):

    """
    Tokenize a string but take into account () and quotes
    :param string: the string you want to tokenize
    :return: the tokens
    """

    mapping = []
    token_i = 0
    in_quote = False
    all_quotes = ["'", '"', "`"]
    in_parentesis_count = 0
    quote_open_with = all_quotes

    for i, s in enumerate(string):

        previous_s = '' if i == 0 else string[i - 1]

        if s == '(' and not in_quote:
            in_parentesis_count += 1

        if s == ')' and not in_quote and in_parentesis_count > 0:
            in_parentesis_count -= 1

        if s in quote_open_with:

            if in_quote and previous_s != '\\':

                in_quote = False
                quote_open_with = all_quotes
            else:
                in_quote = True
                quote_open_with = [s]

        elif s in [" ", "\t", "\n", ','] and not in_quote and in_parentesis_count == 0:
            if previous_s not in [" ", "\t", "\n"]:
                token_i += 1

        mapping += [token_i]


    tokens = []
    current = mapping[0]
    word = ''

    for i, t_i in enumerate(mapping):


        if t_i != current:
            tokens += [word.strip()]
            word = ''
            current = t_i

        word += string[i]

        if i == len(mapping)-1:
            tokens += [word.strip()]


    return tokens

def to_dict(obj, class_key=None):
    """
    Return an object as a dictionary recursively
    :param obj: the object to inspect
    :param class_key: if you want to start on a given key
    :return: a dictionary
    """

    if isinstance(obj, dict):
        data = {}
        for (k, v) in obj.items():
            data[k] = to_dict(v, class_key)
        return data
    elif hasattr(obj, "_ast"):
        return to_dict(obj._ast())
    elif hasattr(obj, "__iter__") and not isinstance(obj, str):
        return [to_dict(v, class_key) for v in obj]
    elif hasattr(obj, "__dict__"):
        data = dict([(key, to_dict(value, class_key))
            for key, value in obj.__dict__.items()
            if not callable(value) and not key.startswith('_')])
        if class_key is not None and hasattr(obj, "__class__"):
            data[class_key] = obj.__class__.__name__
        return data
    else:
        return obj
