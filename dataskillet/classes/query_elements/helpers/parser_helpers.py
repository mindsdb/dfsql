
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
    ret = {}

    tokens_left = tokens

    match_words_map = {word:{'word':word, 'match_index': -1, 'match_count': 0, 'tokens':[]} for word in until_words}



    for i,word in enumerate(tokens):

        if i+1 >= len(tokens):
            next_word = None
        else:
            next_word = tokens[i+1]

        for match_words in until_words:
            match_tokens = match_words.split()

            if word.lower() == match_tokens[0]:

                if len(match_tokens) == 2:
                    if next_word is None or next_word.lower() != match_tokens[1]:
                        continue

                match_words_map[match_words]['match_index'] = i
                match_words_map[match_words]['match_count'] += 1


    match_order = sorted(match_words_map.values(), key=lambda word_dict: word_dict['match_index'])

    matched = [data for data in match_order if data['match_index']>=0]
    if len(matched) ==0:
        return []

    if matched[0]['match_index'] > 0:
        matched = [{'word': '_START_', 'match_index': 0, 'match_count':0, 'tokens':[]}] + matched

    for i, match_dict in enumerate(matched):
        next_index = len(tokens) if i == len(matched) -1 else matched[i+1]['match_index']
        offset = 1 if match_dict['word'] != '_START_' else 0
        match_dict['tokens'] = tokens[match_dict['match_index']+offset:next_index]

    return matched



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
            if word.strip() != '':
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
