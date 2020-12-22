from functools import lru_cache

import gensim
import numpy as np
from gensim.parsing import preprocessing
from nltk import PerceptronTagger, WordNetLemmatizer
from nltk.corpus import stopwords, wordnet

wnl = WordNetLemmatizer()
pos_tagger = PerceptronTagger()
stopwords = stopwords.words("english")


def get_top_items(dict_or_list, n=None, sort_by_index=1, ascending=False):
    """
    Return the top items from a dictionary or a list of tuples
    """
    if not n:
        n = len(dict_or_list)
    if isinstance(dict_or_list, dict):
        return dict(
            sorted(
                dict_or_list.items(),
                key=lambda item: item[sort_by_index],
                reverse=not ascending,
            )[:n]
        )
    return sorted(
        dict_or_list, key=lambda item: item[sort_by_index], reverse=not ascending
    )[:n]


def normalize(dict_or_list):
    if isinstance(dict_or_list, dict):
        factor = 1 / np.nansum(list(dict_or_list.values()))
        return {key: val * factor for key, val in dict_or_list.items()}
    factor = 1 / np.nansum(dict_or_list)
    return [val * factor if val is not None else val for val in dict_or_list]


def tokenize(
    text,
    remove_stopwords=True,
    lower=True,
    deacc=True,
    to_str=False,
    stemming=False,
    lemmatize=False,
    word_min_length=2,
):
    """
    Convert a single sentence into a list of tokens.
    This lowercases, tokenizes (to alphabetic characters only) and converts to unicode.
    """
    tokens = gensim.utils.tokenize(text, lower=lower, deacc=deacc)
    tokens = filter(
        lambda x: len(x) >= word_min_length
        and (not remove_stopwords or x not in stopwords),
        tokens,
    )
    filtered_text = " ".join(tokens)
    filters = []
    if stemming:
        filters.append(preprocessing.stem)
    tokens = preprocessing.preprocess_string(filtered_text, filters)
    if lemmatize:
        tokens = lemmatize_words(tokens, wnl, pos_tagger)
    return " ".join(tokens) if to_str else list(tokens)


def _get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV


def get_word_pos(word, pos_tagger=pos_tagger):
    if not word:
        return word
    pos = _get_wordnet_pos(pos_tagger.tag([word])[0][1]) if pos_tagger else None
    return pos


@lru_cache(maxsize=100000)
def lemmatize_word(word, wnl, pos_tagger=None, pos=None):
    if not word:
        return word
    if not pos:
        pos = get_word_pos(word, pos_tagger)
    lemma = wnl.lemmatize(word, pos) if pos else wnl.lemmatize(word)
    return lemma


def lemmatize_words(words, wnl, pos_tagger=None):
    for word in words:
        if not word:
            yield word
        lemma = lemmatize_word(word, wnl, pos_tagger)
        yield lemma


def argpartition(l, n, highest=True):
    """
    Get the indices of the `n` items with the highest values from the list `l`
    """
    if n >= len(l) or n == 0:
        return list(range(len(l)))
    if highest:
        indices = np.argpartition(l, -n)[-n:]
    else:
        indices = np.argpartition(l, n)[:n]
    return indices
