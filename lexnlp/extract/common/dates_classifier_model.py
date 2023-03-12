__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


import datetime
import itertools

import joblib
from typing import List, Tuple, Callable, Dict, Union, Set, Optional
import regex as re

REG_WORD_SEPARATOR = re.compile(r'[\s\-\.\[\]\{\}\(\),;:\+\\/]+')
REG_NUMBER = re.compile(r'^\d+')


def get_date_features(text,
                      start_index: int,
                      end_index: int,
                      characters: List[str],
                      alphabet_char_set: Optional[Set[str]] = False,
                      include_bigrams=True,
                      window=5,
                      norm=True,
                      count_words=False):
    """
    Get features to use for classification of date as false positive.
    :param text: raw text around potential date
    :param start_index: date start index
    :param end_index: date end index
    :param include_bigrams: whether to include bigram/bicharacter features
    :param window: window around match
    :param characters: characters to use for feature generation, e.g., digits only, alpha only
    :param alphabet_char_set: alphabetic characters only for the provided locale
    :param norm: whether to norm, i.e., transform to proportion
    :param count_words: words count in the string
    :return:
    """
    # Get text window
    window_start = max(0, start_index - window)
    window_end = min(len(text), end_index + window)
    feature_text = text[window_start:window_end].strip()
    date_text = text[start_index:end_index]

    # Build character vector
    char_vec = {}
    char_keys = []
    bigram_keys = {}
    for character in characters:
        key = f'char_{character}'
        char_vec[key] = feature_text.count(character)
        char_keys.append(key)

    # Build character bigram vector
    if include_bigrams:
        bigram_set = [''.join(s) for s in itertools.permutations(characters, 2)]
        bigram_keys = []
        for character in bigram_set:
            key = f'bigram_{character}'
            char_vec[key] = feature_text.count(character)
            bigram_keys.append(key)

    # Norm if requested
    if norm:
        # Norm by characters
        char_sum = sum([char_vec[k] for k in char_keys])
        if char_sum > 0:
            for key in char_keys:
                char_vec[key] /= float(char_sum)

        # Norm by bigrams
        if include_bigrams:
            bigram_sum = sum([char_vec[k] for k in bigram_keys])
            if bigram_sum > 0:
                for key in bigram_keys:
                    char_vec[key] /= float(bigram_sum)

    if count_words:
        # calculate numbers below 31, numbers above 31, words and capitalized words
        numbers_above_31, numbers_below_31, words, cap_words = 0, 0, 0, 0
        for wrd in split_date_words(date_text):  # type: str
            if not wrd:
                continue
            if wrd[0] in alphabet_char_set:
                is_cap = len(wrd) > 1 and wrd[0].lower() != wrd[0] and wrd[1].lower() == wrd[1]
                if is_cap:
                    cap_words += 1
                else:
                    words += 1
                continue
            numbers = [int(n.group(0)) for n in REG_NUMBER.finditer(wrd)]
            if numbers:
                if numbers[0] < 31:
                    numbers_below_31 += 1
                else:
                    numbers_above_31 += 1
        if norm:
            sum_words = numbers_above_31 + numbers_below_31 + words + cap_words
            if sum_words:
                numbers_above_31, numbers_below_31, words, cap_words = \
                    numbers_above_31 / sum_words, numbers_below_31 / sum_words, words / sum_words, cap_words / sum_words
        char_vec['nb31'] = numbers_below_31
        char_vec['na31'] = numbers_above_31
        char_vec['wr_l'] = words
        char_vec['wr_u'] = cap_words

    return char_vec


def split_date_words(date_str: str) -> List[str]:
    return REG_WORD_SEPARATOR.split(date_str)
