"""Date extraction for English.

This module implements date extraction functionality in English.
"""

__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


# pylint: disable=bare-except

# Standard imports
import calendar
import datetime
import locale
import os
import random
from logging import getLogger
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

# Third-party packages
import regex as re

# LexNLP imports
from lexnlp.extract.all_locales.languages import Locale
from lexnlp.extract.common.annotations.date_annotation import DateAnnotation
from lexnlp.extract.common.date_parsing.datefinder import DateFinder
from lexnlp.extract.common.dates import DateParser
from lexnlp.extract.common.dates_classifier_model import get_date_features
from lexnlp.extract.en.date_model import MODEL_DATE, MODULE_PATH, DATE_MODEL_CHARS

logger = getLogger("lexnlp")


# Distance in characters to use to merge two date strings


DATE_MERGE_WINDOW = 10

# Maximum date length
DATE_MAX_LENGTH = 40

# Setup regular expression for "as of" strings
AS_OF_PATTERN = r"""
(made|dated|date)
[\s]+?
as
[\s]+?
of[\s]+?
(.{{0,{max_length}}})
""".format(max_length=DATE_MAX_LENGTH)

RE_AS_OF = re.compile(AS_OF_PATTERN, re.IGNORECASE | re.MULTILINE | re.DOTALL | re.VERBOSE)

# 'january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec'
EN_MONTHS = [['january', 'jan'], ['february', 'feb', 'febr'], ['march', 'mar'],
             ['april', 'apr'], ['may'], ['june', 'jun'],
             ['july', 'jul'], ['august', 'aug'], ['september', 'sep', 'sept'],
             ['october', 'oct'], ['november', 'nov'], ['december', 'dec']]


def get_month_by_name():
    return {month_name: ix + 1 for ix, month_names in enumerate(EN_MONTHS) for month_name in month_names}


MONTH_BY_NAME = get_month_by_name()

MONTH_FULLS = {v.lower(): k for k, v in enumerate(calendar.month_name)}


def get_raw_date_list(text, strict=False, base_date=None, return_source=False, locale=None) -> List:
    return list(get_raw_dates(
        text, strict=strict, base_date=base_date, return_source=return_source, locale=locale))


def get_raw_dates(text, strict=False, base_date=None,
                  return_source=False, locale=None) -> Generator:
    """
    Find "raw" or potential date matches prior to false positive classification.
    :param text: raw text to search
    :param strict: whether to return only complete or strict matches
    :param base_date: base date to use for implied or partial matches
    :param return_source: whether to return raw text around date
    :param locale: locale object
    :return:
    """
    if isinstance(locale, str):
        locale = Locale(locale)
    elif locale is None:
        locale = Locale('')
    # Setup base date
    if not base_date:
        base_date = datetime.datetime.now().replace(
            day=1, month=1, hour=0, minute=0, second=0, microsecond=0)

    # Find potential dates
    date_finder = DateFinder(base_date=base_date)

    for extra_token in date_finder.EXTRA_TOKENS_PATTERN.split('|'):
        if extra_token != 't':
            date_finder.REPLACEMENTS[extra_token] = ' '

    # Iterate through possible matches
    possible_dates = list(date_finder.extract_date_strings(text, strict=strict))
    possible_matched = []

    for i, possible_date in enumerate(possible_dates):
        # Get
        date_string = possible_date[0]
        index = possible_date[1]
        date_props = possible_date[2]

        # Cleanup "day of" strings
        if "of" in date_props["extra_tokens"] or "OF" in date_props["extra_tokens"]:
            num_dig_mod = len(possible_dates[i - 1][2]["digits_modifier"])
            if i > 0 and not possible_matched[i - 1] and num_dig_mod == 1:
                date_props["digits_modifier"].extend(possible_dates[i - 1][2]["digits_modifier"])
                date_string = possible_dates[i - 1][2]["digits_modifier"].pop() \
                                  .replace("st", "") \
                                  .replace("nd", "") \
                                  .replace("rd", "") \
                                  .replace("th", "") + date_string

        # Skip only digits modifiers
        num_dig_mod = len(date_props["digits_modifier"])
        num_dig = len(date_props["digits"])
        num_days = len(date_props["days"])
        num_month = len(date_props["months"])
        num_slash = date_props["delimiters"].count("/")
        num_point = date_props["delimiters"].count(".")
        num_hyphen = date_props["delimiters"].count("-")

        # Remove double months
        if num_month > 1:
            possible_matched.append(False)
            continue

        # Remove wrong months like Dec*ided or Mar*tin
        if num_month == 1 and date_props['extra_tokens'] \
                and (date_props['months'][0] + date_props['extra_tokens'][-1]) in date_string:
            possible_matched.append(False)
            continue

        # Check strange strings
        if num_dig_mod > 0 and num_dig == 0:
            possible_matched.append(False)
            continue

        # Skip DOW only
        if num_days > 0 and num_dig == 0:
            possible_matched.append(False)
            continue

        # Skip DOM only
        if num_month == 0 and num_dig_mod == 0 and num_dig <= 1:
            possible_matched.append(False)
            continue

        # Skip odd date like "1 10"
        if re.match(r'\d{1,2}\s+\d{1,2}', date_string):
            possible_matched.append(False)
            continue

        # Skip floats
        if num_point and not num_month and not re.match(r'\d{2}\.\d{2}\.\d{2,4}', date_string):
            possible_matched.append(False)
            continue

        # Skip odd months from string like "Nil 62. Marquee"
        if re.search(r'\d{2,4}\.\s*[A-Za-z]', date_string):
            possible_matched.append(False)
            continue

        # Skip fractions
        if (num_slash == 1 or num_hyphen == 1) and num_dig > 2:
            possible_matched.append(False)
            continue

        # Skip three-digit blocks and double zero years
        found_triple = False
        found_dz = False
        for digit in date_props["digits"]:
            if len(digit) == 3:
                found_triple = True
            if digit.startswith("00"):
                found_dz = True
        if found_triple or found_dz:
            possible_matched.append(False)
            continue

        # Skip "may" alone
        if num_dig == 0 and num_days == 0 and "".join(date_props["months"]).lower() == "may":
            possible_matched.append(False)
            continue

        # Skip cases like "13.2 may" or "12.12may"
        if (
                num_dig > 0
                and (num_point + num_slash + num_hyphen) > 0
                and "".join(date_props["months"]).lower() == "may"
        ):
            possible_matched.append(False)
            continue

        # Cleanup
        for token in sorted(date_props["extra_tokens"], key=len, reverse=True):
            if token.lower() in ["to", "t"]:
                continue
            date_string = date_string.replace(token, "")
        date_string = date_string.strip()
        date_props["extra_tokens"] = []

        # Skip strings too long
        if len(date_string) > DATE_MAX_LENGTH:
            possible_matched.append(False)
            continue

        # Skip numbers only
        match_delims = set("".join(date_props["delimiters"]))
        bad_delims = {",", " ", "\n", "\t"}
        len_diff_set = len(match_delims - bad_delims)
        if len_diff_set == 0 and num_month == 0:
            possible_matched.append(False)
            continue

        # Parse and skip nones
        date = None
        try:
            date_string_tokens = date_string.split()
            for cutter in range(len(date_string_tokens)):
                for direction in (0, 1):
                    if cutter > 0:
                        if direction:
                            _date_string_tokens = date_string_tokens[cutter:]
                        else:
                            _date_string_tokens = date_string_tokens[:-cutter]
                        date_string = ' '.join(_date_string_tokens)
                    try:
                        date = date_finder.parse_date_string(date_string, date_props, locale=locale)
                    except locale.Error as e:
                        raise e
                    except Exception as e:
                        logger.warning(f'Cannot parse date: {date}\n{e}')
                        date = None
                    if date:
                        break
                else:
                    continue  # executed if the loop ended normally (no break)
                break  # executed if 'continue' was skipped (break)
        except TypeError:
            possible_matched.append(False)
            continue

        if date and not check_date_parts_are_in_date(date, date_props):
            date = None

        if not date:
            possible_matched.append(False)
            continue
        # for case when datetime.datetime(2001, 1, 22, 20, 1, tzinfo=tzoffset(None, -104400))
        if hasattr(date, 'tzinfo'):
            try:
                _ = date.isoformat()
            except ValueError:
                possible_matched.append(False)
                continue
        possible_matched.append(True)

        if isinstance(date, datetime.datetime) and date.hour == 0 and date.minute == 0:
            date = date.date()
        # Append
        if return_source:
            yield (date, index)
        else:
            yield date


def check_date_parts_are_in_date(
        date: datetime.datetime,
        date_props: Dict[str, List[Any]]
) -> bool:
    """
    Checks that when we transformed "possible date" into date, we found
    place for each "token" from the initial phrase
    :param date:
    :param date_string: "13.2 may"
    :param date_props: {'time': [], 'hours': [] ... 'digits': ['13', '2'] ...}
    :return: True if date is OK
    """

    def _ordinal_to_cardinal(s: str) -> Optional[int]:
        n: str = ''
        for char in s:
            if char.isdigit():
                n: str = f'{n}{char}'
        return int(n) if n else None

    units_of_time: Tuple[str, ...] = ('year', 'month', 'day', 'hour', 'minute')
    date_values: Dict[str, int] = {
        unit: getattr(date, unit)
        for unit in units_of_time
    }

    date_prop_digits: List[int] = [int(d) for d in date_props['digits']]
    date_prop_months: List[int] = [
        MONTH_BY_NAME.get(month.lower())
        for month in date_props['months']
    ]
    date_prop_days: List[int] = [
        day for day in
        (_ordinal_to_cardinal(n) for n in date_props['digits_modifier'])
        if day
    ]

    # skip cases like "Section 7.7.10 may"
    if date_prop_months:
        month = date_values.get('month')
        if month:
            if month not in date_prop_months:
                return False

    combined: List[int] = [*date_prop_digits, *date_prop_months, *date_prop_days]
    difference: Set[int] = set(combined).difference(date_values.values())

    removeable: List[int] = []
    reassembled_date: Dict[str, int] = {}
    for k, v in date_values.items():
        if k == 'year':
            short_year = (v - 100 * (v // 100)) if v > 1000 else v
            if short_year in combined:
                reassembled_date[k] = v
                removeable.append(short_year)
                continue
        if v in combined:
            reassembled_date[k] = v
            removeable.append(v)

    diff_digits: List[int] = [digit for digit in difference if digit not in removeable]
    diff_units: Set[str] = date_values.keys() - reassembled_date.keys()

    if any(k for k in diff_units if k in units_of_time[:3]):
        if diff_digits:
            return False
    return True


def get_dates_list(text, **kwargs) -> List:
    return list(get_dates(text, **kwargs))


def get_dates(text: str,
              strict=False,
              base_date=None,
              return_source=False,
              threshold=0.50,
              locale='') -> Generator:
    """
    Find dates after cleaning false positives.
    :param text: raw text to search
    :param strict: whether to return only complete or strict matches
    :param base_date: base date to use for implied or partial matches
    :param return_source: whether to return raw text around date
    :param threshold: probability threshold to use for false positive classifier
    :param locale: locale string
    :return:
    """
    # Get raw dates
    for ant in get_date_annotations(text, strict, locale, base_date, threshold):
        if return_source:
            yield ant.date, ant.coords
        else:
            yield ant.date


def get_date_annotations(text: str,
                         strict: Optional[bool] = None,
                         locale: Optional[str] = '',
                         base_date: Optional[datetime.datetime] = None,
                         threshold: float = 0.50) \
        -> Generator[DateAnnotation, None, None]:
    """
    Find dates after cleaning false positives.
    :param text: raw text to search
    :param strict: whether to return only complete or strict matches
    :param locale: locale string
    :param base_date: base date to use for implied or partial matches
    :param threshold: probability threshold to use for false positive classifier
    :return:
    """

    # Get raw dates
    strict = strict if strict is not None else False
    raw_date_results = get_raw_dates(
        text, strict=strict, base_date=base_date, return_source=True, locale=Locale(locale))

    for raw_date in raw_date_results:
        feature_row = get_date_features(text, raw_date[1][0], raw_date[1][1], characters=DATE_MODEL_CHARS)
        feature_list = len(feature_row) * [0.0]
        for i, col in enumerate(MODEL_DATE.columns):
            feature_list[i] = feature_row[col]
        date_score = MODEL_DATE.predict_proba([feature_list])
        if date_score[0, 1] >= threshold:
            date, coordinates = raw_date
            annotation = DateAnnotation(
                coords=coordinates,
                text=text[slice(*coordinates)],
                date=date,
                score=date_score[0, 1]
            )
            yield annotation


parser = DateParser(DATE_MODEL_CHARS, enable_classifier_check=True, locale=Locale('en-US'), classifier_model=MODEL_DATE)
_get_dates = parser.get_dates
_get_date_list = parser.get_date_list
