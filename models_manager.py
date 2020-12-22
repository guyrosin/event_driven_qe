import logging
import os
import re
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto
from pathlib import Path

from word2vec_model import Word2VecModel
from word2vec_wiki2vec_model import Word2VecWiki2VecModel


class SpecificModelType(Enum):
    Regular = auto()
    Wiki2Vec = auto()


STATIC_YEAR = 9999


class ModelsManager(object):
    def __init__(
        self,
        data_dir_name=sys.path[0],
        from_year=None,
        to_year=None,
        global_model_path=None,
        global_model_type=None,
        specific_model_type=SpecificModelType.Regular,
    ):
        self.data_dir_name = data_dir_name
        self.year_to_model_inner = {}
        self.from_year = from_year if from_year else 0
        self.to_year = to_year if to_year else STATIC_YEAR
        self.global_model_path = global_model_path
        self.specific_model_type = specific_model_type
        self.global_model_type = global_model_type
        self.global_model = None

    def __getitem__(self, year):
        return self.year_to_model.get(year)

    def _build_w2v_model(self, year, file=None):
        if self.specific_model_type == SpecificModelType.Regular:
            return Word2VecModel(year, self.data_dir_name, file)
        if self.specific_model_type == SpecificModelType.Wiki2Vec:
            return Word2VecWiki2VecModel(Path(self.data_dir_name, file), year)

    def _load_w2v_models(self, years=None):
        # build a word2vec model out of each model file in the 'data' folder
        args = []
        for f in os.listdir(self.data_dir_name):
            # look for years in the filename
            m = re.match(r'.+(\d{4}).*$', f)
            if m is None:
                continue
            year = int(m.group(1))
            if years and year not in years:
                continue
            if (
                year not in self.year_to_model_inner
                and self.from_year <= year <= self.to_year
            ):
                args.append((year, f))
        with ThreadPoolExecutor(max_workers=4) as executor:
            self.year_to_model_inner = {
                model.year: model
                for model in list(executor.map(self._build_w2v_model, *zip(*args)))
            }
        if (
            STATIC_YEAR not in self.year_to_model_inner
            and self.global_model_path is not None
        ):
            if self.global_model_type == SpecificModelType.Wiki2Vec:
                self.global_model = Word2VecWiki2VecModel(
                    model_path=self.global_model_path
                )
            elif self.global_model_type == SpecificModelType.Regular:
                self.global_model = Word2VecModel(
                    STATIC_YEAR, model_path=self.global_model_path
                )
            if self.global_model:
                self.global_model.init_sims()
            self.year_to_model_inner[STATIC_YEAR] = self.global_model
        for model in self.year_to_model_inner.values():
            model.init_sims()

    def load_models(
        self, years=None, model_type=None, update_time_range=True, just_global=False
    ):
        if self.year_to_model_inner:  # skip if already loaded
            return
        if just_global:
            self.from_year, self.to_year = min(years), max(years)
            update_time_range = False
            years = [STATIC_YEAR]
            logging.info(
                f"loading {model_type.name + ' ' if model_type else ''}global model from '{self.data_dir_name}'"
            )
        else:
            logging.info(
                f"loading {model_type.name + ' ' if model_type else ''}models from '{self.data_dir_name}'"
            )
        self._load_w2v_models(years)
        if not self.year_to_model_inner:
            logging.error(f"no models found in '{self.data_dir_name}'!")
            exit()

        # convert to an OrderedDict, sorted by key
        ordered_items = sorted(self.year_to_model_inner.items(), key=lambda t: t[0])
        self.year_to_model_inner = OrderedDict(ordered_items)
        if update_time_range:
            self.from_year = ordered_items[0][0]
            self.to_year = (
                ordered_items[-1][0]
                if ordered_items[-1][0] < STATIC_YEAR or len(ordered_items) == 1
                else ordered_items[-2][0]
            )
        return self.year_to_model_inner

    def get_all_years(self):
        return list(range(self.from_year, self.to_year + 1))

    @property
    def year_to_model(self):
        if not self.year_to_model_inner:
            self.load_models()
        return self.year_to_model_inner
