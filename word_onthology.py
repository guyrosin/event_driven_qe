from collections import defaultdict, OrderedDict
from enum import Enum, auto
import json
import logging
import re

import numpy as np

import utils


class EventDetector(Enum):
    WikipediaFrequency = auto()
    Similarity = auto()


class WordOnthology(object):
    def __init__(self, models_manager, global_model=None, use_projected_models=False):
        self.models_manager = models_manager
        self.global_model = (
            models_manager[models_manager.STATIC_YEAR]
            if not global_model
            else global_model
        )
        event_id_year = json.load(
            open('data/event_id_year_since1980.json', encoding='utf-8')
        )
        self.event_year = json.load(
            open('data/event_year_since1980.json', encoding='utf-8')
        )
        self.event_id_name = json.load(
            open('data/event_id_name_since1980.json', encoding='utf-8')
        )
        self.event_id_word_count = json.load(
            open('data/event_id_word_count_since1980.json', encoding='utf-8')
        )
        self.event_id_content_length = json.load(
            open('data/event_id_content_length_since1980.json', encoding='utf-8')
        )
        self.year_to_event_id = defaultdict(list)
        self.year_to_event = defaultdict(list)
        for event_id, year in event_id_year.items():
            if models_manager.from_year <= year <= models_manager.to_year:
                self.year_to_event_id[year].append(event_id)
        for event, year in self.event_year.items():
            if models_manager.from_year <= year <= models_manager.to_year:
                self.year_to_event[year].append(event)
        self.year_to_event_id = dict(sorted(self.year_to_event_id.items()))
        self.events = list(self.event_year.keys())
        self.transformed_temporal_models = use_projected_models

    def find_events_by_similarity(
        self, entities, years, include_score=False, min_score=0.5, top_n=40
    ):
        """
        Find for each given year: events that are closest to the given query
        """
        if not entities:
            return None
        use_only_global_model = not self.transformed_temporal_models
        query_avg = (
            self.global_model.get_average_vector(entities, require_all=True)
            if use_only_global_model
            else None
        )
        years_to_events = OrderedDict([(year, []) for year in years])
        for year in years:
            # find the key events from that year
            model = (
                self.global_model
                if use_only_global_model
                else self.models_manager[year]
            )
            if not use_only_global_model:
                query_avg = model.get_average_vector(entities, require_all=True)
                if query_avg is None:
                    continue
            close_events = [
                (model.get_word(word), score)
                for word, score in model.similar_by_vector(query_avg, topn=top_n)
                if model.get_word(word) in self.events and score > min_score
            ]
            years_to_events[year] = [
                (word, round(score, 4)) if include_score else word
                for word, score in close_events
            ]
        return years_to_events

    def find_events_by_wikipedia_frequency(
        self, word, years, include_score=False, min_occurrences=5, min_score=0.004
    ):
        """
        Find for each given year: events in which Wikipedia page the given word is frequent
        """
        if not word:
            return None
        word = word.lower()
        key_years_to_events = defaultdict(list)
        for key_year in years:
            # find the key events from that year
            top_key_events = []
            for event_id in self.year_to_event_id[key_year]:
                event = self.event_id_name[event_id]
                if (
                    event not in self.global_model
                    or event_id not in self.event_id_word_count
                ):
                    continue
                # require the word to appear at least twice in the Wiki page
                occurrences = sum(
                    [
                        count if re.match(r'\b' + re.escape(word) + r'\w?\b', w) else 0
                        for w, count in self.event_id_word_count[event_id].items()
                    ]
                )
                if occurrences < 2:
                    continue
                # count number of occurrences of the given word in the Wiki page
                # take "terrorist" for "terror", but not "ambush" for "bush"
                # limit the postfix to maximum 3 letters
                occurrences = sum(
                    [
                        count
                        if re.match(r'\b' + re.escape(word) + r'\w{0,3}\b', w)
                        else 0
                        for w, count in self.event_id_word_count[event_id].items()
                    ]
                )
                score = occurrences / self.event_id_content_length[event_id]
                if occurrences > min_occurrences and score > min_score:
                    top_key_events.append((score, event))
            if top_key_events:
                top_key_events.sort(reverse=True)
                key_years_to_events[key_year] = [
                    (item[1], round(item[0], 4)) if include_score else item[1]
                    for item in top_key_events
                ]
        return key_years_to_events

    @staticmethod
    def filter_year_event_scores(year_event_score, max_events_per_year):
        year_event_score_final = defaultdict(list)
        threshold_per_year = 0.5
        threshold_total = 0.5
        total_max_score = 0
        for year, event_score in year_event_score.items():
            # for each year, look at the max score and make sure we don't take events with a much smaller score
            if event_score:
                max_score = max(item[1] for item in event_score)
                total_max_score = max(max_score, total_max_score)
                year_event_score_final[year] = [
                    item
                    for item in event_score
                    if item[1] > threshold_per_year * max_score
                ]
        # look at the max score and make sure we don't take events with a much smaller score
        if total_max_score > 0:
            for year in year_event_score_final.keys():
                year_event_score_final[year] = [
                    item
                    for item in year_event_score_final[year]
                    if item[1] > threshold_total * total_max_score
                ]
        for year, event_score in year_event_score_final.items():
            if len(event_score) > max_events_per_year:  # take the top events
                year_event_score_final[year] = utils.get_top_items(
                    event_score, max_events_per_year
                )
            return year_event_score_final

    def find_events_for_entities(
        self,
        entities,
        method,
        event_score_threshold,
        years=None,
        max_events_per_year=6,
        min_score=None,
        require_all_entities=True,
    ):
        if not all(entity in self.global_model for entity in entities):
            return
        years = self.models_manager.get_all_years() if years is None else years
        if method == EventDetector.WikipediaFrequency:
            word_year_event_score = {
                word: self.find_events_by_wikipedia_frequency(
                    word, years, include_score=True, min_score=min_score
                )
                for word in entities
            }
        elif method == EventDetector.Similarity:
            year_event_score = self.find_events_by_similarity(
                entities, years, min_score=min_score, include_score=True
            )
            year_event_score = self.filter_year_event_scores(
                year_event_score, max_events_per_year
            )
            return year_event_score  # skip the rest of the function
        else:
            raise ValueError(f'unexpected method: {method}')
        if all(
            len(year_event_score) == 0 for year_event_score in word_year_event_score
        ):
            # no events were found
            logging.info(f'* No events were found')
            return
        event_scores = defaultdict(dict)
        for word, year_event_score in word_year_event_score.items():
            for year, event_score_list in year_event_score.items():
                for event, score in event_score_list:
                    event_scores[event][word] = score

        year_event_score_final = defaultdict(list)
        required_entity_count = (
            len(entities) - 1 if len(entities) > 2 else len(entities)
        )
        for event, word_to_score in event_scores.items():
            words = list(word_to_score.keys())
            if not all(word in words for word in entities):
                continue
            if len(word_to_score) < required_entity_count:
                continue
            threshold = (
                event_score_threshold * 2
                if require_all_entities and len(word_to_score) < len(entities)
                else event_score_threshold
            )
            scores = list(word_to_score.values())
            score = float(np.mean(scores))
            if score > threshold and all(score > 0.2 * threshold for score in scores):
                year = self.event_year[event]
                year_event_score_final[year].append((event, round(score, 4)))
        year_event_score_final = self.filter_year_event_scores(
            year_event_score_final, max_events_per_year
        )
        return year_event_score_final
