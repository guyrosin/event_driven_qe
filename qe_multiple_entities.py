import json
import logging
from enum import Enum, auto
from functools import lru_cache

import numpy as np
from nltk import PerceptronTagger, WordNetLemmatizer
from sqlitedict import SqliteDict

import models_manager
import utils
from word_onthology import EventDetector


class MultipleEntitiesQEMethod(Enum):
    none = auto()
    events = auto()
    events_temporal = auto()


class QEMultipleEntities:
    def __init__(
        self,
        qe_single_entity,
        two_entities_qe_method=None,
        k=10,
        onthology=None,
        event_detector=None,
        min_score=None,
        candidates_lambda=None,
    ):
        self.qe_single_entity = qe_single_entity
        self.two_entities_qe_method = two_entities_qe_method
        self.k = k
        self.models_manager = qe_single_entity.models_manager
        self.models_manager = (
            onthology.models_manager if onthology is not None else self.models_manager
        )
        self.global_model = qe_single_entity.global_model
        self.global_model = (
            self.models_manager[models_manager.STATIC_YEAR]
            if self.models_manager is not None
            else self.global_model
        )
        self.onthology = onthology
        self.events = json.load(open('data/events_since1980.json', encoding='utf-8'))
        self.event_id_name = json.load(
            open('data/event_id_name_since1980.json', encoding='utf-8')
        )
        self.pos_tagger = PerceptronTagger()
        self.wnl = WordNetLemmatizer()
        self.lemmatize = lru_cache()(utils.lemmatize_word)
        self.event_to_top_tfidf = SqliteDict(
            'data/event_to_top_tfidf_100.sqlite', flag='r'
        )
        self.event_detector = (
            event_detector if event_detector else EventDetector.WikipediaFrequency
        )
        self.min_score = min_score
        self.candidates_lambda = (
            candidates_lambda if candidates_lambda is not None else 0.6
        )

    def _expand_entities(
        self, entities, two_entities_qe_method=None, lemmatize=True, k=None
    ):
        """
        Apply QE with the selected method and return a string of expansion terms
        """
        if k is None:
            k = self.k
        if lemmatize:
            entities = [
                self.lemmatize(entity, self.wnl, self.pos_tagger) for entity in entities
            ]
        entities = [entity.replace('_', ' ').title() for entity in entities]

        qe_method = (
            two_entities_qe_method
            if two_entities_qe_method is not None
            else self.two_entities_qe_method
        )
        if qe_method == MultipleEntitiesQEMethod.none:
            expansions = {}
        elif qe_method.name.startswith('events'):
            expansions = self.expand_using_events(entities, k=k)
        else:
            raise ValueError('Unknown QE method: {}'.format(qe_method))
        if expansions:
            expansions = utils.normalize(
                {
                    exp.replace('ENTITY/', '').replace('_', ' ').lower(): score
                    for exp, score in expansions.items()
                }
            )
            expansions = {
                utils.tokenize(exp, to_str=True): score
                for exp, score in expansions.items()
            }
            expansions = {exp: score for exp, score in expansions.items() if exp}
        return expansions

    def expand(self, entities, k=None):
        expansions = self._expand_entities(entities, k=k)
        return expansions

    def filter_and_take_top_words(self, word_score, entities, k=None):
        if k is None:
            k = self.k
        word_score.sort(key=lambda tup: tup[1])  # sort by word
        filtered_word_score = []
        for (word_i, word, score) in word_score:
            word_lower = word.lower()
            if entities is not None and any(
                word_lower in entity.lower() for entity in entities
            ):
                continue
            # take the shorter option if a similar word was already selected + boost its score
            if any(
                prev_word.lower() in word_lower
                for (_, prev_word, _) in filtered_word_score
            ):
                filtered_word_score = [
                    (_, word, score * 2)
                    if word.lower() in word_lower
                    else (_, word, score)
                    for (_, word, score) in filtered_word_score
                ]
                continue
            filtered_word_score.append((word_i, word, score))
        word_score = utils.get_top_items(
            filtered_word_score, k, sort_by_index=2
        )  # sort by score, descending
        return word_score

    def _calc_temporal_relevance(self, word, event_year, event_neighbors):
        years_to_calc = [event_year - 1, event_year + 1]
        if not all(
            year in self.models_manager.get_all_years() for year in years_to_calc
        ):
            return np.nan
        scores = []
        for neighbor in event_neighbors:
            hist = {
                year: self.models_manager[year].similarity(word, neighbor)
                for year in years_to_calc
            }
            # skip this word if it doesn't have an embedding for all years around the event's time
            if any(none_val in hist.values() for none_val in [np.nan, 0, None]):
                continue
            score = hist[event_year + 1] / hist[event_year - 1]
            scores.append(score)
        return np.mean(scores) if scores and np.mean(scores) != np.nan else np.nan

    def _find_candidate_words_for_event(self, event, model, entities, query_avg):
        candidates = set()

        # take words with a high tf/idf score in the event's page
        top_tfidf = self.event_to_top_tfidf.get(event)
        if top_tfidf:
            candidates.update(
                model.get_key(w)
                for w in list(top_tfidf)[: int(self.candidates_lambda * self.k)]
                if w in model
            )
        else:
            logging.warning(f'the event {event} does not exist in the TF-IDF model')

        # interpolate with words similar to the query
        topn = (
            self.k
            if not candidates
            else int((1 - self.candidates_lambda) * self.k) + len(entities)
        )
        candidates.update(
            w
            for w, p in model.most_similar(
                [query_avg],
                topn=topn,
                filter_func=lambda word: word in model and not model.is_entity(word),
            )
        )

        candidates = candidates - set(entities)
        return list(candidates)

    def find_words_based_on_events(
        self, entities, year_event_scores, max_words_per_event
    ):
        use_temporal_models = len(self.models_manager.year_to_model) > 1
        query_avg = (
            self.global_model.get_average_vector(entities, require_all=True)
            if not use_temporal_models
            else None
        )

        # look for relevant words (based on the events)
        event_word_score = {}
        for year, event_scores in year_event_scores.items():
            model = (
                self.models_manager[year] if use_temporal_models else self.global_model
            )
            if not model or any(entity not in model for entity in entities):
                # fallback: use the global model
                model = self.global_model
                if not model or any(entity not in model for entity in entities):
                    continue
            if use_temporal_models:
                query_avg = model.get_average_vector(entities, require_all=True)

            for event, score in event_scores:

                if event not in model:
                    # logging.info(f'Ditching {event} because it doesn\'t exist in the model')
                    continue
                candidates = self._find_candidate_words_for_event(
                    event, model, entities, query_avg
                )
                if not candidates:
                    logging.info(f'Event "{event}" has no candidates')
                    continue

                # semantic similarity of each word (candidate) and the query
                query_similarities = model.similarities(query_avg, candidates)

                # similarity with this event
                event_similarities = model.similarities(event, candidates)

                # similarity of the event and the query (duplicated over all word candidates)
                event_query_similarities = np.array(
                    [model.similarity(event, query_avg)] * len(candidates)
                )

                all_scores = [
                    query_similarities * 3,
                    event_similarities,
                    event_query_similarities,
                ]

                top_tfidf = self.event_to_top_tfidf.get(event)
                if top_tfidf:
                    # TF/IDF of the candidate in this event's page
                    tf_idf_scores = [
                        top_tfidf.get(model.get_word(candidate), 0)
                        for candidate in candidates
                    ]
                    all_scores.append(np.array(tf_idf_scores))

                    if use_temporal_models:
                        # score of relevance to the event's neighbors
                        event_neighbors = [w for w in list(top_tfidf)[:5] if w in model]
                        neighbors_scores = [
                            self._calc_temporal_relevance(
                                candidate, year, event_neighbors
                            )
                            for candidate in candidates
                        ]
                        neighbors_scores = utils.normalize(neighbors_scores)
                        all_scores.append(np.array(neighbors_scores))

                final_scores = np.nanmean(all_scores, axis=0)

                # take the words with top scores
                max_words_margin = int(3 * max_words_per_event)
                positive_word_indices = utils.argpartition(
                    final_scores, max_words_margin
                )
                word_score = [
                    (
                        word_i,
                        model.get_word(candidates[word_i]),
                        round(float(final_scores[word_i]), 2),
                    )
                    for word_i in positive_word_indices
                ]

                word_score = self.filter_and_take_top_words(
                    word_score, entities, k=max_words_per_event
                )
                event_word_score[event] = word_score

        if not event_word_score:
            logging.info('* No candidate expansions were found')
            return None

        # aggregate all words of all events and sort the words
        word_score = [
            word_score
            for word_score_list in event_word_score.values()
            for word_score in word_score_list
        ]  # flatten the list

        word_score = self.filter_and_take_top_words(word_score, entities, k=self.k)
        return word_score

    def expand_using_events(self, entities, k=None):
        if k is None:
            k = self.k
        event_score_threshold = 0.004

        # tokenize and remove stopwords
        entities = utils.tokenize(' '.join(entities), remove_stopwords=True)

        key_years = self.models_manager.get_all_years()

        # for each of the detected years, look for relevant events
        year_event_score = self.onthology.find_events_for_entities(
            entities,
            self.event_detector,
            event_score_threshold,
            key_years,
            min_score=self.min_score,
        )

        if not year_event_score:  # no events were found
            logging.info('* No events were found beyond the threshold')
            return None

        # find relevant words based on the events
        event_scores = [
            (event, score)
            for year, event_scores in year_event_score.items()
            for (event, score) in event_scores
        ]
        if not event_scores:  # no events were found
            logging.info('** No events were found beyond the threshold')
            return None
        max_words_per_event = max(10, k // len(event_scores) * 2)
        word_scores = self.find_words_based_on_events(
            entities, year_event_score, max_words_per_event
        )

        # use the top k words as query expansion_score
        expansions = (
            {word: score for (word_i, word, score) in word_scores[:k]}
            if word_scores
            else {}
        )
        return expansions
