import logging
from collections import defaultdict
from concurrent.futures.process import ProcessPoolExecutor
from pathlib import Path

import json
from scipy.optimize import minimize
from scipy.spatial import distance
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from models_manager import ModelsManager
from word2vec_wiki2vec_model import Word2VecWiki2VecModel

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO
)


def calc_distances(x, locations, distance_metric):
    return [distance_metric(x, loc) for loc in locations]


def calc_mse(x, locations, distances, distance_metric):
    target_distances = calc_distances(x, locations, distance_metric=distance_metric)
    mse = mean_squared_error(distances, target_distances)
    return mse


def find_anchor_words(target_model, source_model, word, anchors_num):
    # find anchor words that belong to the target model using an exponential backoff-like method
    anchor_words = []
    coefficient = max(200, anchors_num * 15)
    i = 0
    while len(anchor_words) < anchors_num and i < 5:
        anchor_words = [
            w
            for w, score in source_model.similar_by_word(
                word, anchors_num + coefficient
            )
            if score != 1.0 and w in target_model
        ]
        coefficient *= 2
        i += 1
    if not anchor_words:
        logging.info(f'found no anchors for "{word}"')
        return None
    anchor_words.sort(
        key=lambda w: source_model.wv.vocab.get(source_model.get_key(w)).index
    )
    anchor_words = anchor_words[:anchors_num]
    return anchor_words


def project_word(target_model, source_model, word, anchors_num, distance_metric):
    if word not in source_model:
        # logging.info(f'"{word}" not found in the source model')
        return word, None
    # find anchor words that belong to the target model
    anchor_words = find_anchor_words(target_model, source_model, word, anchors_num)
    if not anchor_words:
        logging.info(f'found no anchors for "{word}"')
        return word, None
    anchor_words.sort(
        key=lambda w: source_model.wv.vocab.get(source_model.get_key(w)).index
    )
    anchor_words = anchor_words[:anchors_num]
    source_anchor_locations = [source_model.word_vec(anchor) for anchor in anchor_words]
    source_distances = calc_distances(
        source_model.word_vec(word, use_norm=True),
        source_anchor_locations,
        distance_metric,
    )
    target_anchor_locations = [
        target_model.word_vec(anchor, use_norm=True) for anchor in anchor_words
    ]
    initial_location = target_anchor_locations[0]
    result = minimize(
        calc_mse,
        initial_location,
        args=(target_anchor_locations, source_distances, distance_metric),
        method='L-BFGS-B',
    )
    key = source_model.get_key(word)
    return key, result.x


def project_words(target_model, source_model, words, anchors_num, distance_metric):
    if not words:
        return target_model
    args = [
        (target_model, source_model, word, anchors_num, distance_metric)
        for word in words
        if word in source_model
    ]
    with ProcessPoolExecutor() as executor:
        source_model.init_sims()  # so that we won't have to do that repeatedly while multi-processing
        new_words = dict(tqdm(executor.map(project_word, *zip(*args)), total=len(args)))
    new_words = {
        word: vector for word, vector in new_words.items() if vector is not None
    }
    target_model.wv.add(list(new_words.keys()), list(new_words.values()))
    target_model.init_sims(replace=True)
    return target_model


def project_to_temporal_models(
    models_dir,
    min_year,
    max_year,
    global_model,
    event_year,
    distance_metric,
    enrich_only_events_of_same_year=False,
):
    models_manager = ModelsManager(models_dir, from_year=min_year, to_year=max_year)
    models_manager.load_models()

    if enrich_only_events_of_same_year:
        year_to_event = defaultdict(list)
        for event, year in event_year.items():
            if models_manager.from_year <= year <= models_manager.to_year:
                year_to_event[year].append(event)

    new_dir = models_dir / 'enriched'
    new_dir.mkdir(parents=True, exist_ok=True)

    for year in range(min_year, max_year + 1):
        model = models_manager[year]
        events_to_enrich = (
            year_to_event[year] if enrich_only_events_of_same_year else events
        )
        new_model = project_words(
            model, global_model, events_to_enrich, anchors_num, distance_metric
        )
        logging.info(
            f'model of {year} was enriched with {len(events_to_enrich)} events'
        )
        file_name = f'word2vec_nyt_{year}.kv'
        new_model.wv.save(str(new_dir / file_name))


if __name__ == '__main__':
    anchors_num = 30
    distance_metric = distance.cosine
    events = json.load(open('data/events_since1980.json', encoding='utf-8'))
    global_model = Word2VecWiki2VecModel('path-to-wikipedia2vec-model')

    min_year = 1981
    max_year = 2018
    project_only_events_of_same_year = True
    logging.info(
        f'enriching models from {min_year}-{max_year} using maximum {anchors_num} anchors'
    )
    models_dir = Path('path-to-temporal-models')
    event_year = json.load(open('data/event_year_since1980.json', encoding='utf-8'))
    project_to_temporal_models(
        models_dir,
        min_year,
        max_year,
        global_model,
        project_only_events_of_same_year,
        distance_metric,
    )
