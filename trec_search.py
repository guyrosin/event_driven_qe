import logging
import os
from enum import Enum, auto
from pathlib import Path

from tqdm import tqdm
from trectools import TrecTopics, TrecTerrier, TrecQrel, TrecEval

import utils
from models_manager import ModelsManager, SpecificModelType
from qe_multiple_entities import MultipleEntitiesQEMethod, QEMultipleEntities
from qe_single_entity import QEMethod, QESingleEntity
from word2vec_wiki2vec_model import Word2VecWiki2VecModel
from word_onthology import WordOnthology, EventDetector

terrier_path = "/path-to-terrier/terrier-project-5.1/bin"
trec_corpora_dir = '/path-to-trec-directory'


class Dataset(Enum):
    Robust04 = (auto(),)
    AP_Disk1_3 = (auto(),)
    WSJ_Disk1_2 = (auto(),)
    TREC12 = (auto(),)


dataset_to_qrel_file = {
    Dataset.Robust04: 'qrels.robust04.txt',
    Dataset.AP_Disk1_3: 'qrels.ap.txt',
    Dataset.WSJ_Disk1_2: 'qrels.wsj.txt',
    Dataset.TREC12: 'qrels.51-200.disk1.disk2.txt',
}

dataset_to_topics_file = {
    Dataset.Robust04: 'topics.robust04.txt',
    Dataset.AP_Disk1_3: 'topics.51-200.txt',
    Dataset.WSJ_Disk1_2: 'topics.51-200.txt',
    Dataset.TREC12: 'topics.51-200.txt',
}

wiki2vec_global_model_path = 'path-to-wikipedia2vec-model'

candidates_lambda_dict = {
    Dataset.Robust04: 0.8,
    Dataset.AP_Disk1_3: 0.8,
    Dataset.WSJ_Disk1_2: 0.9,
    Dataset.TREC12: 0.8,
}


def init_index_location(trec_corpora_dir):
    dataset_to_index_location = {
        Dataset.Robust04: trec_corpora_dir + '/terrier/robust_2004_index',
        Dataset.AP_Disk1_3: trec_corpora_dir + '/terrier/AP_disk1-3_index',
        Dataset.WSJ_Disk1_2: trec_corpora_dir + '/terrier/WSJ_disk1_2_index',
        Dataset.TREC12: trec_corpora_dir + '/terrier/TREC12_index',
    }
    return dataset_to_index_location


class TrecHelper:
    def __init__(self, dataset, qe_method, k):
        self.dataset = dataset
        self.k = k
        self.qe_method = qe_method
        self.qe_multiple_entities = None

    def expand_topics(self, topics, include_unexpanded=False):
        args = [dict(query=query) for query in topics.topics.values()]
        topic_ids = topics.topics.keys()
        topic_expansion_score = dict(
            zip(
                topic_ids,
                tqdm(
                    [self.expand_query(**cur_args) for cur_args in args],
                    total=len(args),
                ),
            )
        )
        if not include_unexpanded:  # remove topics without expansion
            topic_expansion_score = {
                topic_id: exp_score
                for topic_id, exp_score in topic_expansion_score.items()
                if exp_score
            }
        return topic_expansion_score

    def expand_query(self, query):
        if self.qe_method.name.startswith('event'):
            logging.info(f'Query: {query}')
        entities = utils.tokenize(query)
        expansion_score = self.qe_multiple_entities.expand(entities)
        if not expansion_score:
            return None
        expansion_str = ' '.join(expansion_score.keys())
        logging.info(f'Query: "{query}"  -->  "{expansion_str}"')
        return expansion_score


class TrecGlobalHelper(TrecHelper):
    def __init__(self, dataset, qe_method, k, candidates_lambda=None):
        super().__init__(dataset, qe_method, k)
        global_model = Word2VecWiki2VecModel(model_path=wiki2vec_global_model_path)
        single_qe_method = QESingleEntity(
            QEMethod.specific_word2vec, k=self.k, global_model=global_model
        )
        self.qe_multiple_entities = QEMultipleEntities(
            single_qe_method,
            two_entities_qe_method=self.qe_method,
            k=k,
            candidates_lambda=candidates_lambda,
        )


class TrecTemporalHelper(TrecHelper):
    def __init__(
        self, dataset, qe_method, k, just_global=False, candidates_lambda=None
    ):
        super().__init__(dataset, qe_method, k)
        self.models_manager = (
            self.build_models(just_global)
            if self.qe_method != MultipleEntitiesQEMethod.none
            else None
        )
        single_qe_method = QESingleEntity(
            QEMethod.specific_word2vec, k=self.k, models_manager=self.models_manager
        )
        self.qe_multiple_entities = QEMultipleEntities(
            single_qe_method,
            two_entities_qe_method=self.qe_method,
            k=k,
            candidates_lambda=candidates_lambda,
        )

    def build_models(self, just_global=False):
        models_dir = 'path-to-temporal-models-dir'
        models_manager = ModelsManager(
            models_dir,
            global_model_path=wiki2vec_global_model_path,
            global_model_type=SpecificModelType.Wiki2Vec,
            specific_model_type=SpecificModelType.Wiki2Vec,
        )
        if just_global:
            models_manager.load_models(just_global=True, years=list(range(1980, 2019)))
        else:
            models_manager.load_models()
        return models_manager


class EventsSearchHelper(TrecTemporalHelper):
    def __init__(
        self,
        dataset,
        qe_method,
        k,
        event_detector=None,
        use_projected_models=False,
        candidates_lambda=None,
    ):
        super().__init__(
            dataset,
            qe_method,
            k,
            just_global=not use_projected_models,
            candidates_lambda=candidates_lambda,
        )
        self.event_detector = event_detector
        models_manager = (
            self.build_models(just_global=not use_projected_models)
            if self.models_manager is None
            else self.models_manager
        )
        self.onthology = WordOnthology(
            models_manager, use_projected_models=use_projected_models
        )
        self.qe_multiple_entities.onthology = self.onthology
        self.qe_multiple_entities.event_detector = (
            event_detector if event_detector else EventDetector.WikipediaFrequency
        )
        self.single_qe_method = QESingleEntity(
            QEMethod.specific_word2vec, k=self.k, models_manager=self.models_manager
        )


def read_topics_from_file(dataset, data_dir, topics_file):
    topics = TrecTopics()
    topic_num_regex = r'Number:\s+(\d+)'
    topic_title_regex = None if dataset == Dataset.Robust04 else r'Topic:\s+(.+)'
    querytext_tag = 'title'
    topics.read_topics_from_file(
        os.path.join(data_dir, topics_file),
        topic_tag='top',
        numberid_tag='num',
        querytext_tag=querytext_tag,
        title_regex=topic_title_regex,
        number_attr=False,
        number_regex=topic_num_regex,
    )
    topics.topics = {
        topic_id: utils.tokenize(topic, to_str=True)
        for topic_id, topic in topics.topics.items()
    }
    return topics


def translate_trec_metric(name):
    return name.upper().replace('_', '@')


def evaluate_trec_run(trec_run, qrels):
    te = TrecEval(trec_run, qrels)
    p10 = te.get_precision(depth=10)
    ndcg10 = te.get_ndcg(depth=10)
    map_score = te.get_map(depth=1000)
    logging.info(f'P@10={p10:.4f}, ' f'NDCG@10={ndcg10:.4f}, ' f'MAP={map_score:.4f}')


def calc_MLE(term, query_terms):
    """
    MLE = tf(w, Q) / |Q| (frequency in the query)
    """
    return sum(1 for query_term in query_terms if query_term == term) / len(query_terms)


def get_weighting_model(qe_method):
    if qe_method == MultipleEntitiesQEMethod.none:
        return 'BM25'
    else:
        return 'TF_IDF'


def search(
    topics,
    qe_method,
    result_file,
    data_dir,
    index_location,
    QMIX,
    expansions=None,
    show_output=True,
    include_unexpanded=False,
):
    topics_file_path = os.path.join(
        data_dir, f'topics_to_search_by_{qe_method.name}.txt'
    )
    data_dir = os.path.abspath(data_dir)  # for the output run file
    final_topics = {}
    if expansions:  # apply weights
        for topic_id, topic in tqdm(topics.topics.items()):
            expansion = expansions.get(topic_id)
            if expansion:
                topic_entities = utils.tokenize(topic)
                # interpolate with MLE
                expansion = {term: QMIX * score for term, score in expansion.items()}
                for term in topic_entities:
                    weight = (1 - QMIX) * calc_MLE(term, topic_entities)
                    if term in expansion:
                        expansion[term] += weight
                    else:
                        expansion[term] = weight
                expansion = utils.get_top_items(expansion)
                expanded_topic = ' '.join(
                    [f'{term}^{max(score, 0):.3f}' for term, score in expansion.items()]
                )
            elif include_unexpanded:
                expanded_topic = topic
            else:
                continue
            final_topics[topic_id] = expanded_topic
    else:
        if include_unexpanded or qe_method == MultipleEntitiesQEMethod.none:
            final_topics = topics.topics
        else:
            logging.warning('Nothing to search for')
            return
    TrecTopics(final_topics).printfile(
        fileformat=f'terrier',
        filename=topics_file_path,
        debug=show_output,
        single_line_format=True,
    )
    trec_terrier = TrecTerrier(bin_path=terrier_path)
    model = get_weighting_model(qe_method)
    trec_run = trec_terrier.run(
        index=index_location,
        topics=topics_file_path,
        model=model,
        result_file=result_file,
        result_dir=data_dir,
        showoutput=show_output,
        debug=False,
        topics_single_line_format=True,
    )
    return trec_run


def create_helper(
    dataset,
    qe_method,
    num_of_expanding_terms=None,
    event_detector=None,
    use_projected_models=False,
    candidates_lambda=None,
):
    if qe_method.name.startswith('events'):
        return EventsSearchHelper(
            dataset,
            qe_method=qe_method,
            k=num_of_expanding_terms,
            event_detector=event_detector,
            use_projected_models=use_projected_models,
            candidates_lambda=candidates_lambda,
        )
    elif qe_method == MultipleEntitiesQEMethod.none:
        return TrecHelper(dataset, qe_method, num_of_expanding_terms)
    else:
        return TrecTemporalHelper(
            dataset,
            qe_method,
            num_of_expanding_terms,
            candidates_lambda=candidates_lambda,
        )


if __name__ == "__main__":
    data_dir = Path('data').absolute()
    dataset = Dataset.Robust04
    qe_method = MultipleEntitiesQEMethod.events_temporal
    k = 100
    filter_topics_for_events = False
    event_detector = EventDetector.WikipediaFrequency
    use_projected_models = (
        True if qe_method == MultipleEntitiesQEMethod.events_temporal else False
    )
    show_output = False
    run_filename_template = 'run.{}.{}.k{}.txt'
    # QMIX is used for weighting (interpolation parameter between the LM and MLE)
    QMIX = 0.6
    candidates_lambda = candidates_lambda_dict[dataset]

    data_dir = Path('data').absolute()
    dataset_to_index_location = init_index_location(trec_corpora_dir)
    qrel_file = dataset_to_qrel_file[dataset]
    qrels_file_path = data_dir / qrel_file
    index_location = dataset_to_index_location[dataset]
    topics_file = dataset_to_topics_file[dataset]
    qrels = TrecQrel(qrels_file_path)

    topics = read_topics_from_file(dataset, data_dir, topics_file)

    topics.topics = {
        topic_id: utils.tokenize(topic, to_str=True)
        for topic_id, topic in topics.topics.items()
    }

    result_file = run_filename_template.format(
        dataset.name.lower(), qe_method.name.lower(), k
    )

    helper = create_helper(
        dataset,
        qe_method,
        k,
        event_detector=event_detector,
        use_projected_models=use_projected_models,
        candidates_lambda=candidates_lambda,
    )

    topic_expansion_score = (
        helper.expand_topics(topics)
        if qe_method != MultipleEntitiesQEMethod.none
        else None
    )
    trec_run = search(
        topics,
        qe_method,
        result_file,
        data_dir,
        index_location,
        QMIX,
        expansions=topic_expansion_score,
        show_output=show_output,
    )
    if trec_run:
        evaluate_trec_run(trec_run, qrels)
