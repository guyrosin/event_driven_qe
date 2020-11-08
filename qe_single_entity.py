from enum import auto, IntEnum

import models_manager


class QEMethod(IntEnum):
    none = auto()
    global_word2vec = auto()
    specific_word2vec = auto()


class QESingleEntity:
    def __init__(self, qe_method=None, k=2, models_manager=None, global_model=None):
        self.qe_method = qe_method
        self.k = k
        self.models_manager = models_manager
        self.global_model = (
            global_model
            if global_model or not models_manager
            else models_manager[models_manager.STATIC_YEAR]
        )

    def expand_entity(self, entity, time=None, qe_method=None, k=None):
        """
        Apply QE with the selected method and return the expanded query
        """
        if qe_method is None:
            qe_method = self.qe_method
        if k is None:
            k = self.k
        entity = entity.replace('_', ' ').title()

        if qe_method == QEMethod.none:
            related_tuples = []
        elif qe_method == QEMethod.global_word2vec:
            related_tuples = self.expand_entity_word2vec(
                entity, models_manager.STATIC_YEAR, k=k
            )
        elif qe_method == QEMethod.specific_word2vec:
            related_tuples = self.expand_entity_word2vec(entity, time, k=k)
        else:
            raise ValueError('Unknown QE method: {}'.format(qe_method))
        return related_tuples

    def expand_vector(self, vector, time=None, qe_method=None, k=None, entity=None):
        """
        Apply QE with the selected method and return the expanded query
        """
        if qe_method is None:
            qe_method = self.qe_method
        if k is None:
            k = self.k

        if qe_method == QEMethod.none:
            return []
        elif qe_method == QEMethod.global_word2vec:
            related_tuples = self.expand_vector_word2vec(
                vector, models_manager.STATIC_YEAR, k=k, entity=entity
            )
        elif qe_method == QEMethod.specific_word2vec:
            related_tuples = self.expand_vector_word2vec(
                vector, time, k=k, entity=entity
            )
        else:
            raise ValueError('Unknown QE method for vectors: {}'.format(qe_method))
        return related_tuples

    def expand_entity_word2vec(self, entity, time, k=None):
        """
        Get an entity and a timestamp.
        Return tuples (term, score) of the k closest terms from the word2vec model of that time period.
        """
        w2v_model = (
            self.global_model
            if time == models_manager.STATIC_YEAR
            else self.models_manager[time]
            if self.models_manager
            else None
        )
        if not w2v_model:
            return None
        if k is None:
            k = self.k
        key = w2v_model.get_key(entity)
        if not key:
            return None
        vector = w2v_model.word_vec(entity)
        return self.expand_vector_word2vec(vector, time, k, entity)

    def expand_vector_word2vec(self, vector, time, k=None, entity=None):
        """
        Get an entity and a timestamp.
        Return tuples (term, score) of the k closest terms from the word2vec model of that time period.
        `entity` can be either a string or a list of entities that compose the given vector, so they won't be returned.
        """
        w2v_model = (
            self.global_model
            if time == models_manager.STATIC_YEAR
            else self.models_manager[time]
            if self.models_manager
            else None
        )
        if not w2v_model:
            return None
        if k is None:
            k = self.k
        topn = k
        if entity:  # retrieve extra neighbor(s), and then remove the given word(s)
            if isinstance(entity, str):
                entity = [entity]
            topn += len(entity)
        related_tuples = w2v_model.most_similar(
            [vector],
            topn=topn,
            filter_func=lambda word: word in w2v_model
            and not w2v_model.is_entity(word),
        )
        if entity:
            related_tuples = [
                (score, word) for score, word in related_tuples if word not in entity
            ][:k]
        return related_tuples
