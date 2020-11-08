import logging
import os
from pathlib import Path
from gensim import matutils

from gensim.models import Word2Vec, KeyedVectors
from gensim.parsing import PorterStemmer
import numpy as np
import scipy
from scipy.sparse.csr import csr_matrix


class Word2VecModel:
    """
    Wrapper for a word2vec model
    """

    def __init__(
        self, year=None, dir_path=None, file=None, model_path=None, model=None
    ):
        self.year = year
        if model:
            self.wv = model.wv
            return
        if not model_path:
            model_path = Path(dir_path) / file

        if os.path.exists(model_path):  # load the model
            if str(model_path).endswith('.txt'):
                self.wv = KeyedVectors.load_word2vec_format(str(model_path))
            else:
                self.wv = KeyedVectors.load(str(model_path), mmap='r')
            if isinstance(self.wv, Word2Vec):
                self.wv = self.wv.wv
        else:
            raise FileNotFoundError(f"model path {model_path} doesn't exist")

        self.stemmer = PorterStemmer()

    def __contains__(self, word):
        return self.get_key(word) is not None

    def __getitem__(self, word):
        return self.word_vec(word)

    def get_key(self, word):
        """
        If the given word exists in the model, return it
        """
        word = str(word)
        word_variations = [
            word,
            word.lower(),
            word.capitalize(),
            self.stemmer.stem(word),
        ]
        for word_variation in word_variations:
            if word_variation in self.wv:
                return word_variation
        return None

    def similarity(self, word_or_vec1, word_or_vec2):
        if not isinstance(word_or_vec1, str) or not isinstance(word_or_vec2, str):
            vec1 = (
                self.word_vec(word_or_vec1)
                if isinstance(word_or_vec1, str)
                else word_or_vec1
            )
            vec2 = (
                self.word_vec(word_or_vec2)
                if isinstance(word_or_vec2, str)
                else word_or_vec2
            )
            return (
                np.dot(matutils.unitvec(vec1), matutils.unitvec(vec2))
                if vec1 is not None and vec2 is not None
                else None
            )
        key1, key2 = self.get_key(word_or_vec1), self.get_key(word_or_vec2)
        return self.wv.similarity(key1, key2) if key1 and key2 else None

    def word_vec(self, word, use_norm=False):
        key = self.get_key(word)
        if not key:
            return None
        try:
            return self.wv.word_vec(key, use_norm)
        except TypeError:
            return None

    def get_word(self, key):
        """
        returns the string representation of a given key, if it exists in the model
        """
        return key if key in self.wv else None

    def init_sims(self, replace=False):
        if getattr(self.wv, 'vectors_norm', None) is not None:
            return
        try:
            self.wv.init_sims(replace=replace)
        except:
            if replace:
                self.wv.init_sims(replace=False)
                logging.warning(
                    'executed init_sims() with replace=False to avoid an exception'
                )
            else:
                raise

    def similar_by_word(self, word, topn=10):
        if word is None:
            return None
        return self.most_similar(positive=[word], topn=topn)

    def similar_by_vector(self, vector, topn=10, format_output=False):
        if vector is None:
            return None
        word_score_tuples = self.most_similar(positive=[vector], topn=topn)
        if format_output:
            word_score_tuples = [
                (self.get_word(word), score) for word, score in word_score_tuples
            ]
        return word_score_tuples

    def most_similar(
        self,
        positive=None,
        negative=None,
        topn=10,
        exclude_words=None,
        filter_func=None,
    ):
        if exclude_words or filter_func:
            return self.most_similar_filtered(
                positive, negative, topn, exclude_words, filter_func
            )
        if positive is None:
            positive = []
        if negative is None:
            negative = []
        if isinstance(positive, str) and not negative:
            # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
            positive = [positive]
        # note that the given "words" can be either strings or vectors
        positive = [
            self.get_key(pos) if isinstance(pos, str) else pos for pos in positive
        ]
        negative = [
            self.get_key(neg) if isinstance(neg, str) else neg for neg in negative
        ]
        for item in positive + negative:
            if (
                item is None
            ):  # item can be an ndarray, no need to compare to None explicitly
                return None
        return self.wv.most_similar(positive, negative, topn)

    def most_similar_filtered(
        self,
        positive=None,
        negative=None,
        topn=10,
        exclude_words=None,
        filter_func=None,
    ):
        """
        In case we want to exclude some words / include only words that pass a given filter
        """
        if not exclude_words and not filter_func:
            return self.most_similar(positive, negative, topn)
        similar_tuples = []
        total_top = 0
        coefficient = 1.5
        last_addition = topn
        while len(similar_tuples) < topn and coefficient < 100:
            addition = int(last_addition * coefficient)
            total_top += addition
            additional_tuples = self.most_similar(positive, negative, topn=total_top)
            if additional_tuples is None:
                return None
            additional_tuples = additional_tuples[-addition:]
            if exclude_words:
                additional_tuples = [
                    (word, score)
                    for word, score in additional_tuples
                    if word not in exclude_words
                ]
            if filter_func:
                additional_tuples = [
                    (word, score)
                    for word, score in additional_tuples
                    if filter_func(word)
                ]
            remaining = topn - len(similar_tuples)
            additional_tuples = additional_tuples[:remaining]
            similar_tuples.extend(additional_tuples)
            coefficient *= 2
        return similar_tuples

    def is_entity(self, word_or_entity):
        return False

    def get_average_vector(self, words, require_all=True):
        vectors = [self[word] for word in words]
        if not vectors or (require_all and any(vec is None for vec in vectors)):
            return
        avg_vec = (
            np.mean(vectors, axis=0)
            if require_all
            else np.nanmean(np.array(vectors, dtype=np.float), axis=0)
        )
        return avg_vec

    @staticmethod
    def cosine_similarities(vector_1, vectors_all):
        """
        Adapted from gensim.models.KeyedVectors.cosine_similarities for sparse matrices.
        Compute cosine similarities between one vector and a set of other vectors.

        Parameters
        ----------
        vector_1 : numpy.ndarray
            Vector from which similarities are to be computed, expected shape (dim,).
        vectors_all : numpy.ndarray or csr_matrix
            For each row in vectors_all, distance from vector_1 is computed, expected shape (num_vectors, dim).

        Returns
        -------
        numpy.ndarray
            Contains cosine distance between `vector_1` and each row in `vectors_all`, shape (num_vectors,).
        """
        norm = np.linalg.norm(vector_1)
        if norm == 0:
            return None
        if isinstance(vectors_all, csr_matrix):
            all_norms = scipy.sparse.linalg.norm(vectors_all, axis=1)
            dot_products = vectors_all.dot(vector_1)
        else:
            all_norms = np.linalg.norm(vectors_all, axis=1)
            dot_products = np.dot(vectors_all, vector_1)
        similarities = dot_products / (norm * all_norms)
        return similarities

    def similarities(self, word_or_vector, other_words=()):
        """
        Compute cosine similarities from given word or vector to all words in `other_words`.
        If `other_words` is empty, return distance between `word_or_vectors` and all words in vocab.

        Parameters
        ----------
        word_or_vector : {str, numpy.ndarray}
            Word or vector from which distances are to be computed.
        other_words : iterable of str
            For each word in `other_words` distance from `word_or_vector` is computed.
            If None or empty, distance of `word_or_vector` from all words in vocab is computed (including itself).

        Returns
        -------
        numpy.array
            Array containing distances to all words in `other_words` from input `word_or_vector`.

        Raises
        -----
        KeyError
            If either `word_or_vector` or any word in `other_words` is absent from vocab.

        """
        input_vector = (
            self.word_vec(word_or_vector)
            if isinstance(word_or_vector, str)
            else word_or_vector
        )
        if input_vector is None:
            return None

        wv_vectors = self.wv.vectors if hasattr(self.wv, 'vectors') else self.wv.syn0
        if other_words:
            other_indices = [self.wv.vocab[word].index for word in other_words]
            other_vectors = wv_vectors[other_indices]
        else:
            other_vectors = wv_vectors
        similarities = self.cosine_similarities(input_vector, other_vectors)
        return similarities
