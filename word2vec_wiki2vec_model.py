import re

from word2vec_model import Word2VecModel


class Word2VecWiki2VecModel(Word2VecModel):
    """
    Wrapper for a wikipedia2vec model
    """

    def __init__(self, model_path, year=None):
        super().__init__(model_path=model_path, year=year)
        self.entity_prefix = 'ENTITY/'

    def get_key(self, word):
        """
        If the given word exists in the model, returns it. Otherwise, if it's an entity, returns its entity name
        """
        if word in self.wv:
            return word
        if word.lower() in self.wv:
            return word.lower()
        stemmed = self.stemmer.stem(word)
        if stemmed in self.wv:
            return stemmed
        entity_name = self.entity_prefix + '_'.join([w for w in re.split('[ _]', word)])
        if entity_name in self.wv:  # check if 'word' is an entity
            return entity_name
        entity_name = self.entity_prefix + '_'.join(re.split('[ _]', word)).title()
        if entity_name in self.wv:  # check if 'word' is an entity
            return entity_name
        entity_name = self.entity_prefix + '_'.join(re.split('[ _]', word)).capitalize()
        if entity_name in self.wv:  # check if 'word' is an entity
            return entity_name
        return None

    def get_word(self, word_or_entity):
        if self.is_entity(word_or_entity):
            return word_or_entity.split(self.entity_prefix, 1)[1].replace('_', ' ')
        return word_or_entity if word_or_entity in self.wv else None

    def is_entity(self, word_or_entity):
        """
        returns true if the given string is an entity
        """
        if isinstance(word_or_entity, str):
            return word_or_entity.startswith(self.entity_prefix)
        key = self.get_key(word_or_entity)
        return key and key.startswith(self.entity_prefix)
