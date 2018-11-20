import nltk
import string
import codecs
import nltk.data
import re

from nltk.corpus.reader import PlaintextCorpusReader
from nltk.stem import SnowballStemmer
from six import string_types
from nltk.tokenize import *
from nltk.corpus.reader.util import *
from nltk.corpus.reader.api import *
from nltk.corpus import stopwords
from nltk.text import TextCollection
from math import sqrt, log

class IRModels:
    """
    Quick Reader for a plain text corpus.
    This overrides methods to the PlaintextCorpusReader to extract non ponctuation, non stop words, lowercased and stemmed words.
    """

    def __init__(self, corpus):
        self._corpus = corpus
        self._freq = nltk.ConditionalFreqDist(
            [(fileid, token) for fileid in self._corpus.fileids() for token in self._corpus.words(fileid)])

    # def __init__(self, root, lang,
    #              fileids='.*\.txt',
    #              word_tokenizer=WordPunctTokenizer(),
    #              para_block_reader=read_blankline_block,
    #              punctuation=string.punctuation,
    #              encoding='utf8'):
    #
    #     sent_tokenizer = nltk.data.LazyLoader('tokenizers/punkt/' + lang + '.pickle')
    #     stop_words = stopwords.words(lang)
    #     word_stemmer = SnowballStemmer(lang)
    #
    #     PlaintextCorpusReader.__init__(self, root, fileids, word_tokenizer, sent_tokenizer, para_block_reader, encoding)
    #     self._corpus._word_tokenizer = word_tokenizer
    #     self._sent_tokenizer = sent_tokenizer
    #     self._para_block_reader = para_block_reader
    #     self._word_stemmer = word_stemmer
    #     self._lang = lang
    #     self._punctuation = punctuation
    #     self._stop_words = stop_words
    #     self._sent_tokenizer = sent_tokenize
    #     self._stop_words = stop_words
    #     self._freq = nltk.ConditionalFreqDist(
    #         [(fileid, token) for fileid in self._corpus.fileids() for token in self._corpus.words(fileid)])

    def stem(self, word):
        return word


    def tf(self, term, fileid=None, terms=None):
        """
        override the tf method using the formula freq(term) / max(freq(t) for t in text)
        instead of the default one: freq(term) / len(text)
        """
        if fileid is not None:
            text = self._corpus.words(fileid)
        elif len(terms):
            text = terms
        else:
            raise ValueError('tf method should be called either with fileid or terms params')

        term = self.stem(term)
        print('fileid', fileid)
        print('tf', float(text.count(term)) / max([float(text.count(t)) for t in text]))
        return float(text.count(term)) / max([float(text.count(t)) for t in text])

    def idf(self, term):
        """
        override the idf method by ensuring that term is in it's stemmed form
        """
        term = self.stem(term)
        N = len(self._corpus.fileids())
        ni = len([True for file in self._corpus.fileids() if self.freq(file, term)])

        if ni == 0: q = 0
        else: q = N / ni

        return log(q + 1)

    def tf_idf(self, term=None, fileid=None, terms=None):
        """
        :return the tf_idf value for all words of the given fileid if term is None,
         otherwise the tf_idf score for the give term
        """
        if fileid is not None:
            text = self._corpus.words(fileid)
        elif len(terms):
            text = terms
        else:
            raise ValueError('tf method should be called either with fileid or terms params')

        if term is None:
            return {self.stem(term): self.tf(term, terms=text) * self.idf(term) for term in text}
        else:
            return self.tf(term, terms=text) * self.idf(term)

    def freq(self, fileid, term=None):
        """
        :return the frequency value for all words of the given fileid if term is None,
        otherwise the frequency of the given term.
        """
        if term is not None:
            return self._freq[fileid][self.stem(term)]
        else:
            return self._freq[fileid]

    def find(self, term):
        """
        :return the frequency and the tf_idf score for the given term in each file of the corpus
        """
        term = self.stem(term)
        return [{'fileid': file, 'freq': self.freq(file, term), 'tf_idf': self.tf_idf(term=term, fileid=file)}
                for file in self._corpus.fileids()
                if self.freq(file, term) or self.tf_idf(term=term, fileid=file)]

    def match(self, query, fileid):
        """
        Apply boolean model for the given query and the given fileid
        :param query boolean query with operations (and, or, not). example: web and (pdf or video)
        :raise
        """

        def replace(word):
            word = word.group()
            if word in ['and', 'or', 'not']:
                return word
            else:
                return str(self.stem(word) in self._corpus.words(fileid))

        query = re.sub(r'\w+', replace, query)
        print('evaluating', query, 'for', fileid)
        return eval(query)

    def all_match(self, query):
        """
        :return all the documents that matches the given boolean query
        """
        print('all_match', query)
        return [file for file in self._corpus.fileids() if self.match(query, file)]

    def inner_product(self, query, fileid):
        """
        :return the inner product (query \intersect fileid)
        :rtype(double)
        """
        terms = [self.stem(t) for t in self._corpus._word_tokenizer.tokenize(query)]
        r = sum([self.tf_idf(term=term, fileid=fileid) * self.tf_idf(term=term, terms=terms) for term in terms])
        return round(r, 2)

    def dice_coef(self, query, fileid):
        """
        :return the dice coef ((2 * |query \intersect fileid|) \ (|query| + |fileid|))
        """
        terms = [self.stem(t) for t in self._corpus._word_tokenizer.tokenize(query)]
        X = sum([self.tf_idf(term=term, terms=terms) ** 2 for term in terms])
        Y = sum([self.tf_idf(term=term, fileid=fileid) ** 2 for term in terms])

        if X + Y == 0: return 0;
        r = 2 * self.inner_product(query, fileid) / (X + Y)
        return round(r, 2)

    def cosinus_measure(self, query, fileid):
        """
        :return the cosinus measure ((|query \intersect fileid|) \ (sqrt(|query|) + sqrt(|fileid|))
        """
        terms = [self.stem(t) for t in self._corpus._word_tokenizer.tokenize(query)]
        X = sum([self.tf_idf(term=term, terms=terms) ** 2 for term in terms])
        Y = sum([self.tf_idf(term=term, fileid=fileid) ** 2 for term in terms])

        if X + Y == 0: return 0;
        r = self.inner_product(query, fileid) / (sqrt(X) + sqrt(Y))
        return round(r, 2)

    def jaccard_coef(self, query, fileid):
        """
        :return the jaccard coef ((|query \intersect fileid|) \ (|query| + |fileid| - (|query \intersect fileid|)))
        """
        terms = [self.stem(t) for t in self._corpus._word_tokenizer.tokenize(query)]
        X = sum([self.tf_idf(term=term, terms=terms) ** 2 for term in terms])
        Y = sum([self.tf_idf(term=term, fileid=fileid) ** 2 for term in terms])
        inner_product = self.inner_product(query, fileid)

        if X + Y - inner_product == 0: return 0;
        r = inner_product / (X + Y - inner_product)
        return round(r, 2)
