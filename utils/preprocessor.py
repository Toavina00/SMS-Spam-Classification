from typing import Tuple
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder

import numpy as np

from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer


class Preprocessor:
    def __init__(
        self,
        stop_words: bool = True,
        min_df: float | int = 1, 
        max_df: float | int = 1, 
        ngram: Tuple[int, int] | int = 1, 
        X_train: np.ndarray = None, 
        y_train: np.ndarray = None
    ):
        """
        Parameters
        ----------
        stop_words : bool, optional
            Whether to remove English stop words from the text data. The default is True.
        min_df : float or int, optional
            Minimum frequency of words to be included in the vocabulary. The default is 1.
        max_df : float or int, optional
            Maximum frequency of words to be included in the vocabulary. The default is 1.
        ngram : int or tuple of int, optional
            The range of n-grams to be extracted from the text data. The default is 1.
        X_train : array-like of shape (n_samples, n_features), optional
            The training data. If provided, the vocabulary will be built from the training data.
        y_train : array-like of shape (n_samples, n_features), optional
            The target values of the training data. If provided, the one-hot encoder will be fit to the target values.

        Attributes
        ----------
        ngram : int or tuple of int
            The range of n-grams to be extracted from the text data.
        stop_words : bool
            Whether to remove English stop words from the text data.
        min_df : float or int
            Minimum frequency of words to be included in the vocabulary.
        max_df : float or int
            Maximum frequency of words to be included in the vocabulary.
        word_to_idx : dict
            A mapping from words to indices.
        idx_to_word : dict
            A mapping from indices to words.
        vocab_size : int
            The size of the vocabulary.
        """
        
        self.ngram = ngram
        self.stop_words = stop_words
        self.min_df = min_df
        self.max_df = max_df
        self.word_to_idx = None
        self.idx_to_word = None
        self.__count_vectorizer = None
        self.__one_hot_encoder = None
        self.vocab_size = None

        if X_train is not None:
            self.__count_vectorizer = CountVectorizer(
                stop_words='english' if self.stop_words else None,
                min_df=self.min_df,
                max_df=self.max_df,
                token_pattern=r'''(?x)
                    (?:[A-Z]\.)+[A-Z]?         |  # Abbreviations like U.S.A.
                    <[^>]+>                    |  # HTML tags
                    \$?\d+(?:\.\d+)?%?         |  # Numbers, currency and percentages, e.g. 3.14, $12.40, 82%
                    \w+(?:[-']\w+)*               # Words and words with apostrophes/hyphens e.g., can't, we're
                ''',
                ngram_range=self.ngram if isinstance(self.ngram, tuple) else (self.ngram, self.ngram),
                preprocessor=WordNetLemmatizer().lemmatize,
            )
            self.__count_vectorizer.fit(X_train)
            self.word_to_idx = {w: i+1 for i, (w, _) in enumerate(sorted(self.__count_vectorizer.vocabulary_.items(), key=lambda x: x[1], reverse=True))}
            self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}
            self.idx_to_word[0] = "<oov>"
            self.vocab_size = len(self.word_to_idx)
        
        if y_train is not None:
            self.__one_hot_encoder = OneHotEncoder(sparse_output=False)
            self.__one_hot_encoder.fit(y_train.reshape(-1, 1))

    def input_preprocess(self, input: np.ndarray):

        """
        Preprocesses the input text data by converting it into the numerical representation based on the vocabulary built from the training data.

        Parameters
        ----------
        input : array-like of shape (n_samples,)
            The input text data.

        Returns
        -------
        array-like of shape (n_samples, max_length)
            The numerical representation of the input text data.
        """
        
        if self.__count_vectorizer is None:
            raise Exception("CountVectorizer is not initialized")
        
        analyzer = self.__count_vectorizer.build_analyzer()
        out = []
        max_len = 0
        for x in input:
            idx = [self.word_to_idx.get(w, 0) for w in analyzer(x)]
            max_len = max(max_len, len(idx))
            out.append(idx)
        out = [x + [0] * (max_len - len(x)) for x in out]
        return np.array(out)
    
    def output_preprocess(self, output: np.ndarray):

        """
        Preprocesses the output by one-hot encoding.

        Parameters
        ----------
        output : array-like of shape (n_samples,)
            The output to be preprocessed.

        Returns
        -------
        array-like of shape (n_samples, n_classes)
            The one-hot encoded output.
        """

        if self.__one_hot_encoder is None:
            raise Exception("OneHotEncoder is not initialized")
        
        return self.__one_hot_encoder.transform(output.reshape(-1, 1))