import re
import json
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for pkg, path in [("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet")]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()
_lemma_cache = {}


def _lemma(word):
    if word not in _lemma_cache:
        _lemma_cache[word] = _lemmatizer.lemmatize(word)
    return _lemma_cache[word]


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)  # punctuation, numbers, special characters
    words = [w for w in text.split()
             if w not in STOP_WORDS
             and not re.fullmatch(r"x{2,}", w)  # CFPB masks personal info as xxxx
             and len(w) > 1]
    return " ".join(_lemma(w) for w in words)


class WordTokenizer:
    """Maps words to integer ids. 0 is reserved for padding and 1 for unknown words."""

    def __init__(self, num_words=20000):
        self.num_words = num_words
        self.word_index = {}

    def fit_on_texts(self, texts):
        counts = Counter(w for t in texts for w in t.split())
        self.word_index = {"<OOV>": 1}
        for i, (word, _) in enumerate(counts.most_common(self.num_words - 2), start=2):
            self.word_index[word] = i
        return self

    def texts_to_sequences(self, texts):
        return [[self.word_index.get(w, 1) for w in t.split()] for t in texts]

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"num_words": self.num_words, "word_index": self.word_index}, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        tok = cls(d["num_words"])
        tok.word_index = d["word_index"]
        return tok
