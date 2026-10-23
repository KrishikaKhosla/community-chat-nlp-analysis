"""Shared test setup: make sure NLTK corpora are downloaded before any test runs."""

from src.clean import ensure_nltk_data
from src.sentiment import ensure_vader_lexicon

ensure_nltk_data()
ensure_vader_lexicon()
