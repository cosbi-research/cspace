#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gensim.models.phrases import Phrases, Phraser, ENGLISH_CONNECTOR_WORDS
from  gensim.models.word2vec import LineSentence
import sys

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.stderr.write("./bigram.py unigram_dump_file base_name\n")
        sys.exit(1)

    dump_file = sys.argv[1]
    name = sys.argv[2]
    bigram_model_name = name+'.bigrams.pkl'
    phrases_sentences = LineSentence(dump_file)
    phrases = Phrases(phrases_sentences,
                      max_vocab_size=100000000,
                      threshold=1, # for very large corporas
                      connector_words=ENGLISH_CONNECTOR_WORDS) # automatically detects bi-grams
    phrases.save(bigram_model_name)

