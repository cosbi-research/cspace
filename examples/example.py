#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example script on howto use CSpace bio-medical embedding model
"""
import sys
from gensim.models import KeyedVectors
from gensim.models.phrases import Phrases, Phraser, ENGLISH_CONNECTOR_WORDS
from gensim.models import FastText
from gensim.models.fasttext import load_facebook_model
import compress_fasttext
from gensim.corpora import Dictionary

# utils functions for better sentence similarity
import spacy
import numpy as np
from utils import clean_and_normalize_word
def init_nlp():
    try:
        nlp = spacy.load('en_core_web_lg')   # load the language model
    except OSError:
        print("Please run:")
        print("python -m spacy download en_core_web_lg")
        print("To install the english language model 'en_core_web_lg'")
        sys.exit(1)
    nlp.remove_pipe('ner')               # remove the default NER

    return nlp

# keyedvectors format
cspace_path = 'cspace.kv.bin'
cspace_compressed_path = 'cspace.compressed.100k.bin'
fasttext_cspace_path = 'cspace.fasttext.bin'
dictionary_freq_path = 'cspace.dict.pkl'
bigram_path = 'cspace.bigrams.pkl'

# spacy pre-processing
_nlp = init_nlp()
# RECOMMENDED:
# load CSpace full model (KeyedVectors format)
# in keyedvectors format (CANNOT SYNTHESIZE WORDS NOT SEEN DURING TRAINING)
# mmap='r' share the same weights between multiple load call
cspace_model = KeyedVectors.load(cspace_path, mmap='r')

# load CSpace full model (FastText format)
# in FastText format (CAN SYNTHESIZE WORDS NOT SEEN DURING TRAINING)
#cspace_model = load_facebook_model(fasttext_cspace_path)

# load CSpace compressed model
# cspace_model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(cspace_compressed_path)

## AUXILIARY MODELS
# load composite words model based on co-occurrence of words
# (Mikolov et al. 2013)
cspace_phrases = Phrases.load(bigram_path)

dictionary = Dictionary.load(dictionary_freq_path)

# DISEASE-CONDITION RELATEDNESS
cspace_model.similarity('long_covid','myalgic_encephalomyelitis')
# 0.8102189
cspace_model.similarity('covid','myalgic_encephalomyelitis')
# 0.60641325
cspace_model.similarity('long_covid','fatigue')
# 0.8102189
cspace_model.similarity('long_covid','post-exertional_malaise')
# 0.76731175
cspace_model.similarity('long_covid','anxiety')
# 0.5558557
cspace_model.similarity('long_covid','tiredness')
# 0.54872346
cspace_model.similarity('long_covid','cough')
# 0.5306381
cspace_model.similarity('long_covid','dizziness')
# 0.564164
cspace_model.similarity('long_covid','joint_pain')
# 0.52039963
cspace_model.similarity('long_covid','joint_or_muscle_pain')
# 0.50613415
cspace_model.similarity('long_covid','rash')
# 0.45096865

# CONCEPT SIMILARITY
cspace_model.most_similar(positive=['fatigue','myalgic_encephalomyelitis'])
# [('fatigue_and_chronic_fatigue_syndrome', 0.9305593967437744), ('myalgic_encephalomyelitis_and_chronic_fatigue_syndrome', 0.9286814332008362), ('/chronic_fatigue', 0.9204096794128418), ('myalgic_encephalomyelitis_or_chronic_fatigue_syndrome', 0.9200829863548279), ('chronic_fatigue_and_chronic_fatigue_syndrome', 0.9173030257225037), ('chronic_fatigue_or_chronic_fatigue_syndrome', 0.9164567589759827), ('fatigue_in_chronic_fatigue_syndrome', 0.9162179231643677), ('chronic_fatigues', 0.9146023988723755), ('fatigue_post-exertional', 0.9129915833473206), ('fibromyalgia_and_chronic_fatigue', 0.9122573137283325)]


# SENTENCE SIMILARITY
sentence_baseline = 'Studies into long COVID suggest many overlaps with ME/CFS'.lower()
sentence_positive = 'Twenty-five out of 29 known ME/CFS symptoms were reported by at least one selected long COVID study'.lower()
sentence_negative = 'Mean serum or sera concentration time profiles of mRNA-encoded protein product'.lower()

baseline_doc=_nlp(sentence_baseline)
positive_doc=_nlp(sentence_positive)
negative_doc=_nlp(sentence_negative)

baseline_post = clean_and_normalize_word(baseline_doc, remove_stopwords=True)
positive_post = clean_and_normalize_word(positive_doc, remove_stopwords=True)
negative_post = clean_and_normalize_word(negative_doc, remove_stopwords=True)

# apply phrases model
baseline = cspace_phrases[baseline_post]
# ['study', 'long', 'covid', 'suggest', 'overlap', 'cfs']
positive = cspace_phrases[positive_post]
# ['know', 'cfs', 'symptom', 'report', 'select', 'long', 'covid', 'study']
negative = cspace_phrases[negative_post]
# ['mean', 'serum', 'sera', 'concentration', 'time', 'profile', 'mrna', 'encode', 'protein', 'product']

## Word-Mover-Distance

related_distance = cspace_model.wmdistance(baseline, positive)
related_distance
# 0.368

unrelated_distance = cspace_model.wmdistance(baseline, negative)
unrelated_distance
# 0.802

## Weighted Average Similarity
# https://openreview.net/pdf?id=SyK00v5xx
def wa_embed(sent):
    sl=[]
    A=3e-4
    for t in sent:
        try:
            coeff = A/(A+dictionary.dfs[dictionary.token2id[t]]/dictionary.num_docs)
            sl.append(coeff*cspace_model[t])
        except KeyError:
            # missing
            pass
    sent1_embds = np.array(sl)
    sent1_avg_embd = sent1_embds.mean(axis=0)
    return sent1_avg_embd

baseline_embd = wa_embed(baseline)
positive_embd = wa_embed(positive)
negative_embd = wa_embed(negative)

cosine_similarity_positive = np.dot(baseline_embd, positive_embd.T) / (np.linalg.norm(baseline_embd)*np.linalg.norm(positive_embd))
cosine_similarity_positive
# 0.99

cosine_similarity_negative = np.dot(baseline_embd, negative_embd.T) / (np.linalg.norm(baseline_embd)*np.linalg.norm(negative_embd))
cosine_similarity_negative
# 0.62

