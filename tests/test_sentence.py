#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to generate data displayed in Table 5 of paper
"""

from gensim.test.utils import datapath
from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_vectors,load_facebook_model
from gensim.models.phrases import Phrases, Phraser, ENGLISH_CONNECTOR_WORDS
from gensim.models import FastText,Word2Vec
import compress_fasttext
from gensim.corpora import Dictionary
import sys,os,json
import numpy as np
from scipy import stats
from preprocessing.utils import clean_and_normalize_word
import logging
import openai
import spacy
import scipy

classname = 'test_sentence'
logger = logging.getLogger(classname)
# MAX CUMULATIVE EXPLAINED VARIANCE
MAX_CUM_VE=0.6
# https://openreview.net/pdf?id=SyK00v5xx
A = 0.0003

def analogy(x1, x2, y1):
    result = model.most_similar(positive=[y1, x2], negative=[x1])
    return result[0][0]

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

def svd_embedding(embds, max_cum_ve):
    # inspired by
    # https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca
    U, s, Vh = np.linalg.svd(embds)
    # Calculate proportion of variance explained
    eig_vals = s**2 / (embds.shape[0]-1)
    ve = eig_vals / eig_vals.sum()
    cum_ve = np.cumsum(ve)
    # rank is the first index that explain max_cum_ve% of variance
    rank = np.argmax(cum_ve > max_cum_ve)
    # compute n principal components (rank) of embds
    # that explain at least max_cum_ve% of variance
    embds_pc = U[:,:rank] @ np.diag(s[:rank])
    # stack PC one on top of the other 
    embd = embds_pc.reshape((-1,1))
    return embd

def evaluate(model, model_name, phrases, dictionary, fasttext, openai_key):

    openai.api_key = openai_key
    _nlp = init_nlp()
    
    WPAIR_EVALUATION_DS=[
        os.path.join('sentence_similarity', 'Normalized', 'BIOSSES', 'corpora.tsv')
    ]

    evals = []
    for wpair in WPAIR_EVALUATION_DS:
        human_sims = []
        cos_sims=[]
        euclid_dists=[]
        wmd_dists=[]
        svd_sims=[]
        with open(wpair) as f:
            header=True
            for line in f:
                if header:
                    header=False
                    continue

                vals = line.rstrip('\n').split('\t')
                sent1 = vals[0]
                sent2 = vals[1]
                human_pred = vals[-1]

                # apply model with three algorithms
                # average + cosine sim/euclidean distance
                sent1_doc=_nlp(sent1)
                sent2_doc=_nlp(sent2)
                sent1_toks = clean_and_normalize_word(sent1_doc, remove_stopwords=True)
                sent2_toks = clean_and_normalize_word(sent2_doc, remove_stopwords=True)
                # apply bigram model
                sent1_toks_big = phrases[sent1_toks]
                sent2_toks_big = phrases[sent2_toks]
                if openai_key:
                    # get vectors
                    response = openai.Embedding.create(
                            input=' '.join(sent1_toks_big),
                            model="text-embedding-ada-002"
                    )
                    sent1_avg_embd = np.array(response['data'][0]['embedding'])
                else:
                    # compute average of embeddings
                    sl=[]
                    for t in sent1_toks_big:
                        if t in model.key_to_index:
                            try:
                                coeff = dictionary.dfs[dictionary.token2id[t]]/dictionary.num_docs
                                # https://openreview.net/pdf?id=SyK00v5xx
                                coeff_fn = A/(A+coeff)
                            except KeyError:
                                # missing
                                coeff_fn = 1.0
                            sl.append(coeff_fn*model[t])
                    sent1_embds = np.array(sl)
                    sent1_avg_embd = sent1_embds.mean(axis=0)
                    #print(sent1_toks_big)
                    #print(sent1_embds.shape)

                if openai_key:
                    # get vectors
                    response = openai.Embedding.create(
                            input=' '.join(sent2_toks_big),
                            model="text-embedding-ada-002"
                    )
                    sent2_avg_embd = np.array(response['data'][0]['embedding'])
                else:
                    sl=[]
                    for t in sent2_toks_big:
                        if t in model.key_to_index:
                            try:
                                coeff = dictionary.dfs[dictionary.token2id[t]]/dictionary.num_docs
                                # https://openreview.net/pdf?id=SyK00v5xx
                                coeff_fn = A/(A+coeff)
                            except KeyError:
                                # missing
                                coeff_fn = 1.0
                            sl.append(coeff_fn*model[t])
                    sent2_embds = np.array(sl)
                    sent2_avg_embd = sent2_embds.mean(axis=0)
                    #print(sent2_toks_big)
                    #print(sent2_embds.shape)

                # compute cosine similarity of two vectors
                cos_sims.append( np.dot(sent1_avg_embd, sent2_avg_embd.T) / (np.linalg.norm(sent1_avg_embd)*np.linalg.norm(sent2_avg_embd)) )
                # euclidean distance similarity
                euclid_dists.append( np.linalg.norm(sent1_avg_embd - sent2_avg_embd) )
                # WMD distance
                if  openai_key:
                    wmd_dists.append( 0.0 )
                else:
                    wmd_dists.append( model.wmdistance(sent1_toks_big, sent2_toks_big) )
                # SVD cosine similarity
                #sent1_svd_embd = svd_embedding(sent1_embds, max_cum_ve=MAX_CUM_VE)
                #sent2_svd_embd = svd_embedding(sent2_embds, max_cum_ve=MAX_CUM_VE)
                #print(sent1_svd_embd)
                #print(sent2_svd_embd)
                #print(sent1_svd_embd.shape)
                #print(sent2_svd_embd.shape)
                
                #svd_sims.append( np.dot(sent1_svd_embd, sent2_svd_embd.T) / (np.linalg.norm(sent1_svd_embd)*np.linalg.norm(sent2_svd_embd)) )
                # human prediction
                human_sims.append( float(human_pred) )
                
        # end of file processing
        # compute cosine similarity with true values
        pearson_cos = stats.pearsonr(human_sims, cos_sims)
        pearson_euclid = stats.pearsonr(human_sims, euclid_dists)
        pearson_wmd = stats.pearsonr(human_sims, wmd_dists)
        #pearson_svd = stats.pearsonr(human_sims, svd_sims)

        evals.append(
            '\t'.join([model_name,wpair,
                       str(pearson_cos[0]), str(pearson_cos[1]),
                       str(pearson_euclid[0]), str(pearson_euclid[1]),
                       str(pearson_wmd[0]), str(pearson_wmd[1])])
                       #str(pearson_svd[0]), str(pearson_svd[1])])
        )

    return evals

#analogies_result = model.wv.evaluate_word_analogies(datapath('questions-words.txt'))

#analogy = model.most_similar_cosmul(positive=['heart', 'failure'], negative=['liver'])
#print(analogy)

if __name__ == "__main__":
    if len(sys.argv) == 0:
        print("USAGE: python test_sentence.py cspace.kv.bin cspace.bigrams.pkl cspace.dict.pkl")
        sys.exit(1)
        
    path = sys.argv[1]
    if path == 'head':
        # print header
        print('\t'.join(['Model Name', 'corpus',
                        'cosine similarity correlation', 'cosine similarity p-value',
                        'euclidean distance correlation', 'euclidean distance p-value',
                         'word mover distance correlation', 'word mover distance p-value'])
        )
        sys.exit(0)

    openai_key=None
    try:
        openai_key=os.environ['OPENAI_API_KEY']
    except KeyError:
        pass

    bigram_path=None
    if len(sys.argv) > 1:
        bigram_path = sys.argv[2]
    else:
        sys.exit(1)

    if len(sys.argv) > 2:
        dictionary = Dictionary.load(sys.argv[3])
    else:
        sys.exit(1)

    fasttext=path.endswith('.fasttext.bin')
    # this doesn't allow to account of out-of-vocabulary words of FastText.. use full model
    if openai_key is not None:
        model=None
        name='text-embedding-ada-002'
    elif fasttext:
        model = load_facebook_model(path)
        name = os.path.split(path)[1]
    else:
        #model = KeyedVectors.load_word2vec_format(path, binary=True, unicode_errors='ignore')
        name = os.path.split(path)[1]
        model = KeyedVectors.load(path, mmap=None)
        #model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(path)

    # restrict vocabulary
    # restrict_vocab=300000 # as gensim evaluate_word_pairs
    # ok_keys = model.index_to_key[:restrict_vocab]
    # ok_vocab = {k: model.get_index(k) for k in reversed(ok_keys)}
    # original_key_to_index = model.key_to_index
    # model.key_to_index = ok_vocab

    phrases = Phrases.load(bigram_path)
    #frozen_phrases = Phraser(phrases)

    evals = evaluate(model, name, phrases, dictionary, fasttext, openai_key)
    print('\n'.join(evals))
