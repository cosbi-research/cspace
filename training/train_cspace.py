#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gensim
import numpy
import yaml
import os, sys
# allows to get representation for unseen words
# https://radimrehurek.com/gensim/models/fasttext.html
from gensim.models import FastText
from gensim.models import KeyedVectors
from gensim.models.fasttext import save_facebook_model

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.stderr.write("./train_cspace.py <pre-processed sentences file> <number of sentences> <number of words> <hyperparameters yaml file> cspace\n")
        sys.exit(1)
        
    fpath = sys.argv[1].lower()
    fpath_len = int(sys.argv[2])
    fpath_word_n = int(sys.argv[3])
    params_path = sys.argv[4]
    name= sys.argv[5]
    
    with open(params_path) as f:
        params = yaml.safe_load(f)

    w2v_params = params['w2v']
    # overwrite with finetuning params
    w2v_params.update( params['finetune'] )

    model = FastText(**w2v_params)
    print("=== Building vocabulary ===")
    model.build_vocab(corpus_file=fpath, update=False)
    
    print("=== Training ===")
    model.train(corpus_file=fpath, total_examples=fpath_len, total_words=fpath_word_n, epochs=w2v_params['epochs'], compute_loss=False)

    # save in FastText format
    model.save(name+'.pkl')

    ## save in Facebook format
    # save_facebook_model(model, name.fasttext.bin')

    ## save in FastTextKeyedVectors format
    # model.wv.save(name+'.'+w2vtype+'kv.bin')

    ## save in KeyedVectors format ( cannot synthesize embeddings from out-of-vocabulary words)
    kvmodel = model.wv.vectors_for_all(model.wv.key_to_index)
    kvmodel.save(name+'.kv.bin')

    ## save in word2vec format
    # model.wv.save_word2vec_format(name+'.'+w2vtype+'.bin', binary=True)
