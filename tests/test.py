#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to generate data displayed in Table 4 and 6 of paper
"""

from gensim.test.utils import datapath
from gensim.models import KeyedVectors
from gensim.models import FastText,Word2Vec
import compress_fasttext
from test_utils import evaluate_word_pairs, evaluate_openai_word_pairs
import sys,os

def analogy(x1, x2, y1):
    result = model.most_similar(positive=[y1, x2], negative=[x1])
    return result[0][0]

def evaluate(model, fasttext, mesh, openai_key):
   
    if not mesh:
        WPAIR_EVALUATION_DS=[
            datapath('wordsim353.tsv'),
            os.path.join('concept_similarity', 'Normalized','SimLexScores_lemmatized.tsv'),
            os.path.join('concept_similarity', 'Normalized', 'MayoTerms_lemmatized.tsv'),
            os.path.join('concept_similarity', 'Normalized', 'UMNSRS_relatedness_Terms_lemmatized.csv'),
            os.path.join('concept_similarity', 'Normalized', 'UMNSRS_similarity_Terms_lemmatized.csv')
        ]
    else:
        WPAIR_EVALUATION_DS=[
            datapath('wordsim353.tsv'),
            os.path.join('concept_similarity', 'Normalized','SimLexScores_lemmatized.tsv'),
            os.path.join('concept_similarity', 'Normalized', 'MayoMeshID.tsv'),
            os.path.join('concept_similarity', 'Normalized', 'UMNSRS_relatedness_MeshID.csv'),
            os.path.join('concept_similarity', 'Normalized', 'UMNSRS_similarity_MeshID.csv')
        ]

    evals=[]
    for wpair in WPAIR_EVALUATION_DS:
        try:
            if openai_key is not None:
                similarities = evaluate_openai_word_pairs(openai_key, wpair)
            elif fasttext:
                # evaluate word pairs, taking into account FastText ability to synthetize new vectors for OOV words
                similarities = evaluate_word_pairs(model, wpair)#, restrict_vocab=50000000)
            else:
                # standard evaluate word pairs
                similarities = model.evaluate_word_pairs(wpair)#, restrict_vocab=50000000)
            evals.append(os.path.basename(wpair[:-4])+"\t"+str(similarities[0][0])+'\t'+str(similarities[0][1])+'\t'+str(similarities[1].correlation)+'\t'+str(similarities[1].pvalue)+'\t'+str(similarities[2]))
        except ValueError:
            # error
            evals.append(os.path.basename(wpair[:-4])+"\t\t\t\t\t")
        #print(similarities)
    return evals

#analogies_result = model.wv.evaluate_word_analogies(datapath('questions-words.txt'))

#analogy = model.most_similar_cosmul(positive=['heart', 'failure'], negative=['liver'])
#print(analogy)

if __name__ == "__main__":
    path = sys.argv[1]
    mesh = len(sys.argv) > 2 and sys.argv[2].lower() == 'mesh'
    fasttext=path.endswith('.pkl')
    openai_key=None
    try:
        openai_key=os.environ['OPENAI_API_KEY']
    except KeyError:
        pass
    # this doesn't allow to account of out-of-vocabulary words of FastText.. use full model
    if openai_key is not None:
        model=None
    elif fasttext:
        modelf = FastText.load(path, mmap=None)
        model = modelf.wv
    else:
        #model = KeyedVectors.load_word2vec_format(path, binary=True, unicode_errors='ignore')
        #model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(path)
        model = KeyedVectors.load(path, mmap=None)

    # hack
    #if model is not None:
    #    prev_index_to_key = model.index_to_key
    #    model.index_to_key = list(map(lambda x: x if x is not None else '', prev_index_to_key))

    evals = evaluate(model, fasttext, mesh, openai_key)
    print('\n'.join(evals))
