#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script converts the gensim KeyedVectors format
used by CSpace to the following two tsv files:
- vectors.tsv
- metadata.tsv

to be used in the tensorflow projector
https://projector.tensorflow.org/

to visualize embeddings
"""

from gensim.models import KeyedVectors
from copy import deepcopy
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("USAGE: cspace2tfprojector.py <model_path> [top-n vectors]")
        print("Please provide CSpace in KeyedVector format (.bin extension)")
        sys.exit(1)
        
    path = sys.argv[1]
    new_vocab_size = int(sys.argv[2]) if len(sys.argv) > 2 else None
    model = KeyedVectors.load(path, mmap='r')
    if new_vocab_size is not None:
        sorted_vocab = sorted(model.key_to_index.items(), key=lambda x: model.get_vecattr(x[0], 'count'), reverse=True)
        top_vocab_list = deepcopy(sorted_vocab[:new_vocab_size])
    else:
        top_vocab_list = model.key_to_index.items()
    
    with open('metadata.'+str(new_vocab_size)+'.tsv','w') as mf, open('vectors.'+str(new_vocab_size)+'.tsv','w') as vf:
        for concept, idx in top_vocab_list:
            # ith concept
            mf.write(concept+'\n')
            # ith vector
            vf.write('\t'.join([str(x) for x in model[idx]])+'\n')
        

    
