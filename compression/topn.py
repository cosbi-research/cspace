#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script compress the gensim KeyedVectors format
used by CSpace by discarding all concepts
except the top X by frequency (that are usually the most high quality).
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
    sorted_vocab = sorted(model.key_to_index.keys(), key=lambda x: model.get_vecattr(x, 'count'), reverse=True)
    top_vocab_list = deepcopy(sorted_vocab[:new_vocab_size])
    
    topmodel = model.vectors_for_all(top_vocab_list, allow_inference=False, copy_vecattrs=True)

    topmodel.save('cspace.'+str(int(new_vocab_size/1000.0))+'k.kv.bin')
    
