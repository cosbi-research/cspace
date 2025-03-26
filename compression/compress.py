#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script compress the fasttext version of CSpace with
- product quantization: for all vectors, divide 152 dimensions in 76 vectors of 2 dimensions and cluster to 255 clusters.
                        this raises a byte-encoding for every cluster to be used in place of the original dimension
                        compressing 2 floats of 4 bytes to 1 byte (compression rate: 1/8)
- feature selection: top-100.000 word vectors by frequency (more frequent concepts tend to have better embeddings)
"""

import compress_fasttext
import sys
from gensim.models import FastText

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("USAGE: compress.py <model_path>")
        print("Please provide CSpace in FastText format as exported from training phase (see training/train_cspace.py)")
        sys.exit(1)
        
    path = sys.argv[1]

    model = FastText.load(path, mmap='r')
    vectors = model.wv
    small_model = compress_fasttext.prune_ft_freq(vectors, pq=True, qdim=76, new_vocab_size=100000, fp16=False)
    small_model.save('cspace.compressed.100k.bin')
