<p align="center">
  <img src="https://media.githubusercontent.com/media/cosbi-research/cspace/refs/heads/main/logo.png" alt="CSpace logo"/>
</p>

CSpace is a concise word embedding of bio-medical concepts that outperforms all alternatives in terms of out-of-vocabulary ratio (OOV) and semantic textual similarity (STS) task and have comparable performance with respect to transformer-based alternatives in the sentence similarity task.

CSpace also encodes ontological IDs (MeSH, NCBI gene and tax ID) and can be used for measuring the relatedness of diseases, genes or conditions, potentially unlocking previously unknown disease-condition association, as well as for semantic synonyms search.

All our fine-tuned embeddings can be obtained from Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14781672.svg)](https://doi.org/10.5281/zenodo.14781672)

# Table of Contents
- [Run the performance tests](#run-the-performance-tests)
- [Build training dataset](#build-training-dataset)
- [Training code and hyperparameters](#training-code-and-hyperparameters)

# Quick start with CSpace

go to the `examples` folder, 
activate the python virtualenv and run

```
pip install -r requirements.txt
```

then run in the interactive prompt the script `example.py`

for a quick tour of CSpace capabilities.

# Run the performance tests

Go to the `tests` folder,
activate the python virtualenv and run

```
pip install -r requirements.txt
```

then run the `test.py` script in the command line
to get the correlation between CSpace and human judgement on MayoSRS and UMNSRS word similarity test sets.

```
python test.py cspace.kv.bin
```

run the `test_sentence.py` script in the command line
to get the correlation between CSpace and human judgement on BIOSSES sentence similarity test set.

```
python test_sentence.py cspace.kv.bin cspace.bigrams.pkl cspace.dict.pkl
```

# Build training dataset

Go to the `preprocessing` folder for a complete description on how to re-build the training dataset from the raw data-sources.
In `preprocessing/preprocessed_data` you can find a sample of 81K already pre-processed publications.

# Training code and hyperparameters

The training code and hyperparameters can be found in the `training` folder.
As in the other folders, dependencies can be installed with 

```
pip install -r requirements.txt
```
