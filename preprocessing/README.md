# Step 1. Build unigram dataset from raw files

preprocess PubMed (medline), Bio/MedRxiv and PubmedCentral, composed of:
- manuscript collection (mscollection)
- oa (Open Access, commercial and non commercial) 

```
python preprocessor.py medline unigram example_data/medline.list
python preprocessor.py mscollection unigram example_data/mscollection.list
python preprocessor.py oa unigram example_data/oa.list
python preprocessor.py rxiv unigram example_data/rxiv.list
```

# Training 1. Train co-occurrence model on shuffled version of unigram datasets

Go to `training` folder and type

```
cat unigram_medline.dump unigram_mscollection.dump unigram_oa.dump unigram_rxiv.dump | shuf > unigram_complete.dump 
python train_bigrams.py unigram_complete.dump cspace
```

for building `cspace.bigrams.pkl` co-occurrence model on unigram dataset.

```
python train_dict.py unigram_complete.dump cspace.bigrams.pkl cspace 
```

for building `cspace.dict.pkl` the word frequency model on the bigram dataset.


# Step 2. Build ngram dataset from pubtator3 and word co-occurrence model

build n-gram dataset for PubMed and PubmedCentral datasets only.

```
python preprocessor.py medline ngram example_data/medline.list cspace.bigrams.pkl
python preprocessor.py mscollection ngram example_data/mscollection.list cspace.bigrams.pkl
python preprocessor.py oa ngram example_data/oa.list cspace.bigrams.pkl
```

# Training 2. Train CSpace
Go to `training` folder and type

```
cat ngram_medline.dump ngram_mscollection.dump ngram_oa.dump unigram_complete.dump | shuf > ngram_complete.dump
LINES=$(wc -l ngram_complete.dump | cut -f1 -d' ')
WORDS=$(wc -w ngram_complete.dump | cut -f1 -d' ')
python train_cspace.py ngram_complete.dump $LINES $WORDS cspace_hyperparameters.yaml cspace 
```
