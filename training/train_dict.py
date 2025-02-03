import sys
#from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.models.phrases import Phrases, Phraser, ENGLISH_CONNECTOR_WORDS

fname=sys.argv[1]
# load concept co-occurrence model
phrases = Phrases.load(sys.argv[2])
frozen_phrases = Phraser(phrases)

basename=sys.argv[3]

def process_sentence(s):
    return s.rstrip('\n').split()

def process_input(f):
    for line in f:
        l = line.rstrip('\n')
        sentence = l.split()
        bigrams = frozen_phrases.find_phrases([sentence])
        bigrams_words = []
        if len(bigrams) > 0:
            bigrams_words = list(bigrams.keys())
        yield ' '.join(sentence + bigrams_words)

with open(fname) as f:
    dct = Dictionary(map(process_sentence, process_input(f)), prune_at=1000000)

dct.save(basename+'.dict.pkl')
#dct = Dictionary.load(basename+'.dict.pkl')

#with open(fname) as f:
#    corpus = map(lambda line: dct.doc2bow(line.rstrip('\n').split()) , f)
#    model = TfidfModel(corpus)  # fit model
#    model.save(basename+'.sentence_tf_idf_nfc.pkl')
#vector = model[corpus[0]]  # apply model to the first sentence in corpus
