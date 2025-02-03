import spacy,sys
from spacy.tokenizer import Tokenizer
from spacy.lang.en.stop_words import STOP_WORDS

ADDITIONAL_STOPWORDS = {"in"}


def set_environment():
    """Set the environment defining the language model and components."""

    try:
        nlp = spacy.load('en_core_web_lg')   # load the language model
    except OSError:
        print("Please run:")
        print("python -m spacy download en_core_web_lg")
        print("To install the english language model 'en_core_web_lg'")
        sys.exit(1)
    load_stopwords(nlp)                             # laad the list of stopwords
    custom_lemmatization(nlp)                       # set fixed lemmas
    nlp.remove_pipe('ner')                          # remove the NER module
    nlp.tokenizer = custom_tokenizer(nlp)           # set custom tokenizer

    return nlp


def load_stopwords(nlp):
    """Load the default stopwords and add new custom ones."""

    enriched_stopwords = STOP_WORDS.union(ADDITIONAL_STOPWORDS)

    for word in enriched_stopwords:
        for w in (word, word[0].capitalize(), word.upper()):
            lex = nlp.vocab[w]
            lex.is_stop = True


def custom_tokenizer(nlp):
    """Set the tokenizer and add new custom rules."""

    prefix_re = spacy.util.compile_prefix_regex(nlp.Defaults.prefixes)
    suffix_re = spacy.util.compile_suffix_regex(nlp.Defaults.suffixes)
    
    # Maintaining hyphens
    custom_infixes = ['\.\.\.+', '[\/!&:,()\{\}@|]']
    infix_re = spacy.util.compile_infix_regex(tuple(custom_infixes))

    # Splitting on hyphens
    #custom_infixes = ['\.\.\.+', '(?<=[0-9])-(?=[0-9])', '[\/\-!&:,()\{\}|]']
    #infix_re = spacy.util.compile_infix_regex(tuple(list(nlp.Defaults.infixes) + custom_infixes))
    
    return Tokenizer(nlp.vocab, nlp.Defaults.tokenizer_exceptions,
        prefix_search=prefix_re.search,
        infix_finditer=infix_re.finditer,
        suffix_search=suffix_re.search,
        token_match=None)


def custom_lemmatization(nlp):
    """Set new lemmas for mismatched words."""

    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'rated'] = ('rate',)
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'rating'] = ('rate',)
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'rates'] = ('rate',)
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'upregulated'] = ('upregulate',)
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'upregulating'] = ('upregulate',)
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'upregulates'] = ('upregulate',)
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'downregulated'] = ('downregulate',)
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'downregulating'] = ('downregulate',)
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'downregulates'] = ('downregulate',)
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'dysregulated'] = ('dysregulate',)
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'dysregulating'] = ('dysregulate',)
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'dysregulates'] = ('dysregulate',)
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'cocultured'] = ('coculture',)
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'coculturing'] = ('coculture',)
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'cocultures'] = ('coculture',)
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'colocalized'] = ('colocalize',)
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'colocalizing'] = ('colocalize',)
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'colocalizes'] = ('colocalize',)
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'nebulized'] = ('nebulize',)
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'nebulizing'] = ('nebulize',)
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'nebulizes'] = ('nebulize',)
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'plated'] = ('plate',)
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'plating'] = ('plate',)
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'plates'] = ('plate',)
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'spared'] = ('spare',)
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'sparing'] = ('spare',)
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'spares'] = ('spare',)
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'synthetized'] = ('synthetize',)
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'synthetizing'] = ('synthetize',)
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'synthetizes'] = ('synthetize',)
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'synergized'] = ('synergize',)
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'synergizing'] = ('synergize',)
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'synergizes'] = ('synergize',)
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'biotinylated'] = ('biotinylate',)
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'biotinylating'] = ('biotinylate',)
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'biotinylates'] = ('biotinylate',)
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'phosphorylated'] = ('phosphorylate',)
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'phosphorylating'] = ('phosphorylate',)
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'phosphorylates'] = ('phosphorylate',)
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'cause'] = ('cause',) # not a slang!
    nlp.vocab.morphology.lemmatizer.lookups.get_table("lemma_exc")[u'verb'][u'wound'] = ('wound',) # not a slang!
