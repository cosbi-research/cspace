import re
from abbreviations import schwartz_hearst

__re_collapse_spaces__ = re.compile("\s+")
__re_remove_special_chars__ = re.compile("[;:\?\!\.\'\"\*/\),\(\|\s]+")
__re_dot_parens__ = re.compile("\.([A-Z])")
SYMBOLS = ['..','.',',','?',';','*','!','%','^','&','+','"','(',')',':','[',']','{','}','/','|','\'','–','=','≥','>','<','≤',' ']
__bad_tokens__ = re.compile(r'''^(https?://.*|www\..*|[()',";?=<≥≤>\.#=:0-9/\-%«»*—–|^°]+|.*\.(jpg|png|gif|svg)|.|(?:(?:rr|or|hr|p|n)[=<>≥≤][0-9\.%]*))$''')
__post_remove_re__ = re.compile('(\^|—)[0-9]+|[^0-9a-z -_—]|[;\}\)\(\{\[]+')
__post_dotspace_re__ = re.compile('\.')

def collapse_spaces(s):
    return __re_collapse_spaces__.sub(" ", s).strip()

def clean_doc(s):
    #replace double parentheses with single. Double was causing problems with Shwartz-Hearst
    s = s.replace("////", " ").replace("((", "(").replace("))", ")")
    s = __re_dot_parens__.sub(". \\1", s)
    # pair abbreviation/definition
    pairs = schwartz_hearst.extract_abbreviation_definition_pairs(doc_text = s)
    if len(pairs) > 0:
        for key, value in pairs.items():
            acro_paren = re.escape("(" + key + r")") #first mention of acronym (in parentheses)
            s = re.sub(acro_paren, "", s) #remove first mention of acronym (in parentheses)
            s = s.replace(key, value) #replace remaining acronym with full version

    return s

def clean_sentence(sentence, remove_stopwords=False):
    """Remove symbol tokens and stopwords from the token list."""
    cleaned_list = []

    for token in sentence:
        if not __bad_tokens__.match(token.text) and (not remove_stopwords or not token.is_stop):
            cleaned_list.append(token)

    return cleaned_list


def normalize_words(sentence):
    """Normalize and singularize words into lemmas (e.g., is|was|were --> be)."""
    normalized_tokens = []

    for token in sentence:
        normalized_tokens.append(token.lemma_.lower() if token.lemma_ is not None else token.text.lower())

    return normalized_tokens


def clean_and_normalize_word(sentence, remove_stopwords=False):
    normalized_tokens = []

    for token in sentence:
        if not __bad_tokens__.match(token.text) and (not remove_stopwords or not token.is_stop):
            txt = token.lemma_.lower() if token.lemma_ is not None else token.text.lower()
            norm_txt = __post_dotspace_re__.sub('', __post_remove_re__.sub('', txt))
            normalized_tokens.append(norm_txt)

    return normalized_tokens

def word_tokenize(sentence):
    return sentence.split()

def full_section_names(metadata):
    metadata_regex_alternatives = {}
    for k in metadata:
        if isinstance(metadata[k], list):
            metadata_regex_alternatives[k] = '|'.join(metadata[k])
        else:
            metadata_regex_alternatives[k]  = metadata[k]
    return  metadata_regex_alternatives

def create_regexes_map(metadata, annotation_types, metadata_regex_alternatives):
    """"""
    regexes_map = {}

    # Avoid retrieving mesh metadata, take care of automatic annotations here
    automatic_annotation_types = []
    for annotation_type in annotation_types:
        if annotation_type not in ["mesh"]:
            automatic_annotation_types.append(annotation_type)
    
    for field_key, human_name in constants.FIELD_ANNOT_MAP.items():
        field_format = metadata[field_key][0]
        regexes_map[human_name] = enumerate_regexes_for_dict_fields(
            field_format, metadata,
            automatic_annotation_types,
            metadata_regex_alternatives
        )
    return regexes_map

