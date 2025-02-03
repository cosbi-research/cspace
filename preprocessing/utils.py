import re2, re, sys
from abbreviations import schwartz_hearst
import lxml.html
from lxml import etree
from lxml.etree import iterparse

__re_collapse_spaces__ = re2.compile("\s+")
__re_remove_special_chars__ = re2.compile("[;:\?\!\.\'\"\*/\),\(\|\s]+")
__re_dot_parens__ = re2.compile("\.([A-Z])")
SYMBOLS = ['..','.',',','?',';','*','!','%','^','&','+','"','(',')',':','[',']','{','}','/','|','\'','–','=','≥','>','<','≤',' ']
__bad_tokens__ = re2.compile(r'''^(https?://.*|www\..*|[()',";?=<≥≤>\.#=:0-9/\-%«»*—–|^°]+|.*\.(jpg|png|gif|svg)|.|(?:(?:rr|or|hr|p|n)[=<>≥≤][0-9\.%]*))$''')
__post_remove_re__ = re2.compile('(\^|—)[0-9]+|[^0-9a-z -_—]|[;\}\)\(\{\[]+')
__post_dotspace_re__ = re2.compile('\.')

def strip_text(nlp, elem_str):
    try:
        # strip all the tags
        elem_str = lxml.html.fromstring( elem_str ).text_content()
    except (lxml.etree.XMLSyntaxError, lxml.etree.ParserError) as e:
        elem_str = ''

    doctxt = clean_doc(elem_str)
    nlp_doc = nlp(doctxt)
    retvals=[]
    for sentence in nlp_doc.sents:
        retval = clean_and_normalize_word(sentence, remove_stopwords=False)
        # Discard sentences with less that 5 tokens
        retval_len = len(retval)
        if retval_len <= 5:
            continue
        retvalstr = ' '.join(retval)
        retvals.append(retvalstr)
    return '\n'.join(retvals)

def tostring(elem):
    if elem is None:
        return ''
    try:
        s = etree.tostring(elem, encoding="unicode", with_tail=False)
    except UnicodeDecodeError:
        s = '????'
    return s

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
            normalized_tokens.append(collapse_spaces(norm_txt))

    return normalized_tokens


def analyzexml(nlp, curfname):
    try:
        path=[]
        # glossaries, competing interest
        unwanted_section_type = ['ABBR','COMP_INT']
        body_section_type = ['INTRO', 'METHODS', 'RESULTS', 'DISCUSS', 'SUPPL', 'CONCL', 'CASE', 'APPENDIX']

        alternatives=[]
        for event, elem in iterparse(curfname, events=("start","end"), recover=True):
            try:
                if event == 'start':
                    path.append(elem.tag)

                    if elem.tag == 'document':
                        curid=None
                        curtext=None
                        curoffset=None
                        cur_mesh_id = cur_mesh_type = None
                        curanns=[]
                        
                elif event == 'end':
                    if len(path)>2 and path[-2] == 'document' and elem.tag == 'id':
                        curid=elem.text
                    elif len(path)>2 and path[-2] == 'passage' and elem.tag =='offset':
                        curoffset = int(elem.text)
                    elif len(path)>2 and path[-2] == 'passage' and elem.tag =='text':
                        curtext = elem.text
                    ### ANNOTATION ###
                    elif len(path)>2 and 'passage' in path and path[-2]=='annotation' and elem.tag == 'infon' and 'key' in elem.attrib and elem.attrib['key'] == 'identifier':
                        cur_mesh_id = elem.text.lower().replace(':','_')
                    elif len(path)>2 and 'passage' in path and path[-2]=='annotation' and elem.tag == 'infon' and 'key' in elem.attrib and elem.attrib['key'] == 'type':
                        cur_mesh_type = elem.text.lower()
                    elif len(path)>2 and 'passage' in path and path[-2]=='annotation' and elem.tag == 'location':
                        curannotation_offset=int(elem.attrib['offset'])
                        curannotation_len=int(elem.attrib['length'])
                    elif len(path)>2 and 'passage' in path and path[-2]=='annotation' and elem.tag == 'text':
                        curannotation_text=elem.text
                    ## END ANNOTATION ##
                    elif len(path)>2 and path[-2] == 'passage' and elem.tag =='annotation':
                        # apply curannotation to curtext
                        start = curannotation_offset - curoffset 
                        end = start + curannotation_len
                        curanns.append( (start,end, curannotation_text, cur_mesh_type, cur_mesh_id) )
                        
                        # reset
                        cur_mesh_id = None
                        cur_mesh_type = None
                        curannotation_len = None
                        curannotation_offset = None
                        curannotation_text = None
                        
                    elif elem.tag == 'passage':
                        # end of passage, conclude cursection paragraph

                        if curtext is not None:
                            alternative_ngram = curtext
                            alternative_id = curtext
                            for start,end,curannotation_text,cur_mesh_type,cur_mesh_id in sorted(curanns, key=lambda e: e[0], reverse=True):
                                # text ngram version from pubtator3
                                alternative_ngram = alternative_ngram[:start] + (curannotation_text.replace(' ','_')) + alternative_ngram[end:]
                                # id version from putbtator3
                                if cur_mesh_type is not None and cur_mesh_id is not None:
                                    alternative_id = alternative_id[:start] + (cur_mesh_type+'_'+cur_mesh_id) + alternative_id[end:]

                            nlp_ngram = strip_text(nlp, alternative_ngram)
                            alternatives.append(nlp_ngram)
                            if len(alternative_id)>0:
                                nlp_id = strip_text(nlp, alternative_id)
                                if nlp_ngram != nlp_id:
                                    alternatives.append(nlp_id)

                        curtext=None
                        curoffset=None
                        curanns=[]
    
                    path.pop()
                
            except Exception as e:
                sys.stderr.write("Exception "+str(e)+" processing element: "+elem.tag+'\n')
                sys.stderr.write(tostring(elem)+'\n')
                raise e
            
        return alternatives
    
    except Exception as e:
        sys.stderr.write("Exception "+str(e)+" processing "+curfname+'\n')
        raise e
