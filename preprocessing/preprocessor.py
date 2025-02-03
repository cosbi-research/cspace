#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from lxml.etree import iterparse
from lxml import etree
import xml.etree.ElementTree as ET
import pprint,json,traceback,os,tempfile
from datetime import datetime
import re,sys,requests
import tempfile
from utils import strip_text, clean_doc, clean_sentence, normalize_words, clean_and_normalize_word,tostring, analyzexml
from language import set_environment

from gensim.models.phrases import Phrases, Phraser

def medline(nlp, fname, unigram=True, bimodel=None):
    DUMP = 'unigram_medline.dump' if unigram else 'ngram_medline.dump'

    path=[]
    for event, elem in iterparse(fname, events=("start","end")):
        tagName = elem.tag.lower()
        if event == 'start':
            path.append(tagName)

            if tagName == 'pubmedarticle':
                doc_id=None
                title = None
                abstracts=[]
                
        elif event == 'end':
            if tagName == 'pmid':
                doc_id = elem.text
                
            elif 'article' in path:
                if tagName == 'articletitle':
                    title = strip_text(nlp, tostring(elem))                        
                elif 'abstract' in path:
                    if tagName == 'abstracttext':
                        abstracts.append( strip_text(nlp, tostring(elem)) )
            
            elif tagName == 'pubmedarticle':
                # end serialize
                if unigram:
                    with open(DUMP, 'a') as fw:
                        fw.write(title+'\n'+'\n'.join(abstracts)+'\n')
                else:
                    # version 2: co-occurrence model
                    bititle = ' '.join(bimodel[title.split(' ')])
                    with open(DUMP, 'a') as fw:
                        fw.write(bititle+'\n')
                        # apply co-occurrence model (version 2)
                        for ab in abstracts:
                            biab = ' '.join(bimodel[ab.split(' ')])
                            fw.write(biab+'\n')
                    # version 3: pubtator3
                    # version 3.1: pubtator3 ngrams
                    # version 3.2: pubtator3 ontological IDs
                    response = requests.get('https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocxml?pmids='+doc_id)
                    response.raise_for_status()
                    data = response.text
                    _, tp3_name = tempfile.mkstemp('pt3_')
                    with open(tp3_name,'w') as f:
                        f.write(data)
                    variants = analyzexml(nlp, tp3_name)
                    os.remove(tp3_name)

                    with open(DUMP, 'a') as fw:
                        for v in variants:
                            fw.write(v+'\n')
                    
                doc_id=None
                title = None
                abstracts = []
                    
            path.pop()
    
def pubmedcentral(sourcetype, nlp, fname, unigram=True, bimodel=None):
    DUMP = 'unigram_'+sourcetype+'.dump' if unigram else 'ngram_'+sourcetype+'.dump'
    path=[]
    for event, elem in iterparse(fname, events=("start","end")):
        if event == 'start':
            path.append(elem.tag)
                         
            if elem.tag == 'article' or elem.tag =='sub-article':
                pmids=None
                doc_id=None
                title = None
                abstracts=[]
                tables=[]
                body=[]
                
        elif event == 'end':
            if 'response' not in path and ('front' in path or 'front-stub' in path):
                if elem.tag == 'article-id':
                    if 'pub-id-type' in elem.attrib and elem.attrib['pub-id-type'] == 'pmid':
                        if elem.text is None:
                            raise Exception("Failed to get PMID for '"+fname+"'")
                        doc_id = elem.text
                        
                elif (   'title-group' in path and elem.tag == 'article-title'
                       or 'trans-title-group' in path and elem.tag == 'trans-title'):
                    title = strip_text(nlp, tostring(elem))
                        
                elif elem.tag == 'abstract':
                    abstracts.append( strip_text(nlp, tostring(elem)) )
                    
            elif 'table-wrap' in path and 'table-wrap-foot' not in path:
                if elem.tag == 'label':
                    tables.append( strip_text(nlp, elem.text if elem.text is not None else '') )
                elif elem.tag == 'caption':
                    tables.append( strip_text(nlp, tostring(elem)) )
            elif elem.tag == 'table-wrap-foot':
                tables.append( strip_text(nlp, tostring(elem.find('.//attrib')) ) )
            # BODY
            elif elem.tag == 'body':
                body.append( strip_text(nlp, tostring(elem)) )                
            elif elem.tag == 'article' or elem.tag =='sub-article':
                # end serialize
                if unigram:
                    ls=[title]+abstracts+body+tables
                    with open(DUMP, 'a') as fw:
                        fw.write('\n'.join(ls)+'\n')
                else:
                    # version 2: co-occurrence model
                    bititle = ' '.join(bimodel[title.split(' ')])
                    with open(DUMP, 'a') as fw:
                        fw.write(bititle+'\n')
                        for ab in abstracts:
                            biab = ' '.join(bimodel[ab.split(' ')])
                            fw.write(biab+'\n')
                        for bd in body:
                            bibd = ' '.join(bimodel[bd.split(' ')])
                            fw.write(bibd+'\n')
                        for tb in tables:
                            bitb = ' '.join(bimodel[tb.split(' ')])
                            fw.write(bitb+'\n')
                    # version 3: pubtator3
                    # version 3.1: pubtator3 ngrams
                    # version 3.2: pubtator3 ontological IDs
                    response = requests.get('https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocxml?pmids='+doc_id)
                    response.raise_for_status()
                    data = response.text
                    _, tp3_name = tempfile.mkstemp('pt3_')
                    with open(tp3_name,'w') as f:
                        f.write(data)
                    variants = analyzexml(nlp, tp3_name)
                    os.remove(tp3_name)

                    with open(DUMP, 'a') as fw:
                        for v in variants:
                            fw.write(v+'\n')

                    
                doc_id=None
                title = None
                abstracts = []
                tables=[]
                body=[]
                    
            path.pop()
    
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("preprocessor.py <medline|mscollection|oa|rxiv> <unigram|ngram> <medline_xml.list> [co-occurrence_model]")
        sys.exit(1)

    # source type
    stype = sys.argv[1]
    # dump type
    dtype = sys.argv[2]
    slist_f = sys.argv[3]
    nlp = set_environment()

    slist = []
    with open(slist_f) as f:
        for line in f:
            slist.append(line.rstrip())

    unigram=False
    if dtype == 'unigram':
        unigram=True

    frozen_phrases=None
    if not unigram:
        # load co-occurrence model
        phrases = Phrases.load(sys.argv[4])
        frozen_phrases = Phraser(phrases)

    if stype == 'medline':
        for source in slist:
            medline(nlp, source, unigram, bimodel=frozen_phrases)
    if stype in ['mscollection','oa','rxiv']:
        for source in slist:
            pubmedcentral(stype, nlp, source, unigram, bimodel=frozen_phrases)
        
