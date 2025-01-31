#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
script to convert CUI ID to MESH ID
"""

import sys
import re,requests,json

BASE_URI="https://uts-ws.nlm.nih.gov/rest"
API_KEY="<NCBI_API_KEY>"
MESH_URI="https://id.nlm.nih.gov/mesh"
MESH_LABEL_URI="https://id.nlm.nih.gov/mesh/lookup/descriptor"
ID_URI="https://id.nlm.nih.gov/mesh/servlet/explore/relatedFromSubjects?uri="

def get(url):
    ok = False
    while not ok:
        try:
            response = requests.get(url)
            jrels = response.json()
            ok = True
        except requests.exceptions.ConnectionError:
            time.sleep(1)
        except json.decoder.JSONDecodeError:
            time.sleep(1)
            
    return jrels
            
def cui2mesh(cui, category):

    # filter by MSH
    url=BASE_URI+"/content/current/CUI/"+cui+"/relations?sabs=MSH&apiKey="+API_KEY
    jrels = get(url)
    if 'result' not in jrels:
        print("No MSH map found.. try to search in another ontology")
        url=BASE_URI+"/content/current/CUI/"+cui+"/relations?apiKey="+API_KEY
        jrels = get(url)
        mesh = None
        jsnorelres = jrels['result']
        for jsnorel in jsnorelres:
            if 'relatedFromIdName' not in jsnorel:
                continue
            label = jsnorel['relatedFromIdName']
            # try with full name
            url=MESH_LABEL_URI+"?match=startswith&label="+label+"&limit=1"
            jrels=get(url)
            if len(jrels) == 0:
                for word in reversed(label.split()):
                    url=MESH_LABEL_URI+"?match=startswith&label="+word+"&limit=1"
                    jrels=get(url)
                    if len(jrels) > 0:
                        mesh = jrels[0]['resource'].split('/')[-1]
                        break
            else:
                mesh = jrels[0]['resource'].split('/')[-1]
            if mesh is not None:
                break

        if mesh is None:
            # try with relatedIdName
            for jsnorel in jsnorelres:
                if 'relatedIdName' not in jsnorel:
                    continue
                label = jsnorel['relatedIdName']
                url=MESH_LABEL_URI+"?match=startswith&label="+label+"&limit=1"
                jrels=get(url)
                if len(jrels) == 0:
                    for word in label.split():
                        url=MESH_LABEL_URI+"?match=startswith&label="+word+"&limit=1"
                        jrels=get(url)
                        if len(jrels) > 0:
                            mesh = jrels[0]['resource'].split('/')[-1]
                            break
                else:
                    mesh = jrels[0]['resource'].split('/')[-1]
                    
                if mesh is not None:
                    break
            
    else:
        jrelres = jrels['result']
        related_uri=jrelres[0]['relatedFromId']
        mesh = related_uri.split('/')[-1]
        if mesh.upper().startswith('A'):
            # follow link
            url=related_uri+'?apiKey='+API_KEY
            jres = get(url)
            code_uri = jrelres = jres['result']['code']
            mesh = code_uri.split('/')[-1]
    
    return category.lower()+'_mesh_'+mesh.lower()


if __name__ == "__main__":
    path = sys.argv[1]
    outpath = sys.argv[2]

    with open(path) as f, open(outpath, 'w') as fw:
        for line in f:
            row=line.strip()
            if row.startswith('#'):
                fw.write(row+'\n')
            else:
                cols = row.split('\t')
                print("Processing "+cols[0]+"...")
                mesh1 = cui2mesh(cols[0], 'disease')
                print("Processing "+cols[1]+"...")
                mesh2 = cui2mesh(cols[1], 'disease')

                fw.write('\t'.join([mesh1, mesh2]+cols[2:])+'\n')
    
