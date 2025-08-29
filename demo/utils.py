import requests,json,time
from collections import deque 

def navigate_mesh_ontology(oid):
    middle_ids=[]
    middle_names=[]
    top_ids=[]
    top_names=[]
    
    ok = False
    while not ok:
        try:
            response = requests.get('https://id.nlm.nih.gov/mesh/'+oid+'.json')
            orig_resp = jresp = response.json()
            ok = True
        except json.decoder.JSONDecodeError as e:
            raise Exception(response.text, e)
        except requests.exceptions.ConnectionError:
            time.sleep(1)

    other_branches = deque()
    while jresp is not None:
        if 'broaderDescriptor' in jresp or 'preferredMappedTo' in jresp:
            if 'broaderDescriptor' in jresp:
                bdescs = jresp['broaderDescriptor']
            elif 'preferredMappedTo' in jresp:
                bdescs = jresp['preferredMappedTo']

            if type(bdescs) is str:
                bdescs = [bdescs]

            for bdesc in bdescs:
                bid = bdesc.split('/')[-1]
                #if bid not in unwanted_descriptors:
                ok = False
                while not ok:
                    try:
                        bresponse = requests.get('https://id.nlm.nih.gov/mesh/'+bid+'.json')
                        jbresp = bresponse.json()
                        ok = True
                    except requests.exceptions.ConnectionError:
                        time.sleep(1)

                bname_candidates = jbresp['label']
                if type(bname_candidates) is dict:
                    bname_candidates = [bname_candidates]

                bname = None
                for bname_candidate in bname_candidates:
                    if bname_candidate['@language'] == 'en':
                        bname = bname_candidate['@value']
                        break

                if bname is None:
                    bname = bname_candidates[0]['@value']

                if bid in middle_ids or bid in top_ids:
                    # avoid loops
                    continue
                
                if 'broaderDescriptor' in jbresp:
                    middle_ids.append( bid )
                    middle_names.append( bname )
                    other_branches.append(jbresp)
                else:
                    top_ids.append( bid )
                    top_names.append( bname )

            if len(other_branches) > 0:
                jresp = other_branches.pop()
            else:
                jresp = None
        else:
            jresp = None
    
    return (middle_ids,middle_names,top_ids,top_names)


def getmesh(bioentity_id):
    meshid=bioentity_id.upper()
    
    ok = False
    trial=0
    jresp = None
    while not ok and trial<MAX_RETRY:
        try:
            response = requests.get('https://id.nlm.nih.gov/mesh/'+meshid+'.json')
            orig_resp = jresp = response.json()
            ok = True
        except requests.exceptions.ConnectionError:
            time.sleep(1)
        except json.decoder.JSONDecodeError:
            #sys.stderr.write('Unparseable response for '+meshid+'\n')
            return bioentity_id
        
        trial+=1

    res = {}
    if jresp is not None and 'label' in jresp:
        # get human name
        bname_candidates = jresp['label']
        if type(bname_candidates) is dict:
            bname_candidates = [bname_candidates]

        bname = None
        for bname_candidate in bname_candidates:
            if bname_candidate['@language'] == 'en':
                bname = bname_candidate['@value']
                break

        if bname is None:
            bname = bname_candidates[0]['@value']
        
        res['human_name']=bname
    
        if 'treeNumber' in jresp:
            # get tree ids (position in ontology tree)
            res['tree_positions'] = [x.split('/')[-1] for x in jresp['treeNumber']]

    if 'human_name' in res:
        return res['human_name']
    else:
        return bioentity_id


MAX_RETRY=2
NCBI_API_KEY="<YOUR_NCBI_API_KEY>"  # Replace with your actual NCBI API key

def getgene2(bioentity_ids):
    symbols=[]
    for bioentity_id in bioentity_ids.split(';'):
        # split on other specifiers eventually present
        if ':' in bioentity_id:
            bioentity_id = bioentity_id.split(':')[-1]
        # since june 2024
        url = 'https://api.ncbi.nlm.nih.gov/datasets/v2alpha/gene/id/'+bioentity_id
        headers = {'Accept': 'application/json', 'api-key':NCBI_API_KEY}
    
        ok = False
        trial=0
        while not ok and trial<MAX_RETRY:
            try:
                response = requests.get(url, headers=headers)
                orig_resp = jresp = response.json()
                ok = True
            except requests.exceptions.ConnectionError:
                time.sleep(1)
            trial+=1

        res = {}
        if 'reports' in jresp and len(jresp['reports'])>0:
            node = jresp['reports'][0]
            if 'gene' in node:
                gene = node['gene']
                if 'symbol' in gene:
                    res['human_name']=gene['symbol']
                if 'description' in gene:
                    res['description']=gene['description']

                res['species'] = {'id':None, 'human_name':None}
                if 'tax_id' in gene:
                    res['species']['id'] = gene['tax_id']
                if 'common_name' in gene:
                    res['species']['human_name'] = gene['common_name']
                elif 'taxname' in gene:
                    res['species']['human_name'] = gene['taxname']
                    
                res['ids'] = {'ensembl':None, 'omim':None, 'uniprot':None}
                if 'ensembl_gene_ids' in gene:
                    res['ids']['ensembl'] = gene['ensembl_gene_ids']
                if 'omim_ids' in gene:
                    res['ids']['omim'] =  gene['omim_ids']
                if 'swiss_prot_accessions' in gene:
                    res['ids']['uniprot'] =  gene['swiss_prot_accessions']

                if 'synonyms' in gene:
                    res['synonyms'] = gene['synonyms']
                if 'type' in gene:
                    res['category'] = gene['type']
                if 'orientation' in gene:
                    res['orientation'] = gene['orientation']
                if 'chromosomes' in gene:
                    res['chromosomes'] = gene['chromosomes']

        #symbols.append(symbol)
        
    if 'human_name' in res:
        return res['human_name']
    else:
        return bioentity_ids


def getgene(bioentity_id):
    url = 'https://www.ncbi.nlm.nih.gov/gene/'+bioentity_id+'?report=tabular&format=text'
    response = requests.get(url)
    h=3
    for i,l in enumerate(response.text.split('\n')):
        if i < h:
            continue
        row = l.split('\t')
        if len(row) <= 5:
            return getgene2(bioentity_id)
        symbol = row[5]
        if len(row) <= 6 and len(symbol)==0:
            return getgene2(bioentity_id)
        description = row[6]
        if len(row) <= 7 and len(symbol)==0 and len(description)==0:
            return getgene2(bioentity_id)
        other_name = row[7]
        if len(symbol)>0:
            return symbol 
        elif len(description) > 0:
            return description        
        else:
            return other_name