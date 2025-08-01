from utils import getmesh,getgene

def get_names( ls ):
    for ck,v in ls:
        otype=None
        oid=None
        if ck.startswith('disease_mesh_'):
            oid=ck[len('disease_mesh_'):]
            otype='mesh'
        elif ck.startswith('chemical_mesh_'):
            oid=ck[len('chemical_mesh_'):]
            otype='mesh'
        elif ck.startswith('gene_'):
            oid=ck[len('gene_'):]
            otype='gene'

        if otype == 'mesh':
            ck_name=getmesh(oid)
            yield (ck_name,v)
        elif otype == 'gene':
            ck_name=getgene(oid)
            yield (ck_name,v)
        else:
            yield(ck,v)

def shared_similarities(model, c1, c2, topn=200):
    ls1=model.most_similar(c1,topn=topn)
    ls2=model.most_similar(c2,topn=topn)
    #s1={k:v for k,v in ls1}
    s2={k:v for k,v in ls2}
    # get in-order the concepts that are in common between
    # c1 and c2, starting from the most similar to c1
    for ck_name,v in get_names(filter(lambda item: item[0] in s2, ls1)):
        yield (ck_name, v)

def most_similar_genes(model, c1, topn=1000):
    for k,v in model.most_similar(c1,topn=topn):
        if k.startswith('gene_'):
            oid=k[len('gene_'):]
            ck_name=getgene(oid)
            yield (ck_name,k,v)

def most_similar_diseases(model, c1, topn=1000):
    for k,v in model.most_similar(c1,topn=topn):
        if k.startswith('disease_mesh_'):
            oid=k[len('disease_mesh_'):]
            ck_name=getmesh(oid)
            yield (ck_name,k,v)

def most_similar_chemical(model, c1, topn=1000):
    for k,v in model.most_similar(c1,topn=topn):
        if k.startswith('chemical_mesh_'):
            oid=k[len('chemical_mesh_'):]
            ck_name=getmesh(oid)
            yield (ck_name,k,v)
