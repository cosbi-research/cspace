#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
this script computes a exponentially-decayed weighted average of the difference in gain between word2vec training epochs.
Exponentially weighted because we want later epochs count less than early epochs.
Difference in gain of epoch k is absVal_k - absVal_k-1
The base-parameter for the exponential function is ALPHA
"""

import sys,os,re
import pickle
import numpy as np

evaluation_re=re.compile('(\w+).([\w-]+).?([0-9]+)?.tsv')
oov_re=re.compile('(\w+).(\w+).test.oov.tsv')
ALPHA=0.95
MAX_EPOCHS=151
basepath = sys.argv[1]
#out_file = sys.argv[2]

# pre-compute weights
bases = np.array([ALPHA]*MAX_EPOCHS)
exponents = np.power(np.arange(1,MAX_EPOCHS+1), 1.4)
weights = np.power(bases, exponents)

important_datasets=set(['MayoTerms','UMNSRS_similarity_Terms','UMNSRS_relatedness_Terms'])
# mayo is the most important to consider
important_datasets_weights ={'MayoTerms': 1.0, 'UMNSRS_similarity_Terms': 0.7, 'UMNSRS_relatedness_Terms': 0.7}

#fout = open(out_file, 'w')
fout = sys.stdout
fout.write('\t'.join(['Algorithm', 'Query', 'Dataset','Final perf','Final perf Std.Dev','Max perf', 'Max perf Std.Dev',
                      'Max perf epoch','Exponentially Weighted Diff Average', 'Exponentially Weighted Diff Std.Dev',
                      'OOV Ratio', 'Vector Size', 'Window Size', 'α', 'min α', 'CBOW',
                      'Skip-Gram', 'Negative Sampling', 'NS Exponent', 'Max N', 'Down-sample Threshold'])+'\n')

for root, dirs, files in os.walk(basepath):
    if len(files) > 0:
        # leaf folder
        params_str = root.split(os.sep)
        params = {
            'vector_size':152,
            'window_size':30,
            'alpha':0.025,
            'min_alpha': '0.0001',
            'cbow':0,
            'skip_gram':1,
            'negative_sampling':30,
            'negative_sampling_exponent':0.75,
            'max_n':6,
            'sample': 0.001
        }
        for paramstr in params_str:
            if paramstr.startswith('v'):
                # vector size
                params['vector_size'] = int(paramstr[1:])
            elif paramstr.startswith('w'):
                params['window_size'] = int(paramstr[1:])
            elif paramstr.startswith('a'):
                params['alpha'] = paramstr[1:]
            elif paramstr.startswith('cbow'):
                params['cbow'] = 1
                params['skip_gram'] = 0
            elif paramstr.startswith('sg'):
                params['skip_gram'] = 1
                params['cbow'] = 0
                params['negative_sampling']=int(paramstr[2:])
            elif paramstr.startswith('neg_exp'):
                params['negative_sampling_exponent'] = float(paramstr[7:])
            elif paramstr.startswith('maxn'):
                params['max_n'] = int(paramstr[4:])
            elif paramstr.startswith('sample'):
                params['sample'] = float(paramstr[6:])
            elif paramstr.startswith('ma'):
                params['min_alpha'] = paramstr[2:]

        data={}
        for fname in files:
            em = evaluation_re.match(fname)
            if em is not None:
                # version
                ver = em.group(3)
                ver = ver if ver is not None and len(ver)>0 else 'default'
                query_name =em.group(1)
                algorithm = em.group(2)
                if query_name not in data:
                    data[query_name] = {algorithm:{'oov':{}}}
                elif algorithm not in data[query_name]:
                    data[query_name][algorithm] = {'oov':{}}
                
                # get pearson correlation list for each dataset
                with open(os.path.join(root, fname)) as f:
                    header = True
                    for line in f:
                        # skip header
                        if header:
                            header=False
                            continue
                        try:
                            ds,pear_cor,pear_p,spear_cor,spear_p,oov_ratio,epoch = line.split('\t')
                        except:
                            print("ERROR: "+line)
                            continue
                        
                        if ds not in data[query_name][algorithm]:
                            # 0=base to be used for differences
                            data[query_name][algorithm][ds]={ver: {0: 0.0}}
                        elif ver not in data[query_name][algorithm][ds]:
                            data[query_name][algorithm][ds][ver]={0: 0.0}
                        elif epoch not in data[query_name][algorithm][ds][ver]:
                            data[query_name][algorithm][ds][ver][0] = 0.0
                        
                        curcor = float(pear_cor)
                        data[query_name][algorithm][ds][ver][int(epoch.rstrip('\n'))] = curcor
                        
                        if ds not in data[query_name][algorithm]['oov']:
                            data[query_name][algorithm]['oov'][ds]=oov_ratio
                            
        # compute differences
        for query_name in data:
            for algorithm in data[query_name]:
                oov_ratios = data[query_name][algorithm]['oov']

                datasets_dfinal={}
                for ds in data[query_name][algorithm]:
                    if ds == 'oov':
                        continue
                    pear_cors_ds = data[query_name][algorithm][ds]
                    # average across all versions
                    dvals=[]
                    # list of final performances
                    dfinal = []
                    maxlen=0
                    pear_cors_ds_max=[]
                    pear_cors_ds_maxepoch=[]
                    # matrix with epochs on columns and experiments on rows
                    ver_diffs = np.empty((len(pear_cors_ds.values()),MAX_EPOCHS,))
                    ver_diffs.fill(np.nan)
                    for exp_num, ver_epochs in enumerate(pear_cors_ds.values()):
                        max_epoch=0
                        ver_max_value = np.nan
                        vmax = 0
                        vmax_epoch=0
                        prev_epoch_value = np.nan
                        for epoch in range(MAX_EPOCHS):
                            if epoch in ver_epochs:
                                curval = ver_epochs[epoch]
                                
                                if np.isnan(prev_epoch_value):
                                    prev_epoch_value = curval
                                else:
                                    # store difference between this and the previous epoch
                                    ver_diffs[exp_num][epoch] = curval - prev_epoch_value
                                    # update prev value
                                    prev_epoch_value = curval
                                
                                #if epoch > max_epoch: obviously true
                                max_epoch= epoch
                                ver_max_value = curval

                                if vmax < curval:
                                    vmax = curval
                                    vmax_epoch = epoch

                        dfinal.append(ver_max_value)
                        #if pear_cors_ds_max < vmax:
                        pear_cors_ds_max.append( vmax )
                        pear_cors_ds_maxepoch.append( vmax_epoch )

                    # mask so that nans are ignored
                    # results in a masked matrix with epochs improvements (diffs) on columns, and experiments on rows
                    # masked array of shape = n_experiments, max_epochs
                    exp_diffs = np.ma.masked_array(ver_diffs, np.isnan(ver_diffs))
                    # weighted average of diffs for each experiment (ignoring nans)
                    # returns a np.array of shape = n_experiments, 1
                    exp_avg = np.ma.average(exp_diffs, axis=1, weights=weights).filled()
                    # average of averages, and std dev of averages
                    avg = np.mean(exp_avg, axis=0)
                    std = np.std(exp_avg, axis=0)

                    if ds not in datasets_dfinal:
                        datasets_dfinal[ds]=[]
                    
                    datasets_dfinal[ds].extend(dfinal)
                    #['Algorithm', 'Query', 'Dataset','Final performance', 'Final performance Std.Dev','Max performance', Max perf stddev,'Max perf epoch'
                    # 'Exponentially Weighted Diff Average', 'Exponentially Weighted Diff Std.Dev',
                    #  'OOV Ratio', 'Vector Size', 'Window Size', 'Alpha', 'CBOW',
                    #  'Skip-Gram', 'Negative Sampling', 'NS Exponent', 'Max N']
                    fout.write('\t'.join([
                        algorithm,
                        query_name,
                        ds,
                        str(np.mean(dfinal)),
                        str(np.std(dfinal)),
                        str(np.mean(pear_cors_ds_max)),
                        str(np.std(pear_cors_ds_max)),                        
                        str(min(pear_cors_ds_maxepoch)),
                        str(avg),
                        str(std),
                        oov_ratios[ds],
                        str(params['vector_size']),
                        str(params['window_size']),
                        str(params['alpha']),
                        str(params['min_alpha']),
                        str(params['cbow']),
                        str(params['skip_gram']),
                        str(params['negative_sampling']),
                        str(params['negative_sampling_exponent']),
                        str(params['max_n']),
                        str(params['sample'])
                    ])+'\n')


