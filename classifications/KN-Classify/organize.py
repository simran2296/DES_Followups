# A module to group a df by which features have real values

import pandas as pd

def bitrep(arr):
    bit_rep = ''
    for val in arr:
        if val:
            bit_rep += 'F'
        else:
            bit_rep += 'T'
    return bit_rep

def breakup(df):
    #df = pd.read_csv('../../events/test/KNC/KN_feats.csv')

    groups, bad_cols = {}, {}
        
    for index, row in df.iterrows():
        r = row.copy()
        
        br = bitrep(row != 'N')
        
        if br in groups.keys():
            groups[br].append(r.values)
        else:
            groups[br] = [r.values]
            bad_cols[br] = [x for x in df.columns if br[list(df.columns).index(x)] == 'T']
            
    datasets = {k: pd.DataFrame(data=v, columns=df.columns).drop(labels=bad_cols[k], axis=1) for k, v in groups.items()}

    return datasets

#def organize(event_name, sim_include='KN,Ia,CC,AGN,CaRT,ILOT,Mdwarf,SN91bg,Iax,PIa,SLSN,TDE'):
#    objs = sim_include.split(',')
#
#    for obj in objs:
        
    


    
