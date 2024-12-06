# A module to extract features and write to a csv

import getpass
import numpy as np
import os
import pandas as pd
import sys

from features import FeatureExtractor


def extract(lc, flts, FExtractor):
    #use good quality points only
    cond = (lc['PHOTFLAG'].values > 4095)
    good_lc = lc[cond].reset_index(drop=True)
    if good_lc.shape[0] == 0:
        return {}

    good_flts = np.unique(good_lc['FLT'].values)

    data_dict = {}
    for flt in flts:
        if flt in good_flts:
            flt_good = True
        else:
            flt_good = False
        for feat in FExtractor.single_features:
            if flt_good:
                data_dict[feat + '_' + flt] = eval('FExtractor.%s(lc, None, flt)' %feat)
            else:
                data_dict[feat + '_' + flt] = 'N'

    for pair in ['gr', 'gi', 'gz', 'ri', 'rz', 'iz']:
        for feat in FExtractor.double_features:
            data_dict[feat + '_' + pair] = eval('FExtractor.%s(lc, None, pair[0], pair[1])' %feat)
        
    return data_dict
            

def extract_all_to_csv(cut_res, obj, outfile, cut_requirement=-1):
    FExtractor = FeatureExtractor()

    #Track progress
    #total = float(len(list(cut_res.keys())))
    #counter = 0.0

    data = []
    for snid, info in cut_res.items():
        
        #output progress
        #counter += 1
        #progress = counter / total * 100.0
        #sys.stdout.write('\rProgress:  %.2f %%' %progress)
        #sys.stdout.flush()

        #extract features if cut_requirement is met
        if info['cut'] == cut_requirement:
            flts = np.unique(info['lightcurve']['FLT'].values)
            data_dict = extract(info['lightcurve'], flts, FExtractor)
            data_dict['SNID'] = snid
            data_dict['OBJ'] = obj
            data.append(data_dict)

    df = pd.DataFrame(data)
    #drop rows where every values is NaN (these were events with no good observations)
    df = df.dropna(how='all')

    #replace NaNs with 'N' to be consistent
    df = df.fillna('N')

    df.to_csv(outfile, index=False)
    return

def run_extraction(event_name, obj, cut_requirement=-1):
    username = getpass.getuser()
    cut_res_filename = '../../events/%s/cut_results/%s_DESGW_%s_%s_cut_results.npy' %(event_name, username, event_name, obj)
    cut_res = np.load(cut_res_filename).item()
    if not os.path.exists('../../events/%s/KNC' %event_name):
        os.system('mkdir ../../events/%s/KNC'%event_name)
    outfile = '../../events/%s/KNC/%s_feats.csv' %(event_name, obj)
    extract_all_to_csv(cut_res, obj, outfile, cut_requirement)
    return

