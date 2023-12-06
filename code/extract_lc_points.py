#A module to plot mean light curves of all sims passing cuts

import getpass
import numpy as np
import pandas as pd
import sys

event_name = sys.argv[1]
mode = sys.argv[2]
assert mode in ['data', 'sims']
username = getpass.getuser()

cut_res_path = '../events/%s/cut_results/' %event_name

if mode == 'sims':
    #load cut results
    cc_res = np.load(cut_res_path + '%s_DESGW_%s_CC_cut_results.npy' %(username, event_name)).item()
    print("CC Loaded")
    ia_res = np.load(cut_res_path + '%s_DESGW_%s_Ia_cut_results.npy' %(username, event_name)).item()
    print("Ia Loaded")
    kn_res = np.load(cut_res_path + '%s_DESGW_%s_KN_cut_results.npy' %(username, event_name)).item()
    print("KN Loaded")

    #iterate through and get lightcurves
    labels = ['CC', 'Ia', 'KN']
    results = [cc_res, ia_res, kn_res]

elif mode == 'data':
    #load cut results
    data_res = np.load(cut_res_path + 'LightCurvesReal_cut_results.npy').item()
    print("Data Loaded")

    #iterate through and get lightcurvea
    labels = ['DATA']
    results = [data_res]

else:
    print("Something is fucked")



for res, label  in zip(results, labels):
    print(' ')
    total = float(len(list(res.keys())))
    counter = 0.0

    out_data = []

    for snid, info in res.iteritems():

        counter += 1.0
        if int(counter) % 100 == 0:
            progress = counter / total * 100.0
            sys.stdout.write('\r%s Progress: %.2f %%' %(label, progress))
            sys.stdout.flush()

        if int(info['cut']) < 0:
            
            for index, row in info['lightcurve'].iterrows():
                if int(row['PHOTFLAG']) > 4095:
                    if mode == 'sims':
                        out_data.append([snid, 
                                     row['MJD'],
                                     row['FLT'],
                                     row['FLUXCAL'],
                                     row['FLUXCALERR'],
                                     info['metadata']['SIM_REDSHIFT_CMB'],
                                     info['metadata']['SIM_MODEL_INDEX'],
                                     info['metadata']['SIM_TEMPLATE_INDEX']
                                     ])
                    else:
                        out_data.append([snid,
                                     row['MJD'],
                                     row['FLT'],
                                     row['FLUXCAL'],
                                     row['FLUXCALERR'],
                                     -9,
                                     -9,
                                     -9
                                     ])

    df = pd.DataFrame(data=out_data, columns=['SNID', 'MJD', 'FLT', 'FLUXCAL', 'FLUXCALERR', 'REDSHIFT', 'MODEL', 'TEMPLATE'])
    df.to_csv('../events/%s/kn_analysis/%s_lc_data.csv' %(event_name, label)) 
