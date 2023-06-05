# A code to assess the properties of KN after each cut

import getpass
import pandas as pd
import numpy as np
import sys

event_name = sys.argv[1]
username = getpass.getuser()

#read kn model info
info_df = pd.read_csv('../templates/KN_SED.INFO', delim_whitespace=True, skiprows=8, 
                      names=['PARNAMES:', 'SED_FILE', 'SIM_TEMPLATE_INDEX', 'VK', 'LOGXLAN', 'LOGMASS'])



print("Loading cut results")
#read cut results
kn_results = np.load('../events/%s/cut_results/%s_DESGW_%s_KN-tr_cut_results.npy' %(event_name, username, event_name), allow_pickle=True, encoding='latin1').item()
#kn_results = np.load('../events/%s/cut_results/LightCurvesReal_cut_results.npy' %(event_name), allow_pickle=True, encoding='latin1').item()


total = len(list(kn_results.keys()))
counter = 0.0


## build a one time csv to make plot and export
data = []
columns = ['SNID', 'SIM_TEMPLATE_INDEX', 'CUT', 'PEAKMAG_g', 'PEAKMAG_r', 'PEAKMAG_i', 'PEAKMAG_z']
for snid, info in kn_results.items():

    counter += 1.0
    if int(counter) % 100 == 0:
        progress = counter / total * 100.0
        sys.stdout.write('\rProgress: %.2f %%   ' %progress)
        sys.stdout.flush()

    info['lightcurve']['FLT'] = info['lightcurve']['BAND']

    g_data = [float(x) for x in info['lightcurve'][info['lightcurve']['FLT'] == 'g']['FLUXCAL'].values]
    r_data = [float(x) for x in info['lightcurve'][info['lightcurve']['FLT'] == 'r']['FLUXCAL'].values]
    i_data = [float(x) for x in info['lightcurve'][info['lightcurve']['FLT'] == 'i']['FLUXCAL'].values]
    z_data = [float(x) for x in info['lightcurve'][info['lightcurve']['FLT'] == 'z']['FLUXCAL'].values]
    
    if len(g_data) != 0:
        if np.max(g_data) > 0:
            peakmag_g = 27.5 - 2.5 * np.log10(np.max(g_data))
        else:
            peakmag_g = 27.5
    else:
        peakmag_g = 99.0
        
    if len(r_data) != 0:
        if np.max(r_data) > 0:
            peakmag_r = 27.5 - 2.5 * np.log10(np.max(r_data))
        else:
            peakmag_r = 27.5
    else:
        peakmag_r = 99.0

    if len(i_data) != 0:
        if np.max(i_data) > 0:
            peakmag_i = 27.5 - 2.5 * np.log10(np.max(i_data))
        else:
            peakmag_i = 27.5
    else:
        peakmag_i = 99.0
        
    if len(z_data) != 0:
        if np.max(z_data) > 0:
            peakmag_z = 27.5 - 2.5 * np.log10(np.max(z_data))
        else:
            peakmag_z = 27.5
    else:
        peakmag_z = 99.0

    row = [snid, info['metadata']['SIM_TEMPLATE_INDEX'], info['cut'], peakmag_g, peakmag_i, peakmag_r, peakmag_z]
    data.append(row)

df = pd.DataFrame(data=data, columns=columns)

df = df.merge(info_df, on='SIM_TEMPLATE_INDEX')
df.to_csv('../events/%s/analysis/%s_kn_terse_cut_results.csv' %(event_name, event_name))

print("\nDone!")









