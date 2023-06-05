# A module to place cuts on light curves

import os
import pandas as pd
import numpy as np
import sys

#import cuts

event_name = sys.argv[1]
fits_dir_prefix = sys.argv[2]

# Default to local cuts first
sys.path = ['../events/%s' %event_name] + sys.path
import cuts

#deal with output data
transient_class = fits_dir_prefix.split('_')[-1]
#process_log_file = '../events/%s/logs/cut_%s.log' %(event_name, transient_class)

#parse cut_file
cut_filename = '../events/%s/cuts.csv' %event_name
cut_df = pd.read_csv(cut_filename)

#initialize cutlist object
cutlist = cuts.CutList()
all_cuts = cutlist.all_cuts

#check that all cuts in cut_df are defined
undefined_cuts = []
for index, row in cut_df.iterrows():
    if row['NAME'] not in all_cuts:
        undefined_cuts.append(row['NAME'])

if len(undefined_cuts) != 0:
    print("ERROR: The following cuts are not defined in cuts.py: ")
    print(undefined_cuts)

    print("Note: If you have a `cuts.py` file in your event directory, the program will find that first")
    sys.exit()

#load light curve dictionary
data_dict = np.load('../events/%s/sims_and_data/%s_PYTHON/%s.npy' %(event_name, fits_dir_prefix, fits_dir_prefix)).item()
# change the 
#data_dict = data_dict.rename(columns={'BAND': 'FLT'})
#dfnew = df.rename(columns={'popu': 'population'})
#ini_dict['akash'] = ini_dict.pop('akshat')
#data_dict['FLT'] = data_dict.pop('BAND')
#data_dict['lightcurve']['FLT'] = data_dict.pop(['lightcurve']['BAND'])
#del data_dict['BAND']
#data_dict['lightcurve'].rename(columns={"BAND": "FLT"})#, "B": "c"})
#data_dict['1']['lightcurve'].rename(columns={"BAND": "FLT"})

#place cuts in order, save snids passing each cut in list files
# do this by programatically building up a code block
cut_code_block = 'cut_by = 0\n\n'
for index, row in cut_df.iterrows():

    cut_code_block += 'if cutlist.%s(lc, md):\n' %row['NAME']
    cut_code_block += '    ' * (index + 1)
    cut_code_block += 'cut_by += 1\n'
    cut_code_block += '    ' * (index + 1)

compiled_cut_code_block = compile(cut_code_block, '<string>', 'exec')

#place cuts by applying cut_code_block to each light curve
cut_results = {}
for snid, info in data_dict.iteritems():
    
    lc = info['lightcurve']
    #print(info)
    md = info['metadata']
    #exec(compiled_cut_code_block)
    lc['FLT'] = lc['BAND']
    #print(lc)
    exec(compiled_cut_code_block)
    #if not cut, set cut_result to -1. if cut, set cut_result to cut number 
    if cut_by == cut_df.shape[0]:
        cut_results[snid] = {'cut': -1, 'lightcurve': lc, 'metadata': md}
    else:
        cut_results[snid] = {'cut': cut_by + 1, 'lightcurve': lc, 'metadata': md}


#save output    
if not os.path.exists('../events/%s/cut_results' %event_name):
    os.system('mkdir ../events/%s/cut_results' %event_name)

np.save('../events/%s/cut_results/%s_cut_results.npy' %(event_name, fits_dir_prefix), cut_results, allow_pickle=True)

os.system('touch ../events/%s/logs/cut_%s.DONE' %(event_name, transient_class))
