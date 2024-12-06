# a shell to call place cuts on a given dataset and create a report

import numpy as np
import os
import pandas as pd
import sys

event_name = sys.argv[1]
fits_dir_prefix = sys.argv[2]
transient_class = fits_dir_prefix.split('_')[-1]

# assume at this point we only have fits files
## i.e., run convert_dat_to_fits.py as a preprocessing step

# place cuts
if not os.path.exists('../events/%s/cut_results/%s_cut_results.npy'  %(event_name, fits_dir_prefix)):
    os.system('python place_cuts.py %s %s' %(event_name, fits_dir_prefix))
cut_results = np.load('../events/%s/cut_results/%s_cut_results.npy' %(event_name, fits_dir_prefix), allow_pickle=True).item()


# build up cut summary table and candidate table
cut_filename = '../events/%s/cuts.csv' %event_name
cut_df = pd.read_csv(cut_filename)

#print("Cuts to be placed:")
#print(cut_df)

cut_by_dict = { str(x) : 0 for x in cut_df['NUMBER'].values }


table_output = []
table_output_columns = ['SNID', 'RA', 'DEC'] #more can be added later
for snid, info in cut_results.items():
    
    cut_by = str(info['cut'])
    
    #in this case, the object was cut out
    if cut_by != '-1':
        #print(cut_by, cut_by_dict.keys())
        cut_by_dict[cut_by] += 1
        
    #in this case, the object has passed cuts and is a candidate
    else:
        #Track all needed quantities for vetting
        table_output.append([snid,
                             info['metadata']['RA'],
                             info['metadata']['DEC']])

#write candidate table to df and csv
output_df = pd.DataFrame(data=table_output, columns=table_output_columns)
output_df.to_csv('../events/%s/cut_results/%s_candidate_summary.txt' %(event_name, fits_dir_prefix))
        

#write cut summary table
total = len(list(cut_results.keys()))
cut_out_already = 0
table_data = ['CUT\tREMAINING\tNAME\n', '\t'.join(['0', str(total), '\tNone', '\n'])]
for index, row in cut_df.iterrows():
    
    cut_out_already += cut_by_dict[str(row['NUMBER'])]
    remaining = total - cut_out_already

    table_info = '\t'.join([str(row['NUMBER']), str(remaining), '\t' + str(row['NAME']), '\n'])
    table_data.append(table_info)

out_data = ''.join(table_data)

out_file_name = '../events/%s/cut_results/%s_cut_summary.txt' %(event_name, fits_dir_prefix)
out_file = open(out_file_name, 'w+')
out_file.write(out_data)
out_file.close()
                       

#do a final check
try:
    assert remaining == output_df.shape[0]
except:
    print("WARNING: Number of remaining objects (%i) does not agree with number of candidates (%i)." %(remaining, output_df.shape[0]))
    
os.system('touch ../events/%s/logs/get_results_%s.DONE' %(event_name, transient_class))
