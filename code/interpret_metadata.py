# A module to automatically construct metadata information from dat files

import glob
import numpy as np
from optparse import OptionParser
import os
import pandas as pd
import sys
import time

import utils

event_name = sys.argv[1]

if len(sys.argv) > 2:
    parser = OptionParser(__doc__)
    parser.add_option('--boost', default='', help="Number of objects to simulate")
    parser.add_option('--sim_include', default='', help="Objects to simulate")
    options, args = parser.parse_args(sys.argv[2:])
    boost = options.boost.split(',')
    sim_include = options.sim_include.split(',')
else:
    sys.exit()

dat_file_path_file = open('../events/%s/dat_file_path.txt' %event_name, 'r')
dat_file_path = dat_file_path_file.readlines()[0]
dat_file_path_file.close()
if dat_file_path[-1] == '\n':
    dat_file_path = dat_file_path[0:-1]
if dat_file_path[-1] != '/':
    dat_file_path = dat_file_path + '/'

#copy dat files to event directory for future use
if not os.path.exists('../events/%s/sims_and_data/LightCurvesReal' %event_name):
    os.system('mkdir ../events/%s/sims_and_data/' %event_name)
    os.system('cp -r %s ../events/%s/sims_and_data &' %(dat_file_path, event_name))
    total = len(glob.glob('%s*.dat' %dat_file_path))
    done_copying = False
    while not done_copying:
        num_copied = len(glob.glob('../events/%s/sims_and_data/LightCurvesReal/*.dat' %event_name))
        if num_copied > total - 5:
            done_copying = True
        
        time.sleep(2)
        progress = float(num_copied) / total * 100.0
        sys.stdout.write('\rCopying dat files:  %.2f %%' %progress)
        sys.stdout.flush()

sys.stdout.write("\rCopying dat files:  Done!               \n")
sys.stdout.flush()

os.chdir('../events/%s/sims_and_data/LightCurvesReal/' %event_name)
os.system('chmod +rwx *.dat')
os.chdir('../../../../code')

#os.system("python removeNITE.py %s" %(event_name))


if not os.path.exists('../events/%s/exptable.txt' %event_name) and not os.path.exists('../events/%s/sim_gen/SIMLIB.txt' %event_name):
    
    ras, decs, expnums = [], [], []
    dat_files = glob.glob('../events/%s/sims_and_data/LightCurvesReal/*.dat' %event_name)

    total = len(dat_files)
    assert total > 0, "No dat files found"
    counter = 0.0
    for dat_file in dat_files:
        #Track progress
        counter += 1.0
        if total % int(counter) == 10:
            progress = counter / total * 100.0
            sys.stdout.write('\rReading dat files:  %.2f %%    ' %progress)
            sys.stdout.flush()
    
        #parse the dat file
        lines = utils.open_dat(dat_file)
        lc = utils.get_terse_lc(lines)
        md = utils.get_meta_data(lines)
    
        #save ra and dec
        ras.append(float(md['RA']))
        decs.append(float(md['DEC']))
    
        #save expnums
        expnums += [int(x) for x in lc['EXPNUM'].values]


    ### calculate necessart metadata values
    min_ra = np.min(ras) - 5.0
    max_ra = np.max(ras) + 5.0
    min_dec = np.min(decs) - 5.0
    max_dec = np.max(decs) + 5.0


    #compile into a csv
    cols = ['NAME', 'MIN_RA', 'MAX_RA', 'MIN_DEC', 'MAX_DEC']
    cols += ['BOOST--' + x for x in sim_include]
            

    data = [[event_name, min_ra, max_ra, min_dec, max_dec] + boost]

    event_info = pd.DataFrame(data=data, columns=cols)
    event_info.to_csv('../events/%s/DES_metadata.csv' %event_name, index=False)

    #merge with LIGO csv to make full event_metadata
    ligo_df = pd.read_csv('../events/%s/LIGO_metadata.csv' %event_name)
    
    event_metadata_df = ligo_df.merge(event_info, on='NAME')
    event_metadata_df.to_csv('../events/%s/event_metadata.csv' %event_name, index=False)

    #make exptable
    exptable_file = open('../events/%s/exptable.txt' %event_name, 'w+')
    expnums = np.unique(expnums)
    formatted_expnums = [str(x) + '\n' for x in expnums]
    exptable_file.writelines(formatted_expnums)
    exptable_file.close()

sys.stdout.write("\rReading dat files:  Done!                  \n")
sys.stdout.flush()
