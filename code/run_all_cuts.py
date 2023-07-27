# A shell to place all cuts

import getpass
import os
import sys

event_name = sys.argv[1]
sims_data_both = sys.argv[2]
username = getpass.getuser()
sim_include = sys.argv[3:]

try:
    assert sims_data_both in ['sims', 'data', 'both']
except:
    print("ERROR: Second argument must be in ['sims', 'data', 'both']")
    sys.exit()

# SIMS
if sims_data_both in ['sims', 'both']:

    #check for good sims
    try:
        stream = open('../events/%s/logs/monitor_all_sims_report.log' %event_name, 'r')
        info = stream.readlines()
        stream.close()
        good_sims = info[0].strip().split(',')
    except:
        print("WARNING: Continuing on all sims even though some may not have finished. Stuff will probably break a lot.")
        good_sims = sim_include[:]
    #run in parallel
    for obj in good_sims:
        os.system('python run_single_cuts.py %s %s > ../events/%s/logs/process_%s.log &' %(event_name, obj, event_name, obj))

    #monitor progress
    os.system('python monitor_cut_progress.py %s %s' %(event_name, ' '.join(good_sims)))

# DATA
if sims_data_both in ['data', 'both']:

    fits_dir_prefix = 'LightCurvesReal'

    #convert dat to fits
    print("Reading datafiles")
    if not os.path.exists('../events/%s/sims_and_data/LightCurvesReal_FITS' %event_name):
        os.system('python convert_dat_to_fits.py %s' %event_name)

    print("Formatting data light curves")
    #convert fits files to numpy files
    if not os.path.exists('../events/%s/sims_and_data/LightCurvesReal_PYTHON' %event_name):
        #if this step is being rerun, the tarball may need to be unpacked
        if not os.path.exists('../events/%s/sims_and_data/%s_FITS' %(event_name, fits_dir_prefix)):
            os.chdir('../events/%s/sims_and_data/' %event_name)
            os.system('tar -xzf %s_FITS.tar.gz' %fits_dir_prefix)
            os.chdir('../../../code')
        os.system('python parse_fits.py %s %s' %(event_name, fits_dir_prefix)) 

    print("Placing cuts on data")
    #place cuts on fits files
    if not os.path.exists('../events/%s/cut_results/LightCurvesReal_cut_results.npy' %event_name):
        os.system('python get_cut_results.py %s %s' %(event_name, fits_dir_prefix))
    else:
        print("Cut results already exist, skipping cuts on DATA")

