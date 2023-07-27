# Shell to call all simulation generation functions

import getpass
import numpy as np
import os
import pandas as pd
import sys

#read input data
event_name = sys.argv[1]
df = pd.read_csv('../events/%s/event_metadata.csv' %event_name)
username = getpass.getuser()

#read forced conditions
## psf,skymag,deltat (no spaces, comma-separated)
## use 'real' if real conditions are desired
forced_conditions = sys.argv[2]

nobs_force = sys.argv[3]

try:
    obj_to_simulate = sys.argv[4:]
except:
    print("ERROR: Provide space-separated list of objects to generate_sims.py")
    sys.exit()


#check for and nuke existing sims directory
sim_path = '../events/%s/sim_gen' % event_name
if not os.path.exists(sim_path):
    os.system('mkdir %s' %sim_path)


log_path = '../events/%s/logs' %event_name
if not os.path.exists(log_path):
    os.system('mkdir %s' %log_path)


if not os.path.exists('../events/%s/sim_gen/SIMLIB.txt' %event_name):
    print("Querying DB for observing conditions")
    #call makeSimlib_easyaccess.py
    ## run 10 times and track averate and std of returned effective areas
    areas = []
    total = 10
    for i in range(total):
        progress = float(i + 1) / 10 * 100
        sys.stdout.write('\rProgress:  %i %%' %int(progress))
        sys.stdout.flush()
        os.system('python makeSimlib_easyaccess.py -o "../events/%s/sim_gen/SIMLIB_raw.txt"' %event_name + 
                  ' --exptable "../events/%s/exptable.txt"' %event_name +
                  ' --min_ra %.1f --max_ra %.1f --min_dec %.1f --max_dec %.1f > %s/make_simlib.log' %(df['MIN_RA'].values[0], df['MAX_RA'].values[0],
                                                                                                      df['MIN_DEC'].values[0], df['MAX_DEC'].values[0], log_path))
        f = open("../events/%s/sim_gen/SIMLIB_raw.txt" %event_name, 'r')
        eff_area = f.readlines()[-1]
        f.close()
        areas.append(eff_area.split(' ')[-1])

    #call clean_simlib.py
    areas = [float(x) for x in areas]
    area, std = np.mean(areas), np.std(areas)
    log_file = open('../events/%s/logs/eff_area.log' %event_name, 'w+')
    log_file.write('%.6f' %std)
    log_file.close()
    os.system('python clean_simlib.py ../events/%s/sim_gen/SIMLIB_raw.txt ../events/%s/sim_gen/SIMLIB_0.txt %.6f %s' %(event_name, event_name, area, nobs_force))
    #rerun until length of file does not change
    os.system('touch ../events/%s/sim_gen/SIMLIB_0.txt' %event_name)
    simlib_file = open('../events/%s/sim_gen/SIMLIB_raw.txt' %event_name, 'r')
    simlib_lines = len(simlib_file.readlines())
    simlib_file.close()
    new_simlib_file = open('../events/%s/sim_gen/SIMLIB_0.txt'%event_name, 'r')
    new_simlib_lines = len(new_simlib_file.readlines())
    new_simlib_file.close()
    counter = 0
    while simlib_lines != new_simlib_lines:
        os.system('python clean_simlib.py ../events/%s/sim_gen/SIMLIB_%i.txt ../events/%s/sim_gen/SIMLIB_%i.txt %.6f %s' %(event_name, counter, event_name, counter + 1, area, nobs_force))
        simlib_file = open('../events/%s/sim_gen/SIMLIB_%i.txt' %(event_name, counter), 'r')
        simlib_lines = len(simlib_file.readlines())
        simlib_file.close()
        new_simlib_file = open('../events/%s/sim_gen/SIMLIB_%i.txt' %(event_name, counter), 'r')
        new_simlib_lines = len(new_simlib_file.readlines())
        new_simlib_file.close()
        counter += 1
    os.system('mv ../events/%s/sim_gen/SIMLIB_%i.txt ../events/%s/sim_gen/SIMLIB.txt' %(event_name, counter, event_name))
    nfs_simlib = open('../events/%s/sim_gen/nfs_SIMLIB.txt' %event_name,'w')
    DOCANA_TEXT = '''DOCUMENTATION:
    PURPOSE: DES 5-year cadence for SNANA simulation
    REF:
    - AUTHOR: Kessler et al. 2015 (DIFFIMG pipeline)
      ADS:    https://ui.adsabs.harvard.edu/abs/2015AJ....150..172K
    INTENT:   Nominal
    USAGE_KEY:  SIMLIB_FILE
    USAGE_CODE: snlc_sim.exe
    VALIDATE_SCIENCE:
    - Y1 only, Figs 11,12,13 (arxiv.org/abs/1507.05137)
    - DES-SN3YR cosmology results (arxiv.org/abs/1811.02374)
    - SNANA sim for DES bias corrections (arxiv.org/abs/1811.02379)
    - CC and host sims (arxiv.org/abs/2012.07180)
    NOTES:
    - based on Difference Imaging
    VERSIONS:
    - DATE:  2020-02
      AUTHORS: R. Kessler
DOCUMENTATION_END:

# =======================================

'''
    nfs_simlib.write(DOCANA_TEXT)
    
    real_simlib = open('../events/%s/sim_gen/SIMLIB.txt' %event_name,'r')
    nfs_lines = real_simlib.readlines()
    real_simlib.close()
    for nfs_line in nfs_lines:
        nfs_simlib.write(nfs_line)

    nfs_simlib.close()
    
    os.system('mv ../events/%s/sim_gen/SIMLIB.txt ../events/%s/sim_gen/SIMLIB_OLD.txt' %(event_name, event_name))
    os.system('mv ../events/%s/sim_gen/nfs_SIMLIB.txt ../events/%s/sim_gen/SIMLIB.txt' %(event_name, event_name))
else:
    print("Existing SIMLIB found, skipping simlib generation")

if not os.path.exists('../events/%s/sim_gen/SIMGEN_DES_NONIA.input' %event_name):
    #copy templates to sims directory
    os.system('cp ../templates/* ../events/%s/sim_gen/' %event_name)

    #call update_inputs.py
    os.system('python update_inputs.py %s %s' %(event_name, forced_conditions))
else:
    print("Existing SIM_INPUTS found, skipping update inputs")

#generate and retrieve simulations
os.chdir('../events/%s' %event_name)
if not os.path.exists('sims_and_data'):
    os.system('mkdir sims_and_data')

sim_dir_prefix = '/data/des41.b/data/SNDATA_ROOT/SIM/' + username + '_DESGW_' + event_name + '_'
print("Simulating light curves...")


###
#Parallelized and monitored implementation
###

obj_simfile_map = {'AGN': 'AGN_SIMGEN.INPUT',
                   'KN': 'SIMGEN_DES_KN.input',
                   'CC': 'SIMGEN_DES_NONIA.input',
                   'Ia': 'SIMGEN_DES_SALT2.input',
                   'CaRT': 'CART_SIMGEN.INPUT',
                   'ILOT': 'ILOT_SIMGEN.INPUT',
                   'Mdwarf': 'Mdwarf_SIMGEN.INPUT',
                   'SN91bg': 'SN_Ia91bgSIMGEN.INPUT',
                   'Iax': 'SN_IaxSIMGEN.INPUT',
                   'PIa': 'SN_PIaSIMGEN.INPUT',
                   'SLSN': 'SN_SLSNSIMGEN.INPUT',
                   'TDE': 'TDE_SIMGEN.INPUT',
                   'BBH': 'SIMGEN_DES_BBH.input'}
for k in obj_simfile_map.keys():
    obj_simfile_map[k + '-tr'] = 'TR_' + obj_simfile_map[k]

obj_logfile_map = {'AGN': 'sim_agn.log',
                   'KN': 'sim_kn.log',
                   'CC': 'sim_cc.log',
                   'Ia': 'sim_ia.log',
                   'CaRT': 'sim_cart.log',
                   'ILOT': 'sim_ilot.log',
                   'Mdwarf': 'sim_mdwarf.log',
                   'SN91bg': 'sim_91bg.log',
                   'Iax': 'sim_iax.log',
                   'PIa': 'sim_pia.log',
                   'SLSN': 'sim_slsn.log',
                   'TDE': 'sim_tde.log',
                   'BBH': 'sim_bbh.log'}

for k in obj_logfile_map.keys():
    obj_logfile_map[k +'-tr'] = 'sim_tr_' + obj_logfile_map[k].split('_')[-1]

os.chdir('sim_gen')
args = []
for obj in obj_to_simulate:
    if not os.path.exists('../sims_and_data/%s_DESGW_%s_%s_FITS' %(username, event_name, obj)):
        args.append(obj)
        #os.chdir('sim_gen')
        #os.system('snlc_sim.exe %s > ../logs/%s &' %(obj_simfile_map[obj], obj_logfile_map[obj]))
        os.system('~/SNANA-11_05e/bin/snlc_sim.exe %s > ../logs/%s &' %(obj_simfile_map[obj], obj_logfile_map[obj]))

os.chdir('..')

#only do the following if there are running sims
if len(args) != 0:

    os.chdir('../../code')
    monitor_args = ' '.join(args)
    monitor_error = os.system('python monitor_all_sims.py %s %s' %(event_name, monitor_args))

    if monitor_error != 0:
        garbage = raw_input("ERROR in monitor_all_sims.py. Simulations are still running. Press ENTER to continue when you are sure sims have finished.")


    os.chdir('../events/%s' %event_name)
    for obj in args:
        os.system('mv ' + sim_dir_prefix + '%s  sims_and_data/%s_DESGW_%s_%s_FITS' %(obj, username, event_name, obj))
        #os.system('rm -rf ' + sim_dir_prefix + '%s' %(obj))

else:
    print("Previous sims found, moving on")
