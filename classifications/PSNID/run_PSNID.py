# A shell to collect sims and data and then trigger psnid

import datetime
import getpass
import numpy as np
import os
import pandas as pd
import random
import sys
import time

event_name = sys.argv[1]
try:
    cutoff = int(sys.argv[2])
except:
    print("second command line argument must be an integer")
    sys.exit()

signal = sys.argv[3] # single obj
background = sys.argv[4] # comma-separated list of objects
username = getpass.getuser()

print("Updating PSNID input files")
# update paths in PSNID files
modes = {'DATA': {'filename': 'psnid_DATA.nml', 'version': 'VERSION: LightCurvesReal', 'list': "   SNCID_LIST_FILE = 'good_snids_DATA.txt'"}}
modes[signal] = {'filename': 'psnid_%s.nml' %signal, 'version': 'VERSION: %s_DESGW_%s_%s' %(username, event_name, signal), 'list': "   SNCID_LIST_FILE = 'good_snids_%s.txt'" %signal}
for obj in background.split(','):
    modes[obj] = {'filename': 'psnid_%s.nml' %obj, 'version':'VERSION: %s_DESGW_%s_%s' %(username, event_name, obj), 'list': "   SNCID_LIST_FILE = 'good_snids_%s.txt'" %obj}

for obj, mode in modes.items():
    stream = open('template_psnid.nml', 'r')
    lines = stream.readlines()
    stream.close()

    outlines = []
    for line in lines:
        if line.find('PRIVATE_DATA_PATH') != -1:
            outlines.append(line.split('REPLACEME')[0] + os.getcwd() + "'\n")
        elif line[0:8] == 'VERSION:':
            outlines.append(mode['version'] + '\n')
        elif line.find('SNCID_LIST_FILE') != -1:
            outlines.append(mode['list'] + '\n')
        elif line[0:7] == 'OUTDIR:':
            outlines.append('OUTDIR:  OUTPUT_%s\n' %obj)
        else:
            outlines.append(line)

    stream = open(mode['filename'], 'w+')
    stream.writelines(outlines)
    stream.close()


#garb = raw_input("Waiting...")
print("Collecting simulations")
# collect sims
sim_include = signal + ',' + background 
os.system('python get_sims.py %s %s' %(event_name, sim_include))

#make good snid lists for sims
print("Making good snid lists")
for obj in sim_include.split(','):
    print(obj)
    good_snids = []
    d = np.load('../../events/%s/cut_results/%s_DESGW_%s_%s_cut_results.npy' %(event_name, username, event_name, obj)).item()
    for snid, info in d.items():
        if int(info['cut']) > cutoff or int(info['cut']) == -1:
            good_snids.append(int(snid))

    del d

    if len(good_snids) > 20000:
        good_snids = random.sample(good_snids, k=20000)

    stream = open('good_snids_%s.txt' %obj, 'w+')
    stream.writelines([str(x) + '\n' for x in good_snids])
    stream.close()



#garb = raw_input("Waiting...")
print("Collecting data")
# collect data
os.system('python get_candidates.py %s %i' %(event_name, cutoff))



#run psnid
for obj, mode in modes.items():
    os.system('split_and_fit.pl %s' %mode['filename'])
    print("Running PSNID on %s. Started at %s" %(obj, datetime.datetime.now().strftime('%H:%M')))

    time.sleep(60)
    done = False
    while not done:
        time.sleep(10)

        stream = open('OUTPUT_%s/MERGE.LOG' %obj, 'r')
        lines = stream.readlines()
        stream.close()

        done = len([x for x in lines if x.find('MERGED') != -1]) == 1

    time.sleep(60)
    if not os.path.exists('OUTPUT_%s/SPLIT_JOBS_LCFIT.tar.gz' %obj):
        print("Error in fitting simulations")
        sys.exit()


#garb = raw_input("Waiting...")
print("Cleaning up...")
# clean up
for k in modes.keys():
    os.system('rm psnid_%s.nml' %k)

event_dir = '../../events/%s/PSNID' %event_name
if not os.path.exists(event_dir):
    os.system('mkdir %s' %event_dir)
    os.system('mkdir %s/psnid_logs' %event_dir)

# send good snids to event directory
os.system('mv good_snids_*.txt %s' %event_dir)

# send fit results to event directory
for k in sim_include.split(','):
    os.system('mv OUTPUT_%s/%s_DESGW_%s_%s/FITOPT000.FITRES %s/%s.FITRES' %(k, username, event_name, k, event_dir, k))
os.system('mv OUTPUT_DATA/LightCurvesReal/FITOPT000.FITRES %s/DATA.FITRES' %event_dir)

# send logs to event directory
for k in sim_include.split(',') + ['DATA']:
    log_dir = '%s/psnid_logs/%s' %(event_dir, k)
    os.system('mkdir %s' %log_dir)
    os.system('mv OUTPUT_%s/SPLIT_JOBS_LCFIT.tar.gz %s' %(k, log_dir))
    os.system('mv OUTPUT_%s/*.LOG %s' %(k, log_dir))
    os.system('mv OUTPUT_%s/*.nml %s' %(k, log_dir))

# remove non-essential outputs
os.system('rm -rf OUTPUT_*')

# Remove sims and data that were copied to the local directory
for k in sim_include.split(','):
    os.system('rm %s_DESGW_%s_%s_FITS.tar.gz' %(username, event_name, k)) 

for k in sim_include.split(','):
    os.system('rm -rf %s_DESGW_%s_%s' %(username, event_name, k)) 
os.system('rm -rf LightCurvesReal')

print("Done!")
