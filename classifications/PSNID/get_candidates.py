# collect dat files from list of snids

import os
import sys

#if os.path.exists('LightCurvesReal/'):
#    os.system('rm -rf LightCurvesReal')
os.system('mkdir LightCurvesReal')

event_name = sys.argv[1]
try:
    cutoff = int(sys.argv[2])
except:
    print("second command line argument must be an integer")
    sys.exit()

### collect snids for everything passing cut 3 
print("Importing modules")
import numpy as np
import pandas as pd

print("Loading cut results")
d = np.load('../../events/%s/cut_results/LightCurvesReal_cut_results.npy' %event_name).item()
snids = []
for snid, info in d.iteritems():
    cutnum = int(info['cut'])
    if cutnum > cutoff or cutnum < 0:
        snids.append(snid)

#save good snids
stream = open('good_snids_DATA.txt', 'w+')
stream.writelines([str(x) + '\n' for x in snids])
stream.close()

dat_files = ['des_real_00%s.dat' % str(x) for x in snids]

stream = open('../../events/%s/dat_file_path.txt' %event_name, 'r')
path = stream.readlines()[0].strip()
if path[-1] != '/': path += '/'
stream.close()


total = len(dat_files)
counter = 0.0

for dat_file in dat_files:
    counter += 1.0
    progress = counter / total * 100.0
    sys.stdout.write('\rCopying files:  %.2f %%    ' %progress)
    sys.stdout.flush()
    os.system('cp %s LightCurvesReal' %(path + dat_file))

#make necessary psnid files
os.chdir('LightCurvesReal')
os.system('ls -1 *.dat > LightCurvesReal.LIST')
os.system('touch LightCurvesReal.IGNORE')
os.system('touch LightCurvesReal.README')
os.chdir('..')

#run snana text to fits
dat_path = 'LightCurvesReal'
out_dir = 'LightCurvesReal_out'
#os.system('rm -rf LightCurvesReal')
os.system('snana.exe NOFILE PRIVATE_DATA_PATH %s VERSION_PHOTOMETRY %s VERSION_REFORMAT_FITS %s' %(dat_path, dat_path, out_dir))
#os.system('rm -rf LightCurvesReal')                                                                                                                                 
#clean up files for PSNID
os.system('rm LightCurvesReal/*')
os.chdir('LightCurvesReal_out')
os.system('gzip -d LightCurvesReal_out_HEAD.FITS.gz')
os.system('gzip -d LightCurvesReal_out_PHOT.FITS.gz')
os.system('mv LightCurvesReal_out_HEAD.FITS LightCurvesReal')
os.system('mv LightCurvesReal_out.IGNORE LightCurvesReal')
os.system('mv LightCurvesReal_out.LIST LightCurvesReal')
os.system('mv LightCurvesReal_out_PHOT.FITS LightCurvesReal')
os.system('mv LightCurvesReal_out.README LightCurvesReal')
