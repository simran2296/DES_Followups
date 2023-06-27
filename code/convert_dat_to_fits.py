# A wrapper for the snana fits conversion feature

import os
import sys



event_name = sys.argv[1]
if not os.path.exists('../events/%s/sims_and_data/' %event_name):
    os.system('mkdir sims_and_data')

fits_path = '../events/%s/sims_and_data/LightCurvesReal_FITS' % event_name
if os.path.exists(fits_path):
    os.system('rm -r %s' %fits_path)

os.system('cp -r ../events/%s/sims_and_data/LightCurvesReal .' %event_name)

log_path = '../events/%s/logs' %event_name
if not os.path.exists(log_path):
    os.system('mkdir %s' %log_path)


#check for list, readme, and ignore files
os.system('rm LightCurvesReal/*.IGNORE LightCurvesReal/*.README LightCurvesReal/*.LIST')
os.chdir('LightCurvesReal')
os.system('ls -1 *.dat > LightCurvesReal.LIST')
os.system('touch LightCurvesReal.IGNORE')
os.system('touch LightCurvesReal.README')
readme = open("LightCurvesReal.README",'w')
docana_lines = '''DOCUMENTATION:
    PURPOSE:  BBH SNANA simulation
    INTENT:   Nominal
    USAGE_KEY:  SIMLIB_FILE
    USAGE_CODE: snlc_sim.exe
    VALIDATE_SCIENCE: included in JLA & Pantheon cosmology analyses (maybe)
    VERSIONS:
    - DATE:  2012-03-12
      AUTHORS: R. Kessler (and me?)
DOCUMENTATION_END:

'''
readme.write(docana_lines)
readme.close()

os.chdir('..')

dat_path = 'LightCurvesReal'
dat_path_but_cooler = dat_path + '_1'

os.system('~/SNANA-11_04o/bin/snana.exe NOFILE PRIVATE_DATA_PATH %s VERSION_PHOTOMETRY %s VERSION_REFORMAT_FITS %s > %s/convert_dat_to_fits.log' %(dat_path, dat_path, dat_path_but_cooler, log_path))

os.system('rm -r LightCurvesReal')
os.system('mkdir FITS_LightCurvesReal')
os.system('mv LightCurvesReal_1/* FITS_LightCurvesReal')
os.system('mv FITS_LightCurvesReal/LightCurvesReal_1.IGNORE FITS_LightCurvesReal/LightCurvesReal.IGNORE')
os.system('mv FITS_LightCurvesReal/LightCurvesReal_1.README FITS_LightCurvesReal/LightCurvesReal.README')
os.system('mv FITS_LightCurvesReal/LightCurvesReal_1.LIST FITS_LightCurvesReal/LightCurvesReal.LIST')

os.system('gunzip FITS_LightCurvesReal/LightCurvesReal_1_HEAD.FITS.gz')
os.system('gunzip FITS_LightCurvesReal/LightCurvesReal_1_PHOT.FITS.gz')


os.system('mv FITS_LightCurvesReal/LightCurvesReal_1_HEAD.FITS FITS_LightCurvesReal/LightCurvesReal_HEAD.FITS')
os.system('mv FITS_LightCurvesReal/LightCurvesReal_1_PHOT.FITS FITS_LightCurvesReal/LightCurvesReal_PHOT.FITS')
os.system('mv FITS_LightCurvesReal ../events/%s/sims_and_data/LightCurvesReal_FITS' %event_name)
os.system('rm -r LightCurvesReal_1')
