# update snana templates to fit event of interest

import getpass
import os
import pandas as pd
import sys
import utils

import force_ligo_dist

#read input data
event_name = sys.argv[1]
df = pd.read_csv('../events/%s/event_metadata.csv' %event_name)
username = getpass.getuser()
forced_conditions = sys.argv[2]

#insert forced conditions into simlib
if forced_conditions != 'real':
    forced_values = ' '.join(forced_conditions.split(','))
    os.system('python force_conditions.py %s %s' %(event_name, forced_values))

#get effective area from simlib
simlib_filename = '../events/%s/sim_gen/SIMLIB.txt' %event_name
simlib_file = open(simlib_filename, 'r')
simlib_info = simlib_file.readlines()
simlib_file.close()
eff_area = float([x for x in simlib_info[-5:] if x[0:15] == 'EFFECTIVE_AREA:'][0].split(' ')[1][0:-1])

docana_lines = ['DOCUMENTATION:',
'    PURPOSE:  BBH SNANA simulation',
'    INTENT:   Nominal',
 '   USAGE_KEY:  SIMLIB_FILE',
'    USAGE_CODE: snlc_sim.exe',
'    VALIDATE_SCIENCE: included in JLA & Pantheon cosmology analyses (maybe)',
'    VERSIONS:',
'    - DATE:  2012-03-12',
'    AUTHORS: R. Kessler (and me?)',
'DOCUMENTATION_END:', '']

file_prefix = '../events/%s/sim_gen/' %event_name
file_list = ['AGN_SIMGEN.INPUT', 'SIMGEN_DES_KN.input', 'SIMGEN_DES_NONIA.input', 'SIMGEN_DES_SALT2.input',
             'CART_SIMGEN.INPUT', 'ILOT_SIMGEN.INPUT', 'Mdwarf_SIMGEN.INPUT', 'SN_Ia91bgSIMGEN.INPUT',
             'SN_IaxSIMGEN.INPUT', 'SN_PIaSIMGEN.INPUT', 'SN_SLSNSIMGEN.INPUT', 'TDE_SIMGEN.INPUT', 'SIMGEN_DES_BBH.input'] 
objs = ['AGN', 'KN', 'CC', 'Ia', 'CaRT', 'ILOT', 'Mdwarf', 'SN91bg', 'Iax', 'PIa', 'SLSN', 'TDE', 'BBH']

# Add trigger files
boosts = {x[7:]: df[x].values[0] for x in df.columns if x.find('BOOST') != -1}
trigger_files, trigger_objs = [], []
for k, v in boosts.items():
    if k.find('-tr') != -1:
        trigger_files.append(file_list[objs.index(k[0:-3])])
        trigger_objs.append(k)

file_list += trigger_files
objs += trigger_objs

#iterate through files and update:
# - genversion (update during loop)
# - gnerange_peakmjd
# - genrange_mjd
# - solid_angle (conversion: .00082 sterradians = 3 sqdeg )
# - mjd explode (kn only)

mjd_min = str(int(float(df['MJD_EXPLODE'].values[0])) - 60)
mjd_max = str(int(float(df['MJD_EXPLODE'].values[0])) + 30)
mjd_exp = str(df['MJD_EXPLODE'].values[0])
solid_angle = str(round(eff_area / 3.0 * .00082, 6))
if 'LIGO_distance_(Mpc)' in df.columns:
    min_z, max_z = utils.get_ligo_z_range(df['LIGO_distance_(Mpc)'].values[0], df['LIGO_sigma_(Mpc)'].values[0]) 
else:
    min_z = None
    max_z = None


for filename, obj in zip(file_list, objs):

    if obj not in boosts.keys():
        # Only edit files that requested simulations
        continue
    
    #construct new input information
    header = ['#',
              '# SIMGEN file for ' + obj,
              '#',
              ' ',
              '#--------------------------------------------------------------------',
              '# Event-specific info to be overwritten',
              '#--------------------------------------------------------------------',
              ' ',
              'GENRANGE_PEAKMJD: ' + mjd_min + ' ' + mjd_max,
              'GENRANGE_MJD: ' + mjd_min + ' ' + mjd_max,
              'GENVERSION: %s_DESGW_' %username  + event_name + '_' + obj,
              'SOLID_ANGLE: ' + solid_angle,
              'NGEN_LC: ' + str(boosts[obj])]

    
    # LIGO redshift bounds
    if obj in ['KN', 'BBH', 'KN-tr', 'BBH-tr']:
        header.append('GENRANGE_REDSHIFT: ' + str(min_z) + ' ' + str(max_z))

    # Triggers
    if obj.find('-tr') != -1:
        header.append('MJD_EXPLODE: ' + mjd_exp)

    buffer_lines = ['#--------------------------------------------------------------------',
                    "# Don't need to change anything after this point  (I sure hope)",
                    '#--------------------------------------------------------------------']

    input_lines = [x + '\n' for x in docana_lines + header + buffer_lines]

    #open template file
    template_file = open(file_prefix + filename, 'r')
    template_lines = template_file.readlines()
    template_file.close()
    
    #construct output file
    output_lines_ = input_lines + template_lines[len(input_lines) + 1 - len(docana_lines):]

    # final check to remove any duplicated lines (keep the first instance)
    output_lines = []
    output_keys = []
    for line in output_lines_:
        line_info = line.split(':')
        if len(line_info) == 1:
            output_lines.append(line)
        else:
            key = line_info[0]
            if key not in output_keys:
                output_lines.append(line)
                output_keys.append(key)

    #write to output file
    if obj.find('-tr') == -1:
        out_filename = filename
    else:
        out_filename = 'TR_' + filename
 
    outfile = open(file_prefix + out_filename, 'w+')
    outfile.writelines(output_lines)
    outfile.close()


# do LIGO specific distribution
if 'LIGO_distance_(Mpc)' in df.columns and ('KN-tr' in trigger_objs or 'BBH-tr' in trigger_objs):
    # Run on triggered input files
    for obj_filename in ['TR_SIMGEN_DES_KN.input', 'TR_SIMGEN_DES_BBH.input']:
        try:
            force_ligo_dist.run(obj_filename, event_name, df['LIGO_distance_(Mpc)'].values[0], df['LIGO_sigma_(Mpc)'].values[0])
        except IOError:
            # triggered version was not requested
            continue

#File header looks like this:
"""
# 
# SIMGEN file for KN
#

#--------------------------------------------------------------------
# Event-specific info to be overwritten
#--------------------------------------------------------------------

GENRANGE_PEAKMJD: 58598 58628
GENRANGE_MJD: 58598 58628
GENVERSION:  RM_Net_AGN
EXPOSURE_TIME: 1.5
SOLID_ANGLE:  0.0082
MJD_EXPLODE: 55555

#--------------------------------------------------------------------
# Don't need to change anything after this point  (I sure hope)
#--------------------------------------------------------------------
"""
