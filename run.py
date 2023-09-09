## Outer Shell for Follow-up Pipeline

### This shell will both generate sims and place cuts on observations


import os
import pandas as pd
import sys

from optparse import OptionParser

# move to code directory
os.chdir('code')

header =  """
_________________________________________________________________________________
    ____  _____ ____    _____     _ _                                            
   |  _ \| ____/ ___|  |  ___|__ | | | _____      __          _   _ _ __         
   | | | |  _| \___ \  | |_ / _ \| | |/ _ \ \ /\ / /  _____  | | | | '_ \        
   | |_| | |___ ___) | |  _| (_) | | | (_) \ V  V /  |_____| | |_| | |_) |       
   |____/|_____|____/  |_|  \___/|_|_|\___/ \_/\_/            \__,_| .__/        
                                                                   |_|           
       _                _           _       ____  _            _ _                 
      / \   _ __   __ _| |_   _ ___(_)___  |  _ \(_)_ __   ___| (_)_ __   ___    
     / _ \ | '_ \ / _` | | | | / __| / __| | |_) | | '_ \ / _ \ | | '_ \ / _ \   
    / ___ \| | | | (_| | | |_| \__ \ \__ \ |  __/| | |_) |  __/ | | | | |  __/   
   /_/   \_\_| |_|\__,_|_|\__, |___/_|___/ |_|   |_| .__/ \___|_|_|_| |_|\___|   
                          |___/                    |_|                           
---------------------------------------------------------------------------------
"""

print('\n\n\n')
print(header)
print('\n')

#Determine mode of operation based on presence of event_name
event_name = sys.argv[1]
if not os.path.exists('../events/%s' %event_name):
    mode = 'interactive'
else:
    mode = 'normal'

#parse for bosst, num_kn, and num_agn
parser = OptionParser(__doc__)
parser.add_option('--boost', default='', help="Number of objects")
parser.add_option('--clean_up', default="None", help="Things to be cleaned up")
parser.add_option('--sim_include', default="KN,Ia,CC,AGN", help="Transient classes to simulate")
parser.add_option('--force_conditions', default='real', help="PSF,SKYMAG,DELTAT")
parser.add_option('--make_cuts', action='store_true', help='auto-generate detectability cuts from DB')
parser.add_option('--clobber', action='store_true')
parser.add_option('--nobs_force', default='2', help='Number of obs to require in libids')
parser.add_option('--run_psnid', action='store_true')
parser.add_option('--signal', default='None', help='Object class to use as signal in ML')
parser.add_option('--background', default='None', help='Object class list (comma-separated) to use as background in ML')
parser.add_option('--classify_cutoff', default=100, help='Final cut to apply before classification')
options, args = parser.parse_args(sys.argv[2:])
boost = options.boost
clean_up = str(options.clean_up)
sim_include = str(options.sim_include).strip().split(',')
forced_conditions = str(options.force_conditions)
make_cuts = options.make_cuts
clobber = options.clobber
nobs_force = options.nobs_force
run_psnid = options.run_psnid
classify_cutoff = int(options.classify_cutoff)
signal = options.signal
background = options.background

# format boost if a single number is specified for all classes
if len(boost.split(',')) == 1:
    boost = ','.join([boost] * len(sim_include))

# Command-line argument error checking
allowed_sims = ['KN', 'Ia', 'CC', 'AGN', 'CaRT', 'ILOT', 'Mdwarf', 'SN91bg', 'Iax', 'PIa', 'SLSN', 'TDE', 'BBH']
allowed_sims += [x + '-tr' for x in allowed_sims] #let any object be triggered
if len(sim_include) == 1:
    if sim_include[0] == 'all':
        sim_include = allowed_sims[:]
else:
    for obj in sim_include:
        if obj not in allowed_sims:
            print("%s is not one of the available models. Choose from the following list:" %obj)
            print(allowed_sims)
            sys.exit()

assert len(boost.split(',')) == len(sim_include), "Boosts must map 1-1 with sims"

if run_psnid:
    if signal == 'None' or background == 'None':
        print("You must specify both the --signal and the --background to use the --run_psnid argument")
        sys.exit()
    if signal not in sim_include:
        print("--signal must be one of the --sim_include objects")
        sys.exit()
    for obj in background.split(','):
        if obj not in sim_include:
            print("%s is not in --sim_include so it cannot be part of --background" %obj)
            sys.exit()

# Protect against triggered Ia sims
if 'Ia-tr' in sim_include:
    print("WARNING: Triggered SNe-Ia simulations are not supported by SNANA")
    idx = sim_include.index('Ia-tr')
    garb = sim_include.pop(idx)
    boost_list = boost.split(',')
    garb = boost_list.pop(idx)
    boost = ','.join(boost_list)
    

if mode == 'interactive':
    print("Event %s not found. Do you want to run in interactive mode?\n" %event_name)
    answer = raw_input('(y/n)')
    while answer.lower() not in ['y', 'n']:
        answer = raw_input("Please enter y or n: ")

    if answer == 'n':
        print("Exiting...")
        sys.exit()

    #prompt the user for all needed values and create necessary files
    err_code = os.system('python interactive_setup.py %s' %event_name)
    if err_code:
        sys.exit(err_code)

    #generate the cuts for the user.
    make_cuts = True

print("Running in normal mode.\n")

#clobber all existing runs if desired
if clobber:
    clean_up = 'everything'

#clean up if desired
if clean_up != "None":
    os.system("python clean_up.py %s %s" %(event_name, clean_up))

#interpret metadata
os.system('python interpret_metadata.py %s --boost %s --sim_include %s' %(event_name, boost, ','.join(sim_include)))

#optionally set up detectability cuts
if make_cuts:
    os.system("python setup_cuts.py %s" % event_name)

#generate sims
os.system('python generate_all_sims.py %s %s %s %s' %(event_name, forced_conditions, nobs_force, ' '.join(sim_include)))

#apply cuts to sims and data
os.system('python run_all_cuts.py %s both %s' %(event_name, ' '.join(sim_include)))

#analyze results
os.system('python analyze_results.py %s %s' %(event_name, ','.join(sim_include)))

#write out a report with uncertainties
os.system('python write_report.py %s %s terse' %(event_name, ','.join(sim_include)))

#classify with psnid if desired
if run_psnid:
    os.chdir('../classifications/PSNID')
    os.system('python run_PSNID.py %s %i %s %s' %(event_name, classify_cutoff, signal, background))
    os.system('python featurize.py %s %s %s' %(event_name, signal, background))
    os.system('python classify.py %s %s %s' %(event_name, signal, background))

#create parameter plots
os.system('python run_analysis.py %s' %(event_name))

#create contour plot
os.system('python make_kn_contour_plot.py %s' %(event_name))
