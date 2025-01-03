# A module to trigger analysis scripts and plot production

#This scirpt should be modularized so that only desired components are run each time

import os
import sys

event_name = sys.argv[1]

# make analysis directory if necessary
if not os.path.exists("../events/%s/BBH_analysis/" %event_name):
    os.mkdir("../events/%s/BBH_analysis/" %event_name)


# merge BBH model properties with cut results and peak magnitudes
os.system("python BBH_properties.py %s" %event_name)

# make BBH model efficiency plots
os.system("python BBH_template_efficiencies.py %s" %event_name)

sys.exit()

# extract light curves from data
os.system('python extract_lc_points.py %s data' %event_name)

#sys.exit()
# extract light cuvres from sims
os.system('python extract_lc_points.py %s sims' %event_name)

# find mean light curves

# find mean KN light curves at 10% detection limit

# plot mean light curves
