# A module to make all training datasets

import glob
import numpy as np
import os
import pandas as pd
import sys
import time

event_name = sys.argv[1]
sim_include = sys.argv[2]

log_dir = '../../events/%s/logs/' %event_name

#sim_include='KN,Ia,CC,AGN,CaRT,ILOT,Mdwarf,SN91bg,Iax,PIa,SLSN,TDE'

objs = sim_include.split(',')

# start all dataset jobs
for obj in objs:
    os.system('python make_dataset.py %s %s > %sKNC_%s_prog.log &' %(event_name, obj, log_dir, obj))

# wait for all to finish
running = True
while running:
    dataset_files = glob.glob('../../events/%s/KNC/*_datasets.npy' %event_name)
    if len(dataset_files) != len(objs):
        sys.stdout.write('\rExtracting: %s' %' '.join([x for x in objs if ','.join(dataset_files).find(x) == -1]))
        sys.stdout.flush()
        time.sleep(4)
    else:
        running = False

# organize resulting datasets by light curve properties
sys.stdout.write('\rExtracting: Done!  --  Reading Features: Waiting')
sys.stdout.flush()
datasets = {obj: np.load('../../events/%s/KNC/%s_datasets.npy' %(event_name, obj)).item() for obj in objs}

sys.stdout.write('\rExtracting: Done!  --  Reading Features: Done!  --  Organizing Datasets: Waiting')
sys.stdout.flush()
grouped_datasets = {}
for obj, dataset in datasets.items():
    for label, df in dataset.items():
        if label in grouped_datasets.keys():
            grouped_datasets[label].append(df)
        else:
            grouped_datasets[label] = [df]

merged_datasets = {label: pd.concat(data) for label, data in grouped_datasets.items()}
np.save('../../events/%s/KNC/training_data.npy' %event_name, merged_datasets)

# finish
sys.stdout.write('\rExtracting: Done!  --  Reading Features: Done!  --  Organizing Datasets: Done!    \n')
sys.stdout.flush()
