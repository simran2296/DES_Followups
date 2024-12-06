# A module to perform ML classification

import numpy as np
import os
import sys

from classify import classify

event_name = sys.argv[1]
knc_dir = '../../events/%s/KNC/' %event_name

# Load training data
training_datasets = np.load(knc_dir + 'training_data.npy' %event_name).item()

# Load real data
real_datasets = np.load(knc_dir + 'DATA_datasets.npy' %event_name).item()

# Track output files
results_dir = knc_dir + 'DATA_results/' 
if not os.path.exists(results_dir):
    os.system('mkdir %s' %results_dir)
id_num, id_map = 10000, {}

counter, total = 0.0, float(len(list(real_datasets.keys())))
# Begin classifications
for code, test_df in real_datasets.items():
    # Track progress
    counter += 1.0
    progress = counter / total * 100
    sys.stdout.write('\rClassifying:  %.1f %%   ' %progress)
    sys.stdout.flush()

    # Target matched training data
    train_df = training_datasets[code]

    # Track results
    test_df_outfile = results_dir + 'kncres_%i.csv' %id_num
    report_outfile = results_dir + 'report_%i.csv' %id_num
    id_map[id_num] = code
    id_num += 1

    # Classify data
    classify(train_df, [test_df], [test_df_outfile], [report_outfile], n_jobs=-1, output_auc=False)

# Write the ID map to a file
np.save(results_dir + 'id_map.npy', id_map)
