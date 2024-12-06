# A module to run cross classificaitons on different training sets

import numpy as np
import pandas as pd
import sys

from classify import classify

event_name = sys.argv[1]
mode = sys.argv[2]
assert mode in ['good', 'fair', 'poor', 'all']

test_param = sys.argv[3]
assert test_param in ['seeing', 'skymag', 'deltat', 'all']

try:
    skip = int(sys.argv[4])
except:
    skip = 0

# Load allowable values
allowed_conditions = np.load('allowed_conditions.npy').item()
seeing_values = allowed_conditions['SEEING']
skymag_values = allowed_conditions['SKYMAG']
deltat_values = allowed_conditions['DELTAT']
del allowed_conditions

# Read index file into dict
df = pd.read_csv('../../events/%s/KNC_validation/conditions_map.txt' %event_name, delim_whitespace=True)
df = df.apply(pd.to_numeric)
condition_dict = {}
for index, row in df.iterrows():
    r = row.copy()
    condition_dict[r['INDEX']] = {'SEEING': r['SEEING'],
                                  'SKYMAG': r['SKYMAG'],
                                  'DELTAT': r['DELTAT']}


# reverse the dictionary for quick lookup
index_dict = {'%.2f_%.2f_%.2f' %(float(info['SEEING']), float(info['SKYMAG']), float(info['DELTAT'])): index for index, info in condition_dict.items()}


#####
# SEEING
#####
def run_seeing(mode, fixed_seeing, fixed_skymag, fixed_deltat, event_name=event_name, index_dict=index_dict, seeing_values=seeing_values, skymag_values=skymag_values, deltat_values=deltat_values, force_skip=0):
    assert mode != 'all'
    
    # Variable seeing, fixed skymag and deltat
    if force_skip == 0:
        #only open new file if cold start... The script will automatically append in the warm start case
        seeing_outfile = open('../../events/%s/KNC_validation/seeing_%s_results.txt' %(event_name, mode), 'w+')
        seeing_outfile.write('TRAINING\tTESTING\tSCORE\n')
        seeing_outfile.close()


    skip_counter = 0
    for train_seeing in seeing_values:
        # Allow for a warm start
        if skip_counter < force_skip:
            skip_counter += 1
            continue

        print("%s\tLoading training data" %train_seeing)
        #load training set corresponding to conditions
        train_cond_string = '%.2f' %float(train_seeing) + '_' + fixed_skymag + '_' + fixed_deltat
        cond_index = int(index_dict[train_cond_string])
        training_set = np.load('../../events/%s/KNC_validation/datasets/ts_%s.npy' %(event_name, cond_index)).item()

        #load all test sets into memory that we will classify based on this training set
        print("%s\tLoading testing data" %train_seeing)
        test_sets = {}
        for test_seeing in seeing_values:
            cond_string = '%.2f' %float(test_seeing) + '_' + fixed_skymag + '_' + fixed_deltat
            cond_index = int(index_dict[cond_string])
            test_sets['%.2f' %float(test_seeing)] = np.load('../../events/%s/KNC_validation/datasets/ts_%s.npy' %(event_name, cond_index)).item()

        # For each training set, collect and classify like test sets
        all_auc_scores, all_samples, all_test_cond_strings = [], [], []
        
        total = len(list(training_set.keys()))
        counter = 0
        for code, train_df in training_set.items():
            #Track progress
            counter += 1
            sys.stdout.write('\rClassifying  %i / %i     ' %(counter, total))
            sys.stdout.flush()

            #collect like test sets
            test_dfs, outfiles, report_files, test_cond_strings = [], [], [], []
            for test_seeing in test_sets.keys():
                test_cond_string = test_seeing + '_' + fixed_skymag + '_' + fixed_deltat
                #poor observing conditions may lead to all light curves for a given code being cut out
                try:
                    test_dfs.append(test_sets[test_seeing][code])
                except:
                    #put in placeholder df and flag row
                    test_dfs.append(train_df)
                    test_cond_string += 'BAD'
                    
                test_cond_strings.append(test_cond_string)
                outfiles.append('../../events/%s/KNC_validation/garb.csv' %event_name)
                report_files.append('../../events/%s/KNC_validation/garb.txt' %event_name)

            auc_scores, samples = classify(train_df, test_dfs, outfiles, report_files, n_jobs=-1, output_auc=True)
            all_auc_scores.append(auc_scores)
            all_samples.append(samples)
            all_test_cond_strings.append(test_cond_strings)

        print("Collecting results")
        # Restructure lists and load into dataframe
        data = []
        for ii in range(len(all_test_cond_strings)):
            if len(all_auc_scores[ii]) == len(all_test_cond_strings[ii]):
                for jj in range(len(all_test_cond_strings[ii])):
                    data.append([all_test_cond_strings[ii][jj], all_auc_scores[ii][jj], all_samples[ii][jj]])
        #filter out rows flagged as bad (hopefully they are already gone)
        data = [x for x in data if x[0].find('BAD') == -1]
        out_df = pd.DataFrame(data=data, columns=['TEST_COND', 'AUC', 'SAMPLES'])

        print(out_df)

        # Reduce to final scores and save
        for test_cond in np.unique(out_df['TEST_COND'].values):
            partial_df = out_df[out_df['TEST_COND'].values == test_cond]
            score = np.dot(partial_df['AUC'].values, partial_df['SAMPLES'].values) / np.sum(partial_df['SAMPLES'].values)
            
            seeing_outfile = open('../../events/%s/KNC_validation/seeing_%s_results.txt' %(event_name, mode), 'a')
            seeing_outfile.write('%s\t%s\t%s\n' %(train_cond_string, test_cond, score))
            seeing_outfile.close()

    return

#####
# SKYMAG
#####
def run_skymag(mode, fixed_seeing, fixed_skymag, fixed_deltat, event_name=event_name, index_dict=index_dict, seeing_values=seeing_values, skymag_values=skymag_values, deltat_values=deltat_values, force_skip=0):
    assert mode != 'all'
    
    # Variable seeing, fixed skymag and deltat
    if force_skip == 0:
        #only open new file if cold start... The script will automatically append in the warm start case
        skymag_outfile = open('../../events/%s/KNC_validation/skymag_%s_results.txt' %(event_name, mode), 'w+')
        skymag_outfile.write('TRAINING\tTESTING\tSCORE\n')
        skymag_outfile.close()


    skip_counter = 0
    for train_skymag in skymag_values:
        # Allow for a warm start
        if skip_counter < force_skip:
            skip_counter += 1
            continue

        print("%s\tLoading training data" %train_skymag)
        #load training set corresponding to conditions
        train_cond_string = fixed_seeing + '_%.2f' %float(train_skymag) + '_' + fixed_deltat
        cond_index = int(index_dict[train_cond_string])
        training_set = np.load('../../events/%s/KNC_validation/datasets/ts_%s.npy' %(event_name, cond_index)).item()

        #load all test sets into memory that we will classify based on this training set
        print("%s\tLoading testing data" %train_skymag)
        test_sets = {}
        for test_skymag in skymag_values:
            cond_string = fixed_seeing + '_%.2f' %float(test_skymag) + '_' + fixed_deltat
            cond_index = int(index_dict[cond_string])
            test_sets['%.2f' %float(test_skymag)] = np.load('../../events/%s/KNC_validation/datasets/ts_%s.npy' %(event_name, cond_index)).item()

        # For each training set, collect and classify like test sets
        all_auc_scores, all_samples, all_test_cond_strings = [], [], []
        
        total = len(list(training_set.keys()))
        counter = 0
        for code, train_df in training_set.items():
            #Track progress
            counter += 1
            sys.stdout.write('\rClassifying  %i / %i     ' %(counter, total))
            sys.stdout.flush()

            #collect like test sets
            test_dfs, outfiles, report_files, test_cond_strings = [], [], [], []
            for test_skymag in test_sets.keys():
                test_cond_string = fixed_seeing + '_' + test_skymag + '_' + fixed_deltat
                #poor observing conditions may lead to all light curves for a given code being cut out
                try:
                    test_dfs.append(test_sets[test_skymag][code])
                except:
                    #put in placeholder df and flag row
                    test_dfs.append(train_df)
                    test_cond_string += 'BAD'
                    
                test_cond_strings.append(test_cond_string)
                outfiles.append('../../events/%s/KNC_validation/garb.csv' %event_name)
                report_files.append('../../events/%s/KNC_validation/garb.txt' %event_name)

            auc_scores, samples = classify(train_df, test_dfs, outfiles, report_files, n_jobs=-1, output_auc=True)
            all_auc_scores.append(auc_scores)
            all_samples.append(samples)
            all_test_cond_strings.append(test_cond_strings)

        print("Collecting results")
        # Restructure lists and load into dataframe
        data = []
        for ii in range(len(all_test_cond_strings)):
            if len(all_auc_scores[ii]) == len(all_test_cond_strings[ii]):
                for jj in range(len(all_test_cond_strings[ii])):
                    data.append([all_test_cond_strings[ii][jj], all_auc_scores[ii][jj], all_samples[ii][jj]])
        #filter out rows flagged as bad (hopefully they are already gone)
        data = [x for x in data if x[0].find('BAD') == -1]
        out_df = pd.DataFrame(data=data, columns=['TEST_COND', 'AUC', 'SAMPLES'])

        print(out_df)

        # Reduce to final scores and save
        for test_cond in np.unique(out_df['TEST_COND'].values):
            partial_df = out_df[out_df['TEST_COND'].values == test_cond]
            score = np.dot(partial_df['AUC'].values, partial_df['SAMPLES'].values) / np.sum(partial_df['SAMPLES'].values)
            
            skymag_outfile = open('../../events/%s/KNC_validation/skymag_%s_results.txt' %(event_name, mode), 'a')
            skymag_outfile.write('%s\t%s\t%s\n' %(train_cond_string, test_cond, score))
            skymag_outfile.close()

    return


#####
# DELTAT
#####
def run_deltat(mode, fixed_seeing, fixed_skymag, fixed_deltat, event_name=event_name, index_dict=index_dict, seeing_values=seeing_values, skymag_values=skymag_values, deltat_values=deltat_values, force_skip=0):
    assert mode != 'all'
    
    # Variable seeing, fixed skymag and deltat
    if force_skip == 0:
        #only open new file if cold start... The script will automatically append in the warm start case
        deltat_outfile = open('../../events/%s/KNC_validation/deltat_%s_results.txt' %(event_name, mode), 'w+')
        deltat_outfile.write('TRAINING\tTESTING\tSCORE\n')
        deltat_outfile.close()


    skip_counter = 0
    for train_deltat in deltat_values:
        # Allow for a warm start
        if skip_counter < force_skip:
            skip_counter += 1
            continue

        print("%s\tLoading training data" %train_deltat)
        #load training set corresponding to conditions
        train_cond_string = fixed_seeing + '_' + fixed_skymag + '_%.2f' %float(train_deltat)
        cond_index = int(index_dict[train_cond_string])
        training_set = np.load('../../events/%s/KNC_validation/datasets/ts_%s.npy' %(event_name, cond_index)).item()

        #load all test sets into memory that we will classify based on this training set
        print("%s\tLoading testing data" %train_deltat)
        test_sets = {}
        for test_deltat in deltat_values:
            cond_string = fixed_seeing + '_' + fixed_skymag + '_%.2f' %float(test_deltat)
            cond_index = int(index_dict[cond_string])
            test_sets['%.2f' %float(test_deltat)] = np.load('../../events/%s/KNC_validation/datasets/ts_%s.npy' %(event_name, cond_index)).item()

        # For each training set, collect and classify like test sets
        all_auc_scores, all_samples, all_test_cond_strings = [], [], []
        
        total = len(list(training_set.keys()))
        counter = 0
        for code, train_df in training_set.iteritems():
            #Track progress
            counter += 1
            sys.stdout.write('\rClassifying  %i / %i     ' %(counter, total))
            sys.stdout.flush()

            #collect like test sets
            test_dfs, outfiles, report_files, test_cond_strings = [], [], [], []
            for test_deltat in test_sets.keys():
                test_cond_string = fixed_seeing + '_' + fixed_skymag + '_' + test_deltat
                #poor observing conditions may lead to all light curves for a given code being cut out
                try:
                    test_dfs.append(test_sets[test_deltat][code])
                except:
                    #put in placeholder df and flag row
                    test_dfs.append(train_df)
                    test_cond_string += 'BAD'
                    
                test_cond_strings.append(test_cond_string)
                outfiles.append('../../events/%s/KNC_validation/garb.csv' %event_name)
                report_files.append('../../events/%s/KNC_validation/garb.txt' %event_name)

            auc_scores, samples = classify(train_df, test_dfs, outfiles, report_files, n_jobs=-1, output_auc=True)
            all_auc_scores.append(auc_scores)
            all_samples.append(samples)
            all_test_cond_strings.append(test_cond_strings)

        print("Collecting results")
        # Restructure lists and load into dataframe
        data = []
        for ii in range(len(all_test_cond_strings)):
            if len(all_auc_scores[ii]) == len(all_test_cond_strings[ii]):
                for jj in range(len(all_test_cond_strings[ii])):
                    data.append([all_test_cond_strings[ii][jj], all_auc_scores[ii][jj], all_samples[ii][jj]])
        #filter out rows flagged as bad (hopefully they are already gone)
        data = [x for x in data if x[0].find('BAD') == -1]
        out_df = pd.DataFrame(data=data, columns=['TEST_COND', 'AUC', 'SAMPLES'])

        print(out_df)

        # Reduce to final scores and save
        for test_cond in np.unique(out_df['TEST_COND'].values):
            partial_df = out_df[out_df['TEST_COND'].values == test_cond]
            score = np.dot(partial_df['AUC'].values, partial_df['SAMPLES'].values) / np.sum(partial_df['SAMPLES'].values)
            
            deltat_outfile = open('../../events/%s/KNC_validation/deltat_%s_results.txt' %(event_name, mode), 'a')
            deltat_outfile.write('%s\t%s\t%s\n' %(train_cond_string, test_cond, score))
            deltat_outfile.close()

    return


#####
# Main condition handling
#####

if mode in ['good', 'all']:

    good_seeing = '%.2f' %np.min(df['SEEING'].values)
    good_skymag = '%.2f' %np.max(df['SKYMAG'].values)
    good_deltat = '%.2f' %np.min(df['DELTAT'].values)

    if test_param in ['seeing', 'all']:
        run_seeing('good', good_seeing, good_skymag, good_deltat, force_skip=skip)

    if test_param in ['skymag', 'all']:
        run_skymag('good', good_seeing, good_skymag, good_deltat, force_skip=skip)

    if test_param in ['deltat', 'all']:
        run_deltat('good', good_seeing, good_skymag, good_deltat, force_skip=skip)

if mode in ['fair', 'all']:

    fair_seeing = '%.2f' %sorted(df['SEEING'].values)[int(float(df.shape[0]) / 2)]
    fair_skymag = '%.2f' %sorted(df['SKYMAG'].values)[int(float(df.shape[0]) / 2)]
    fair_deltat = '%.2f' %sorted(df['DELTAT'].values)[int(float(df.shape[0]) / 2)]

    if test_param in ['seeing','all']:
        run_seeing('fair', fair_seeing, fair_skymag, fair_deltat, force_skip=skip)

    if test_param in ['skymag','all']:
        run_skymag('fair', fair_seeing, fair_skymag, fair_deltat, force_skip=skip)

    if test_param in ['deltat','all']:
        run_deltat('fair', fair_seeing, fair_skymag, fair_deltat, force_skip=skip)


if mode in ['poor', 'all']:

    poor_seeing = '%.2f' %np.max(df['SEEING'].values)
    poor_skymag = '%.2f' %np.min(df['SKYMAG'].values)
    poor_deltat = '%.2f' %np.max(df['DELTAT'].values)

    if test_param in ['seeing','all']:
        run_seeing('poor', poor_seeing, poor_skymag, poor_deltat, force_skip=skip)

    if test_param in ['skymag','all']:
        run_skymag('poor', poor_seeing, poor_skymag, poor_deltat, force_skip=skip)

    if test_param in ['deltat','all']:
        run_deltat('poor', poor_seeing, poor_skymag, poor_deltat, force_skip=skip)


