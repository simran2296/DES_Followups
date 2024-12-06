# A module to perform ML classification on a single dataset

import numpy as np
import pandas as pd
import sys
pd.set_option('use_inf_as_null', True)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

def classify(train_df, test_dfs, test_df_outfiles, report_files, n_jobs=-1, output_auc=False):
    # drop NaN rows in training set
    starting_shape = train_df.shape[0]
    train_df = train_df.replace([np.inf, -np.inf, 'inf', 'nan', 'NaN'], np.nan).dropna(axis=0, how='any').reset_index(drop=True)
    if train_df.shape[0] < starting_shape:
        print("WARNING: %i samples dropped from training set due to NaN values." %(starting_shape - train_df.shape[0]))
    
    # require a large enough training set
    if train_df.shape[0] < 5:
        if output_auc:
            return [0.0], [0]
        else:
            return

    #print("Beginning classification")
    # Encode KN vs All scheme
    kn_truth = [1 if x == 'KN' else 0 for x in train_df['OBJ'].values]
    train_df['KN'] = kn_truth

    # require presence of KN in the training set
    if train_df[train_df['KN'] == 1].shape[0] < 10:
        if output_auc:
            return [0.0], [0]
        else:
            return

    # Force numeric features
    metadata_cols = ['OBJ']
    numeric_cols = [x for x in train_df.columns if x not in metadata_cols]
    train_df[numeric_cols] = train_df[numeric_cols].apply(pd.to_numeric)

    #try dropping any extra large values
    train_df[numeric_cols] = train_df[numeric_cols][~(train_df[numeric_cols] > 1.e6).any(axis=1)]

    # Break train_df in training and validation sets
    all_features = [x for x in numeric_cols if x not in ['CID', 'SNID', 'KN']]

    #drop slope features for now
    all_features = [x for x in all_features if x[0:5] != 'slope']

    X = train_df[all_features]
    y = train_df['KN']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6, stratify=y)

    # Instantiate classifier
    rfc = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=6, criterion='gini')
    
    # Perform a grid search to find the best hyperparameters
    #print("Performing hyperparameter gridsearch")
    param_grid = {'criterion': ['gini', 'entropy'],
                  'n_estimators': [10, 50, 100, 500],
                  'max_depth': [3, 5, 10, 20],
                  'class_weight': ['balanced_subsample', 'balanced', {0: 1, 1: 1}, {0: 5, 1:5}]}
    gs = GridSearchCV(rfc, param_grid, n_jobs=n_jobs, cv=5)
    try:
        gs.fit(X_train, y_train)

        # Instantiate a new RFC with the best parameters
        rfc = gs.best_estimator_
        error = False

    except:
        error = True
        rfc = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=6, criterion='entropy', class_weight='balanced_subsample')

    # Determine which features are prone to overfitting
    feature_names = np.array(all_features)

    if error:
        print("\nERROR DETECTED. Beginning diagnostic tests...")
        num_good = 0
        print("Testing %i features" %len(all_features))
        for feat in all_features:
            print("Testing %s..." %feat)
            try:
                rfc.fit(X_train[feat].reshape(-1, 1), y_train)
                print("%s GOOD" %feat)
                num_good += 1
            except:
                print("%s BAD" %feat)

        if num_good == len(all_features):
            print("\nNo errors found in single feature testing.\n")
            print("\nTesting multiple features...\n")

            working_feats = [all_features[0]]
            for index in range(len(all_features[1:])):
                next_feat = all_features[index + 1]
                working_feats.append(next_feat)
                try:
                    rfc.fit(X_train[working_feats], y_train)
                    print("%s GOOD" %next_feat)
                except:
                    print("%s BAD" %next_feat)

        sys.exit()


    fi = rfc.feature_importances_
    sorted_fi = sorted(fi)

    #print("Optimizing classification features")
    feature_dict = {}
    ## Method 1: use only featrues above maximum gradient
    feature_importance_cut = sorted_fi[np.argmax(np.gradient(sorted_fi))]
    feats = feature_names[np.where(rfc.feature_importances_ > feature_importance_cut)]
    if len(feats) > 0:
        rfc_working = gs.best_estimator_.fit(X_train[feats], y_train)
        feature_dict[1] = {'FEATURES': feats,
                           'SCORE': rfc_working.score(X_test[feats], y_test),
                           'CUTOFF': feature_importance_cut}
    else:
        feature_dict[1] = {'FEATURES': feats,
                           'SCORE': 0.0,
                           'CUTOFF': feature_importance_cut}

    ## Method 2: use only features above a slightly lower cutoff
    feature_importance_cut = sorted_fi[np.argmax(np.gradient(sorted_fi))] / (0.25 * len(feature_names))
    feats = feature_names[np.where(rfc.feature_importances_ > feature_importance_cut)]
    if len(feats) > 0:
        rfc_working = gs.best_estimator_.fit(X_train[feats], y_train)
        feature_dict[2] = {'FEATURES': feats,
                           'SCORE': rfc_working.score(X_test[feats], y_test),
                           'CUTOFF': feature_importance_cut}
    else:
        feature_dict[2] = {'FEATURES': feats,
                           'SCORE': 0.0,
                           'CUTOFF': feature_importance_cut}

    ## Method 3: use all features
    rfc_working = gs.best_estimator_.fit(X_train, y_train)
    feature_dict[3] = {'FEATURES': all_features,
                       'SCORE': rfc_working.score(X_test, y_test),
                       'CUTOFF': 0.0}

    # Choose best feature selection method
    best_score = 0.0
    for m, info in feature_dict.items():
        if info['SCORE'] > best_score:
            method = m
            feats = info['FEATURES']
            best_score = info['SCORE']

    #print("Performing final classification")
    # Train final classifier on all training data
    rfc_final = gs.best_estimator_
    rfc_final.fit(X[feats], y)

    #track samples and roc scores
    samples, roc_auc_scores = [], []
    
    for test_df, test_df_outfile, report_file in zip(test_dfs, test_df_outfiles, report_files):
        metadata_cols = ['OBJ', 'KN']
        numeric_cols = [x for x in test_df.columns if x not in metadata_cols]
        test_df[numeric_cols] = test_df[numeric_cols].apply(pd.to_numeric)
 
        #Clean test df and skip if no samples remain
        starting_shape = test_df.shape[0]
        test_df = test_df.copy().replace([np.inf, -np.inf, 'inf', 'nan', 'NaN', 'NAN'], np.nan).dropna(axis=0, how='any').reset_index(drop=True)
        
        if test_df.shape[0] < starting_shape:
            print("WARNING: %i samples dropped from testing set due to NaN values." %(starting_shape - test_df.shape[0]))
        if test_df.shape[0] == 0:
            continue

        # Predict test data
        try:
            scores = rfc_final.predict_proba(test_df[feats].replace([np.inf, -np.inf, 'inf', 'nan', 'NaN', 'NAN'], np.nan).dropna(axis=0, how='any'))
        except:
            print(sys.getsizeof(test_df))
            print([type(x) for x in np.max(test_df)[np.max(test_df) > 1.e7].values])
            test_df[feats] = test_df[feats][~(test_df[feats] > 1.e7).any(axis=1)]
            scores = rfc_final.predict_proba(test_df[feats].replace([np.inf, -np.inf, 'inf', 'nan', 'NaN', 'NAN'], np.nan).dropna(axis=0, how='any'))
            
        test_df.replace([np.inf, -np.inf, 'inf'], np.nan).dropna(axis=0, how='any')['PROB_KN'] = scores[:,1]

        # Calculate ROC AUC if desired
        if output_auc:
            test_set_truth = [1 if x == 'KN' else 0 for x in test_df['OBJ'].values]
            if len(np.unique(test_set_truth)) > 1:
                roc_auc = roc_auc_score(test_set_truth, scores[:,1])
            else:
                roc_auc = 1.0
        else:
            roc_auc = -1.0

        # Output results
        write_output(all_features, fi, gs.best_params_, feature_dict, roc_auc, test_df_outfile, report_file)
        test_df.to_csv(test_df_outfile, index=False)
        roc_auc_scores.append(roc_auc)
        samples.append(test_df.shape[0])
        
    # Return ROC AUC if desired
    if output_auc:
        # perform weighted average by number of samples
        #return np.dot(np.array(roc_auc_scores), np.array(samples)) / np.sum(np.array(samples))
        return roc_auc_scores, samples
    else:
        return

def write_output(features, feature_importances, best_params, feature_dict, auc, test_df_outfile, report_file):
    outlines = []
    outlines.append('\nRESULTS:\n------------------------------')
    outlines.append('Available Features and Raw Importances:')
    for feat, imp in zip(features, feature_importances):
        outlines.append("F:\t%.5f\t%s" %(imp, feat))
    outlines.append('\nFeature Optimization:')
    for method, info in feature_dict.items():
        outlines.append("M_START: %i" %method)
        outlines.append("\tCUTOFF: %.5f" %info['CUTOFF'])
        outlines.append("\tFEATURES:")
        for feat in info['FEATURES']:
            outlines.append("\t\t%s" %feat)
        outlines.append("\tSCORE: %.5f" %info['SCORE'])
        outlines.append("M_END: %i\n" %method)
    outlines.append('\nValidated Classifier Parameters:')
    for param, value in best_params.items():
        outlines.append("P:\t%s: %s" %(param, value))
    outlines.append('\nFinal Test Set AUC: %s' %auc)
    outlines.append('\nFinal Test Set Outfile: %s\n' %test_df_outfile)

    formatted_lines = [x + '\n' for x in outlines]

    outfile = open(report_file, 'w+')
    outfile.writelines(formatted_lines)
    outfile.close()

    return

