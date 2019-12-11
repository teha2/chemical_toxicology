# =============================================================================
# Final model of Random Forests with RDKit descriptors
# used for prediction of new chemicals
# outputs the saved models
# =============================================================================
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import glob 
import pickle
# We assume that you have created two subfolders inside the directory where this script is present
# The folder names are: 
# "Data" (contains rdkit output in .csv format with additional column 'Binary Activity'; names of the files start with "rdkit_")
# "Saved_models" (this is where the results of the script will go)

spreadsheets = "Data/rdkit_*.csv"
files = glob.glob(spreadsheets) # retrieves all csv files, returns their path 
filenames = [os.path.basename(x) for x in files] # keep only the filenames not path

n_targets = len(filenames)
for f in range(n_targets):
    target = files[f]
    target_n = filenames[f]
    target_name = target_n[:-4]
    dat = pd.read_csv(target, sep=',')
    
    dat = dat.dropna(0, how='any') #0: removes rows that have NA, 1: removes column that has NA
    
    y = dat.loc[:, dat.columns == 'Binary Activity']
    
    X = dat.loc[:, dat.columns != 'Binary Activity']
    
    corr_mat = X.corr(method = 'pearson')
    corrs = corr_mat[(corr_mat>= 0.8)& (corr_mat<1.0)]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)
    
    ######################## use CV for optimizing parameters of a RF ##########################
    grid_values = { 
        'n_estimators': [10,50, 100, 200],    
        'max_depth' : [5,6,7,8,9], }
    
    clf = RandomForestClassifier(random_state= 0)
    clf_cv = GridSearchCV(clf, param_grid = grid_values,scoring='roc_auc') #'recall','roc_auc','f1'
    clf_cv.fit(X_train, y_train.values.ravel())
    mean_test_scores = clf_cv.cv_results_['mean_test_score']
    
    ##################### train classifier using the optimized parameters #########################
    clf_optim = RandomForestClassifier(n_estimators = clf_cv.best_params_["n_estimators"],max_features = 'auto',max_depth = clf_cv.best_params_["max_depth"],random_state = 100).fit(X, y.values.ravel())
    model_name = target_name+'_traditional_saved_model.sav'
    fpath = os.path.join('Saved_models',model_name)
    with open(fpath, 'wb') as m:
        pickle.dump(clf_optim, m)
    
