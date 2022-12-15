#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 19:34:00 2022

@author: tomb
"""

import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
#from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor  #GBM algorithm
from sklearn.metrics import accuracy_score, confusion_matrix
#from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from xgboost.sklearn import XGBRegressor, XGBClassifier

df = pd.read_csv('/home/tomb/nfl_models/modeling_data/nfl_spreads_w15.csv')


begin_year=2014
cutoff_week=4
val_year_cutoff=2020
cur_week=15

df['spread_abs']=df['spread_favorite']*-1
df['line']=df['over_under_line']
df['theWeek']=df['schedule_week']
df['spread_sqrd']=df['spread_abs']**2
df['spread_int_line']=df['spread_abs']*df['line']


df = df[df['fav_cover'] != 'push']


#create target fields
df['cover_int']=0
df['cover_int'][(df['fav_cover']=='cover')]=1

df.replace(np.nan, 0.01, inplace=True)

ids= ['spread_sqrd',
'spread_int_line',
'line',
'spread_abs',
'team_id','cover_int','schedule_week','schedule_season','spread_favorite','over_under_line','home_matchup_id']

features = pd.read_csv('/home/tomb/nfl_models/modeling_data/weekly_features/feats_week10_common.csv')
features.drop_duplicates(inplace=True)
print(features.shape)
#features = features.tail(n=500)
good = list(set(features['feature']))
cols = [i for i in good if not ('left' in i or 'center' in i or 'right' in i)]+ids

df = df[cols]
df.to_csv('/home/tomb/nfl_models/modeling_data/weekly_features/w15_final_feats_cut.csv', index=False)


df.fillna(df.median(), inplace=True)
df.drop_duplicates(inplace=True)

targ='cover_int'

df.replace(np.inf, .01, inplace=True)
df.replace(-np.inf, -.01, inplace=True)
df.replace(np.nan, 0, inplace=True)

df=df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
df.dropna(axis=1, how='all', inplace=True)
df.reset_index(inplace=True, drop=True)

df = df[df['schedule_season'] >= begin_year]
df = df[df['schedule_week'] >= cutoff_week]
test = df[(df['schedule_season'] >= val_year_cutoff)]
#test = df[(df['schedule_season'] >= val_year_cutoff) & (df['schedule_week'] >= 10)]
test_10 = test[(test['schedule_season'] == 2022) & (test['schedule_week'] == cur_week)]
test=test[~test.index.isin(test_10.index)]

y_test = test[targ]

data_Model = df[~df.index.isin(test.index)]
data_Model = data_Model[~data_Model.index.isin(test_10.index)]
data_Model = data_Model[(data_Model['schedule_week'] >= 4)|(df['schedule_season'] < val_year_cutoff)]
y_train = data_Model[targ]







plyr_game_cols =[x for x in data_Model.columns[data_Model.columns.str.contains('player_game')]]
week_game_cols =[x for x in data_Model.columns[data_Model.columns.str.contains('week_rank')]]
pred_cols = [x for x in data_Model.columns if x not in ['team_id','unique_team_id','home_matchup_id','fav_cover','cover_int','over_under_result','score_home_fav','score_away_fav','team_conference_x','team_conference_y','team_division_x','team_division_y','fav_homeoraway_y','pf_x','pf_y','WLT','outlier','wl_x','wl_y','OU_result','pf','opponent_id_overall_pff_x','opponent_id_overall_pff_y', 'opponent_id_x','opponent_id_overall_pff','wl','WL','home_matchup_id_x','home_matchup_id_y','opponent_id_overall_pff_x','over_under_result','def_pff_prush_x','unique_team_id_x','unique_team_id_y','home_matchup_id']+plyr_game_cols+week_game_cols]
# pred_cols = [x for x in data_Model.columns if x not in ['team_id','fav_cover','cover_int','over_under_result','score_home_fav','score_away_fav',
#                                                     'unique_team_id_y','home_matchup_id']]
                                                    
                                                        
                                                        
X_train = data_Model[pred_cols]
X_test = test[pred_cols]
cur_test = test_10[pred_cols]
print(cur_test)

                                                        
pipe_lr = Pipeline([('scl', MinMaxScaler()),
#('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
('pca', PCA(n_components=X_train.shape[1]-25)),
('clf', LogisticRegression(penalty='l1', C=.2, solver='liblinear',random_state=42))])

pipe_lr2 = Pipeline([('scl', MinMaxScaler()),
#('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
('pca', PCA(n_components=X_train.shape[1]-20)),
('clf', LogisticRegression(penalty='l2', C=.3, solver='liblinear',random_state=42))])

pipe_dt = Pipeline([('scl', MinMaxScaler()),
#('feature_selection', SelectFromModel(LinearSVC(penalty="l1",C=.1, dual=False))),
('pca', PCA(n_components=X_train.shape[1]-100)),
('clf', LogisticRegression(penalty='l1', C=.5, solver='liblinear',random_state=42))])

pipe_svm = Pipeline([('scl', MinMaxScaler()),
#('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
('pca', PCA(n_components=X_train.shape[1]-50)),
('clf', LogisticRegression(penalty='l1', C=.5, solver='liblinear',random_state=42))])

pipe_svm2 = Pipeline([('scl', MinMaxScaler()),
#('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
('pca', PCA(n_components=X_train.shape[1]-10)),
('clf', LogisticRegression(penalty='l1', C=.5, solver='liblinear',random_state=42))])

pipe_xgb = Pipeline([('scl', MinMaxScaler()),
#('feature_selection', SelectFromModel(LinearSVC(penalty="l1", C=.1, dual=False))),
('pca', PCA(n_components=X_train.shape[1]-20)),
('clf', XGBClassifier(n_estimators=700, random_state=42))])


pipe_rf= Pipeline([('scl', MinMaxScaler()),
#('feature_selection', SelectFromModel(LinearSVC(penalty="l1", C=1, dual=False))),
('pca', PCA(n_components=X_train.shape[1]-50)),
#('clf', svm.SVC(probability=True, C=50, gamma=.1, kernel='rbf', random_state=42))])
('clf', RandomForestClassifier(n_estimators=300, n_jobs=4, random_state=42))])

pipe_gbm= Pipeline([#('scl', MinMaxScaler()),
#('feature_selection', SelectFromModel(LinearSVC(penalty="l1", C=.01, dual=False))),
('clf', XGBClassifier(n_estimators=500, random_state=42))])

pipe_gbm2= Pipeline([#('scl', MinMaxScaler()),
#('feature_selection', SelectFromModel(LinearSVC(penalty="l1", C=.1, dual=False))),
('pca', PCA(n_components=X_train.shape[1]-25)),
('clf', GradientBoostingClassifier(n_estimators=700, learning_rate=.01, random_state=42))])


# List of pipelines for ease of iteration
pipelines = [pipe_lr, pipe_lr2, pipe_dt, pipe_svm, pipe_svm2, pipe_xgb, pipe_rf, pipe_gbm, pipe_gbm2]

# Dictionary of pipelines and classifier types for ease of reference
pipe_dict = {0:'LR', 1:'lr2', 2:'lr3', 3:'svm', 4:'svm2', 5:'xgb', 6:'rForest', 7:'gbm', 8:'gbm2'}

# Fit the pipelines
for pipe in pipelines:
    print(pipe)
    pipe.fit(X_train, y_train)

# Compare accuracies
for idx, val in enumerate(pipelines):
    print('%s pipeline test accuracy: %.3f' % (pipe_dict[idx], val.score(X_test, y_test)))


# Identify the most accurate model on test data
best_acc = 0.0
best_clf = 0
best_pipe = ''
for idx, val in enumerate(pipelines):
    if val.score(X_test, y_test) > best_acc:
        best_acc = val.score(X_test, y_test)
        best_pipe = val
        best_clf = idx
print(idx, (best_pipe))
        
# from logitboost import LogitBoost

  
# lboost = LogitBoost(n_estimators=500, random_state=0, bootstrap=True, learning_rate=.5)
# lboost.fit(X_train, y_train)
        
# y_pred_train = lboost.predict(X_train)
# y_pred_test = lboost.predict(X_test)
# y_pred_test = lboost.predict(cur_test)

# accuracy_train = accuracy_score(y_train, y_pred_train)
# accuracy_test = accuracy_score(y_test, y_pred_test)

# report_train = classification_report(y_train, y_pred_train)
# report_test = classification_report(y_test, y_pred_test)
# print('Training\n%s' % report_train)
# print('Testing\n%s' % report_test)

# xgb=XGBClassifier(n_estimators=700, random_state=42)
# xgb_fit = xgb.fit(X_train, y_train)
# preds = xgb.predict(cur_test)
# from pyglmnet import GLM, simulate_glm
# # create an instance of the GLM class
# glm = GLM(distr='poisson', score_metric='pseudo_R2', reg_lambda=0.01)

# # fit the model on the training data
# glm.fit(np.array(X_train), np.array(y_train))

# # predict using fitted model on the test data
# yhat = glm.predict(np.array(X_test))

# # score the model on test data
# pseudo_R2 = glm.score(np.array(X_test), np.array(y_test))
# print('Pseudo R^2 is %.3f' % pseudo_R2)



result = pd.DataFrame((pipe_lr).predict_proba(X_test))
resultlr = result[[0]]
resultlr.columns = ['lr_preds']

result = pd.DataFrame((pipe_lr2).predict_proba(X_test))
resultlr2 = result[[0]]
resultlr2.columns = ['lr2_preds']

result = pd.DataFrame((pipe_dt).predict_proba(X_test))
resultlr3 = result[[0]]
resultlr3.columns = ['lr3_preds']

result = pd.DataFrame((pipe_svm).predict_proba(X_test))
resultsvm = result[[0]]
resultsvm.columns = ['svm_preds']

result = pd.DataFrame((pipe_xgb).predict_proba(X_test))
resultdt = result[[0]]
resultdt.columns = ['xgb_preds']

result = pd.DataFrame((pipe_svm2).predict_proba(X_test))
resultsvm2 = result[[0]]
resultsvm2.columns = ['svm2_preds']

result = pd.DataFrame((pipe_rf).predict_proba(X_test))
resultrf = result[[0]]
resultrf.columns = ['rf_preds']

result = pd.DataFrame((pipe_gbm).predict_proba(X_test))
resultgbm = result[[0]]
resultgbm.columns = ['gbm_preds']

result = pd.DataFrame((pipe_gbm2).predict_proba(X_test))
resultgbm2 = result[[0]]
resultgbm2.columns = ['gbm2_preds']

result = pd.DataFrame((best_pipe).predict(X_test))
result.columns = ['preds']
result_proba = pd.DataFrame((best_pipe).predict_proba(X_test))
result_proba.columns = ['cover_prob','no_cover_prob']
X_test.reset_index(inplace=True, drop=True)
y_test.reset_index(inplace=True, drop=True)
test_ids = test[['team_id','home_matchup_id','schedule_week','schedule_season','spread_favorite','over_under_line']].reset_index(drop=True)
data_conc = pd.concat([y_test, result, result_proba, resultlr, resultlr2, resultlr3, resultsvm, resultsvm2, resultdt, resultrf, resultgbm, resultgbm2, test_ids], axis=1)
data_conc['ens'] = (data_conc['gbm_preds']+data_conc['rf_preds'])/2

#    result_lr_reg = pd.DataFrame((pipe_lr_reg).predict(cur_test2))
#    result_lr_reg.columns = ['reg_lr']
#    
#    result_svr_reg = pd.DataFrame((pipe_svr_reg).predict(cur_test2))
#    result_svr_reg.columns = ['reg_svr']

result = pd.DataFrame((pipe_lr).predict_proba(cur_test))
resultlr = result[[0]]
resultlr.columns = ['lr_preds']

result = pd.DataFrame((pipe_lr2).predict_proba(cur_test))
resultlr2 = result[[0]]
resultlr2.columns = ['lr2_preds']

result = pd.DataFrame((pipe_dt).predict_proba(cur_test))
resultlr3 = result[[0]]
resultlr3.columns = ['lr3_preds']

result = pd.DataFrame((pipe_svm).predict_proba(cur_test))
resultsvm = result[[0]]
resultsvm.columns = ['svm_preds']

result = pd.DataFrame((pipe_xgb).predict_proba(cur_test))
resultdt = result[[0]]
resultdt.columns = ['xgb_preds']

result = pd.DataFrame((pipe_svm2).predict_proba(cur_test))
resultsvm2 = result[[0]]
resultsvm2.columns = ['svm2_preds']

result = pd.DataFrame((pipe_rf).predict_proba(cur_test))
resultrf = result[[0]]
resultrf.columns = ['rf_preds']

result = pd.DataFrame((pipe_gbm).predict_proba(cur_test))
resultgbm = result[[0]]
resultgbm.columns = ['gbm_preds']

result = pd.DataFrame((pipe_gbm2).predict_proba(cur_test))
resultgbm2 = result[[0]]
resultgbm2.columns = ['gbm2_preds']

result = pd.DataFrame((best_pipe).predict(cur_test))
result.columns = ['preds']
result_proba = pd.DataFrame((best_pipe).predict_proba(cur_test))
result_proba.columns = ['cover_prob','no_cover_prob']
cur_test.reset_index(inplace=True, drop=True)

test_ids = test_10[['team_id','home_matchup_id','schedule_week','schedule_season','spread_favorite','over_under_line']].reset_index(drop=True)
data_conc_cur = pd.concat([result, result_proba, resultlr, resultlr2, resultlr3, resultsvm,resultsvm2, resultdt, resultrf, resultgbm, resultgbm2, test_ids], axis=1)
data_conc_cur.insert(0, 'fav_cover', 'unknown')
#data_conc_cur['actual'] = np.where(data_conc_cur['fav_cover'] == 'over', 1, 0)
data_conc_cur['ens'] = (data_conc_cur['xgb_preds']+data_conc_cur['gbm2_preds'])/2


import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import classification_report

data_conc_2022 = data_conc[data_conc['schedule_season'] == 2022]

def conf_matrix(y_true, y_pred,
                classes,
                normalize=False,
                title='Confusion matrix',
                cmap=plt.cm.Reds):
    """
    Mostly stolen from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    
    Normalization changed, classification_report stats added below plot
    """
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Configure Confusion Matrix Plot Aesthetics (no text yet)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
    
    # Calculate normalized values (so all cells sum to 1) if desired
    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(),2) #(axis=1)[:, np.newaxis]
    
    # Place Numbers as Text on Confusion Matrix Plot
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=12)
    
    
    # Add Precision, Recall, F-1 Score as Captions Below Plot
    rpt = classification_report(y_true, y_pred)
    rpt = rpt.replace('avg / total', '      avg')
    rpt = rpt.replace('support', 'N Obs')
    
    plt.annotate(rpt,
                 xy = (0,-0.3),
                 xytext = (-50, -140),
                 xycoords='axes fraction', textcoords='offset points',
                 fontsize=12, ha='left')    
    
    # Plot
    plt.tight_layout()
    
data_conc_2022['preds']=data_conc_2022['preds'].astype(str)
data_conc_2022['cover_int']=data_conc_2022['cover_int'].astype(str)
conf_matrix(data_conc_2022['cover_int'], data_conc_2022['preds'], classes=['cover','no cover'], normalize=False, title='Conf_Matrix')


import dataframe_image as dfi
df_styled = data_conc_cur.style.background_gradient()
dfi.export(df_styled,"/home/tomb/nfl_models/predictions_final_"+str(cur_week)+".png")



