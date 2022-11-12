# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 09:55:52 2022

@author: core8
"""

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
#import tom_feats_500 as tf
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.simplefilter("ignore", category=ConvergenceWarning)
from tabulate import tabulate


modelfile='nfl_spreads_all10v3.csv'
  

numfeats=100
#df = pd.read_csv('/home/tomb/nfl_models/modeling_data/nfl_spreads_all9.csv')
feat_csv='feats_week10_bestTs_v4'   
df=pd.read_csv('C:/Users/core8/Documents/Python Scripts/pythia/modeling_data/'+modelfile, sep=',', \
               header=0,low_memory=False)

#df=df0.reset_index(inplace=True,drop=True)


feats_file=pd.read_csv('C:/football2022/featSets/'+feat_csv+'.csv', sep=',', header=0)
feats=list(feats_file['feature'])[0:numfeats]
feats=list(set(feats))
begin_year=2014
cutoff_week=5
val_year_cutoff=2022
cur_week=10
replaceQ_inf=.95
replaceQ_neginf=.05

years=[2021,2022]
weeks=[5,6,7,8,9,10,11,12,13,14,15,16]
thresh=.5
thresh_confident_cover=.7

#years=[2021,2022]
#weeks=[9,10]



df['spread_abs']=df['spread_favorite']*-1
df['line']=df['over_under_line']
df['theWeek']=df['schedule_week']
df['spread_sqrd']=df['spread_abs']**2
df['spread_int_line']=df['spread_abs']*df['line']


df = df[df['fav_cover'] != 'push']
df.reset_index(inplace=True,drop=True)

#create target fields
df['cover_int']=0


#df['cover_int'][(df['fav_cover']=='cover')]=1
df.loc[df['fav_cover']=='cover', 'cover_int'] = 1

#df.replace(np.nan, 0.01, inplace=True)


booster='gblinear'
eval_metric='auc'
eval_metric_regressor='rmsle'
ids=['team_id','fav_cover','cover_int','score_home_fav','score_away_fav',\
     'unique_team_id','home_matchup_id','schedule_season','schedule_week']
df_feats_only = df[feats]
df_ids=df[ids]



list_of_infs = list(df_feats_only.columns.to_series()[np.isinf(df_feats_only).any()])
for f in list_of_infs:
    upperQ=df_feats_only[f].quantile(replaceQ_inf)
    lowerQ=df_feats_only[f].quantile(replaceQ_neginf)
    df_feats_only.replace([np.inf],upperQ, inplace=True)
    df_feats_only.replace([-np.inf],lowerQ, inplace=True)
    
df_feats_only=df_feats_only.fillna(df_feats_only.median())
#df_feats_only = pd.DataFrame(StandardScaler().fit_transform(df_feats_only))
#df_feats_only.columns=feats


df=pd.merge(df_ids,df_feats_only, left_index=True, right_index=True, how='left')


#df.fillna(df.median(), inplace=True)
#df.drop_duplicates(inplace=True)

targ='cover_int'

#df.replace(np.inf, .01, inplace=True)
#df.replace(-np.inf, -.01, inplace=True)
#df.replace(np.nan, 0, inplace=True)
#df=df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
#df.dropna(axis=1, how='all', inplace=True)




df.sort_values(['schedule_season','schedule_week'], ascending=True, inplace=True)
df.reset_index(inplace=True,drop=True)
df = df[df['schedule_season'] >= begin_year]
df = df[df['schedule_week'] >= cutoff_week]

truetest_week = df[(df['schedule_season'] == 2022) & (df['schedule_week'] == cur_week)]
truetest_week.reset_index(inplace=True,drop=True)
tt_ids=truetest_week[ids]








def training_pipe(x_data):
    pipe_lr = Pipeline([('scl', MinMaxScaler()),
    #('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
    ('pca', PCA(n_components=x_data.shape[1]-10)),
    ('clf', LogisticRegression(penalty='l1', C=.1, solver='liblinear',random_state=42))])
    
    pipe_lr2 = Pipeline([#('scl', MinMaxScaler()),
    ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
    #('pca', PCA(n_components=x_data.shape[1]-150)),
    ('clf', LogisticRegression(penalty='l2', C=.1, solver='liblinear',random_state=42))])
    
    pipe_dt = Pipeline([('scl', MinMaxScaler()),
    #('feature_selection', SelectFromModel(LinearSVC(penalty="l1",C=.1, dual=False))),
    ('pca', PCA(n_components=x_data.shape[1]-10)),
    ('clf', LogisticRegression(penalty='l1', C=.5, solver='liblinear',random_state=42))])
    
    pipe_svm = Pipeline([('scl', MinMaxScaler()),
    ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
    #('pca', PCA(n_components=x_data.shape[1]-150)),
    ('clf', svm.SVC(probability=True, C=10, gamma=.001, kernel='rbf', random_state=42))])
    
    pipe_svm2 = Pipeline([('scl', MinMaxScaler()),
    #('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
    ('pca', PCA(n_components=x_data.shape[1]-10)),
    ('clf', svm.SVC(probability=True, C=100, gamma=.01, kernel='rbf', random_state=42))])
    
    pipe_xgb = Pipeline([('scl', MinMaxScaler()),
    ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", C=.1, dual=False))),
    #('pca', PCA(n_components=x_data.shape[1]-20)),
    ('clf', XGBClassifier(n_estimators=700, random_state=42,use_label_encoder=False,booster=booster,eval_metric=eval_metric))])
    
    
    pipe_rf= Pipeline([#('scl', MinMaxScaler()),
    ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", C=1, dual=False))),
    #('pca', PCA(n_components=x_data.shape[1]-50)),
    #('clf', svm.SVC(probability=True, C=50, gamma=.1, kernel='rbf', random_state=42))])
    ('clf', RandomForestClassifier(n_estimators=700, n_jobs=4, random_state=42))])
    
    pipe_gbm= Pipeline([#('scl', MinMaxScaler()),
    ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", C=.01, dual=False))),
    ('clf', XGBClassifier(n_estimators=500,random_state=42,use_label_encoder=False,booster=booster,eval_metric=eval_metric))])
    
    pipe_gbm2= Pipeline([#('scl', MinMaxScaler()),
    ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", C=.1, dual=False))),
    #('pca', PCA(n_components=x_data.shape[1]-25)),
    ('clf', GradientBoostingClassifier(n_estimators=700, learning_rate=.01, random_state=42))])
    

    # List of pipelines for ease of iteration
    pipe_dict = {'LR':pipe_lr,'lr2':pipe_lr2,'lr3':pipe_dt,'svm':pipe_svm,\
                 'svm2':pipe_svm2,'xgb':pipe_xgb,'rFor':pipe_rf,'gbm':pipe_gbm,'gbm2':pipe_gbm2}

    return pipe_dict




        
count=0        
global_results=[]
testing_results=[]

headers=['model','year','week','games','wins','losses','winPct','Obs',\
                       'cnfdtCov','cnfdtCovHit','cnfdtNOCov','cnfdtNOCovHit']

for yr in years:
    for wk in weeks:
        
        test = df[(df['schedule_season'] == yr) & (df['schedule_week']==wk)]
        y_test = test[targ]
        y_test=pd.DataFrame(y_test)
        y_test.reset_index(inplace=True,drop=True)
        idx_val=test.index[0]
        
        data_Model = df[df.index<idx_val]
   
        y_train = data_Model[targ]
                                              
        X_train = data_Model[feats]
        X_test = test[feats]
#        cur_test = test_10[feats]
        
        pipe_dict=training_pipe(X_train)
        
        
        
        if (( yr == val_year_cutoff) & (wk == cur_week)):  #this block is for this week's predictions   
            
            count+=1
            print("Reached Week:",cur_week,"Making this week's predictions and exiting.....")
            
            pred_probs=[]
            decisions=[]
            conf_cvr_list=[]
            conf_nocvr_list=[]
            for mdl in pipe_dict:
#            print(pipe)
                local_model=pipe_dict[mdl].fit(X_train, y_train)
                no=mdl+'_no'
                yes=mdl+'_yes'    
                hit=mdl+'_hit'
                mdl_decision=mdl+'_decision'
                mdl_confident_cover=mdl+'_confident_cover'
                mdl_confident_nocover=mdl+'_confident_NOcover'
                
                prd=local_model.predict_proba(truetest_week[feats])
                prd_df=pd.DataFrame(prd)                
                prd_df.columns=[no,yes]   
                
                prd_df[mdl_decision]=0   
                prd_df.loc[prd_df[yes]>=thresh,mdl_decision] = 1

                #cover confidence
                prd_df[mdl_confident_cover]=0
                prd_df.loc[prd_df[yes]>=thresh_confident_cover,mdl_confident_cover] = 1  
                
                #NO cover confidence
                prd_df[mdl_confident_nocover]=0
                prd_df.loc[prd_df[yes]<1-thresh_confident_cover,mdl_confident_nocover] = 1    
                

                pred_probs.append(prd_df[yes])
                decisions.append(prd_df[mdl_decision])
                conf_cvr_list.append(prd_df[mdl_confident_cover])
                conf_nocvr_list.append(prd_df[mdl_confident_nocover])
                
            pred_concat=pd.concat(pred_probs,axis=1)
            decisions_concat=pd.concat(decisions,axis=1)
            pred_conf_cover=pd.concat(conf_cvr_list,axis=1)
            pred_conf_nocover=pd.concat(conf_nocvr_list,axis=1)            
            
            this_weekPredDF=pd.concat([tt_ids,pred_concat,decisions_concat,pred_conf_cover,pred_conf_nocover],axis=1)     
            
            break
         
        else:
            local_results=[]
            print("Testing on: Year=",yr," & week=",wk,'\t', "High Confidence Threshold=","{:.2%}".format(thresh_confident_cover))
            for mdl in pipe_dict:
    #            print(pipe)
                local_model=pipe_dict[mdl].fit(X_train, y_train)
                no=mdl+'_no'
                yes=mdl+'_yes'    
                hit=mdl+'_hit'
                mdl_decision=mdl+'_decision'
                
                mdl_confident_cover=mdl+'_confident_cover'
                mdl_confident_nocover=mdl+'_confident_NOcover'
                mdl_confident_cover_hit=mdl+'_confident_cover_hit'
                mdl_confident_nocover_hit=mdl+'_confident_NOcover_hit'
                
                prd=local_model.predict_proba(X_test)
                prd_df=pd.DataFrame(prd)
                
                prd_df.columns=[no,yes]
            
                prd_df=pd.concat([prd_df,y_test],axis=1)    
                
                
                prd_df[mdl_decision]=0   
                prd_df.loc[prd_df[yes]>=thresh,mdl_decision] = 1
                prd_df[hit]=0
                prd_df.loc[prd_df[mdl_decision]==prd_df['cover_int'], hit] = 1
            
                games=len(X_test)
                wins=prd_df[hit].sum()
                losses=games-wins
                win_pct=round(wins/games,3)
                nobs_train=len(X_train)
                

                
                
                #cover confidence
                prd_df[mdl_confident_cover]=0
                prd_df.loc[prd_df[yes]>=thresh_confident_cover,mdl_confident_cover] = 1           
                prd_df[mdl_confident_cover_hit]=0
                prd_df.loc[((prd_df[mdl_confident_cover]==1) & (prd_df['cover_int']==1)), mdl_confident_cover_hit] = 1                
                prd_df.loc[prd_df[mdl_confident_cover]==0, mdl_confident_cover_hit] = np.nan 
                
                
                #NO cover confidence
                prd_df[mdl_confident_nocover]=0
                prd_df.loc[prd_df[yes]<1-thresh_confident_cover,mdl_confident_nocover] = 1           
                prd_df[mdl_confident_nocover_hit]=0
                prd_df.loc[((prd_df[mdl_confident_nocover]==1) & (prd_df['cover_int']==0)), mdl_confident_nocover_hit] = 1                
                prd_df.loc[prd_df[mdl_confident_nocover]==0, mdl_confident_nocover_hit] = np.nan                 
                
                cnf_cvr_pks=prd_df[mdl_confident_cover_hit].count()
                cnf_cvr_pks_hits=(prd_df[mdl_confident_cover_hit].sum()).astype(int)
#                conf_cover_rate="{:.2%}".format(cnf_cvr_pks_hits/cnf_cvr_pks)
                cnf_nocvr_pks=prd_df[mdl_confident_nocover_hit].count()
                cnf_nocvr_pks_hits=(prd_df[mdl_confident_nocover_hit].sum()).astype(int)                
#                conf_nocover_rate="{:.2%}".format(cnf_nocvr_pks_hits/cnf_nocvr_pks)
                
                
#                print(mdl,'\t',"{:.2%}".format(win_pct),'\t',wins,'\t',losses,'\t',nobs_train,\
#                      '\t',cnf_cvr_pks,cnf_cvr_pks_hits,cnf_nocvr_pks,cnf_nocvr_pks_hits)

                results=[mdl,yr,wk,games,wins,losses,win_pct,nobs_train,\
                         cnf_cvr_pks,cnf_cvr_pks_hits,cnf_nocvr_pks,cnf_nocvr_pks_hits]
                
                
                local_results.append(results)

                global_results.append(results)    
                
                
            local_results_frame=pd.DataFrame(local_results)
            local_results_frame=local_results_frame.round(3)
            local_results_frame.sort_values(local_results_frame.columns[6], ascending=False,inplace=True)                
            
            print(tabulate(local_results_frame,tablefmt='fancy_grid',headers=headers))    
            print('\n')

results_frame=pd.DataFrame(global_results)


results_frame.columns=headers


win_pct_summary_by_model=pd.DataFrame(results_frame.groupby(['model'],as_index=False)[['winPct','wins','losses']].mean())
win_pct_summary_by_model_by_year=pd.DataFrame(results_frame.groupby(['model','year'],as_index=False)[['winPct','wins','losses']].mean())

conf_cover_picks=pd.DataFrame(results_frame.groupby(['model'],as_index=False)[['games','cnfdtCov','cnfdtCovHit']].sum())
conf_cover_picks['cnfdtCovPct']=conf_cover_picks['cnfdtCovHit']/conf_cover_picks['cnfdtCov']
conf_nocover_picks=pd.DataFrame(results_frame.groupby(['model'],as_index=False)[['cnfdtNOCov','cnfdtNOCovHit']].sum())
conf_nocover_picks['cnfdNOCovPct']=conf_nocover_picks['cnfdtNOCovHit']/conf_nocover_picks['cnfdtNOCov']

hi_conf=pd.merge(conf_cover_picks,conf_nocover_picks, left_on=['model'], right_on=['model'], how='left')
final=pd.merge(win_pct_summary_by_model,hi_conf, left_on=['model'], right_on=['model'], how='left')
final.sort_values('winPct', ascending=False,inplace=True) 
final=final.round(3)
print("Final Test Results")

cols1=final.columns
print(tabulate(final,tablefmt='fancy_grid',headers=cols1))

