
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import re
from string import ascii_letters, digits
import utils.cleaning_dicts
#import matplotlib
import os
import sys
import warnings
warnings.filterwarnings('ignore')


# In[2]:

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

os.path.abspath(os.getcwd())
#get_ipython().system('ls')
cur_dir=os.chdir('C:/Users/core8/Documents/Python Scripts/pythia/')

# ### IMPORTANT: Users must change the week values to the current week in the cell below

# In[3]:

cur_week_int = 14
cur_week_str = str(cur_week_int)
version='2z'

# ### Read in, clean and process all pff position datasets

# In[4]:

####################################################################################
                ###   Read-in and clean all passing datasets ###
####################################################################################

passing_depth = pd.read_csv('./historic_data/pff_data/passing_depth_hist.csv')
passing_allowed_pressure = pd.read_csv('./historic_data/pff_data/passing_allowed_pressure_hist.csv')
passing_pressure = pd.read_csv('./historic_data/pff_data/passing_pressure_hist.csv')
passing_concept = pd.read_csv('./historic_data/pff_data/passing_concept_hist.csv')
time_in_pocket = pd.read_csv('./historic_data/pff_data/time_in_pocket_hist.csv')
passing_summ_conc = pd.read_csv('./historic_data/pff_data/passing_summ_conc_hist.csv')

passing_depth_new = pd.read_csv('./scripts/nfl_all/passing_depth_2022.csv')
passing_allowed_pressure_new = pd.read_csv('./scripts/nfl_all/passing_allowed_pressure_2022.csv')
passing_pressure_new = pd.read_csv('./scripts/nfl_all/passing_pressure_2022.csv')
passing_concept_new = pd.read_csv('./scripts/nfl_all/passing_concept_2022.csv')
time_in_pocket_new = pd.read_csv('./scripts/nfl_all/time_in_pocket_2022.csv')
passing_summ_conc_new = pd.read_csv('./scripts/nfl_all/passing_summ_conc_2022.csv')
                                 
passing_depth = pd.concat([passing_depth, passing_depth_new], axis=0).reset_index(drop=True)
passing_allowed_pressure = pd.concat([passing_allowed_pressure, passing_allowed_pressure_new], axis=0).reset_index(drop=True)
passing_pressure = pd.concat([passing_pressure, passing_pressure_new], axis=0).reset_index(drop=True)
passing_concept = pd.concat([passing_concept, passing_concept_new], axis=0).reset_index(drop=True)
time_in_pocket = pd.concat([time_in_pocket, time_in_pocket_new], axis=0).reset_index(drop=True)
passing_summ_conc = pd.concat([passing_summ_conc, passing_summ_conc_new], axis=0).reset_index(drop=True)
                                 

def drop_non_qbs(df):
    df=df.rename(columns={"player_id": "numeric_id"})
    df=df[df['position'] == 'QB']
    df['position']=df['position'].astype(str).str.lower()
    df['team_name']=df['team_name'].astype(str).str.lower()       
    df['player']=df['player'].str.replace('[^a-zA-Z0-9]', '').str.lower()
    df['team_name']=df['team_name'].str.lower()
    df['team_name']=df['team_name'].replace("oak","lv")
    df['year'] = df['year'].astype(str)
    df['week'] = df['week'].astype(str)
    
        ##  pass team name through dictionary to clean ##
    df['team_name'] = df['team_name'].map(utils.cleaning_dicts.clean_team_pff).fillna(df['team_name'])
    df['position'] = df['position'].map(utils.cleaning_dicts.pos_dict).fillna(df['position'])

    
    df.insert(0, "p_id", (df['player']+'_'+df['team_name']+'_'+df['year']+'_'+df['week']))
    df.insert(1, "unique_team_id", (df['team_name']+'_'+df['year']+'_'+df['week']))
    df.insert(2, "player_team_id", (df['player']+'_'+df['team_name']+'_'+df['year']))
    df.insert(3, "team_id_impute", (df['team_name']+'_'+df['year']))
    return df
    
passing_depth = drop_non_qbs(passing_depth)
passing_allowed_pressure = drop_non_qbs(passing_allowed_pressure)
passing_pressure = drop_non_qbs(passing_pressure)
passing_concept = drop_non_qbs(passing_concept)
time_in_pocket = drop_non_qbs(time_in_pocket)
passing_summ_conc = drop_non_qbs(passing_summ_conc)


passing_depth = passing_depth[passing_depth.columns.drop(list(passing_depth.filter(regex='left|right|center')))]

####################################################################################
				###   Read-in and clean all receiving datasets ### scripts/nfl_all
####################################################################################

rec_summ_conc = pd.read_csv('./historic_data/pff_data/rec_summ_conc_hist.csv')
receiving_concept = pd.read_csv('./historic_data/pff_data/receiving_concept_hist.csv')
receiving_depth = pd.read_csv('./historic_data/pff_data/receiving_depth_hist.csv')
receiving_scheme = pd.read_csv('./historic_data/pff_data/receiving_scheme_hist.csv')
                                 
rec_summ_conc_new = pd.read_csv('./scripts/nfl_all/rec_summ_conc_2022.csv')
receiving_concept_new = pd.read_csv('./scripts/nfl_all/receiving_concept_2022.csv')
receiving_depth_new = pd.read_csv('./scripts/nfl_all/receiving_depth_2022.csv')
receiving_scheme_new = pd.read_csv('./scripts/nfl_all/receiving_scheme_2022.csv')

receiving_depth = receiving_depth[receiving_depth.columns.drop(list(receiving_depth.filter(regex='left|right|center')))]
receiving_depth_new = receiving_depth_new[receiving_depth_new.columns.drop(list(receiving_depth_new.filter(regex='left|right|center')))]
                                
rec_summ_conc = pd.concat([rec_summ_conc, rec_summ_conc_new], axis=0).reset_index(drop=True)
receiving_concept = pd.concat([receiving_concept, receiving_concept_new], axis=0).reset_index(drop=True)
receiving_depth = pd.concat([receiving_depth, receiving_depth_new], axis=0).reset_index(drop=True)
receiving_scheme = pd.concat([receiving_scheme, receiving_scheme_new], axis=0).reset_index(drop=True)   

                    

def drop_non_recs(df):
    df=df.rename(columns={"player_id": "numeric_id"})
    df= df[df.position.str.match('WR|TE|HB|FB')]
    df['position']=df['position'].astype(str).str.lower()
    df['team_name']=df['team_name'].astype(str).str.lower()       
    df['player']=df['player'].str.replace('[^a-zA-Z0-9]', '').str.lower()
    df['team_name']=df['team_name'].str.lower()
    df['team_name']=df['team_name'].replace("oak","lv")
    df['year'] = df['year'].astype(str)
    df['week'] = df['week'].astype(str)
    
        ##  pass team name through dictionary to clean ##
    df['team_name'] = df['team_name'].map(utils.cleaning_dicts.clean_team_pff).fillna(df['team_name'])
    df['position'] = df['position'].map(utils.cleaning_dicts.pos_dict).fillna(df['position'])

    
    df.insert(0, "p_id", (df['player']+'_'+df['team_name']+'_'+df['year']+'_'+df['week']))
    df.insert(1, "unique_team_id", (df['team_name']+'_'+df['year']+'_'+df['week']))
    df.insert(2, "player_team_id", (df['player']+'_'+df['team_name']+'_'+df['year']))
    df.insert(3, "team_id_impute", (df['team_name']+'_'+df['year']))
    return df

rec_summ_conc = drop_non_recs(rec_summ_conc)
receiving_concept = drop_non_recs(receiving_concept)
receiving_depth = drop_non_recs(receiving_depth)
receiving_scheme = drop_non_recs(receiving_scheme)


####################################################################################
				###   Read-in and clean all rushing datasets ###
####################################################################################

rush_summ_conc = pd.read_csv('./historic_data/pff_data/rush_summ_conc_hist.csv')
rush_summ_conc_new = pd.read_csv('./scripts/nfl_all/rush_summ_conc_2022.csv')                                 
                                 
rush_summ_conc = pd.concat([rush_summ_conc, rush_summ_conc_new], axis=0)
 

def drop_non_rbs(df):
    df=df.rename(columns={"player_id": "numeric_id"})
    df= df[df.position.str.match('WR|HB|FB|QB')]
    df['position']=df['position'].astype(str).str.lower()
    df['team_name']=df['team_name'].astype(str).str.lower()       
    df['player']=df['player'].str.replace('[^a-zA-Z0-9]', '').str.lower()
    df['team_name']=df['team_name'].str.lower()
    df['team_name']=df['team_name'].replace("oak","lv")
    df['year'] = df['year'].astype(str)
    df['week'] = df['week'].astype(str)
        ##  pass team name through dictionary to clean ##
    df['team_name'] = df['team_name'].map(utils.cleaning_dicts.clean_team_pff).fillna(df['team_name'])
    df['position'] = df['position'].map(utils.cleaning_dicts.pos_dict).fillna(df['position'])

    
    df.insert(0, "p_id", (df['player']+'_'+df['team_name']+'_'+df['year']+'_'+df['week']))
    df.insert(1, "unique_team_id", (df['team_name']+'_'+df['year']+'_'+df['week']))
    df.insert(2, "player_team_id", (df['player']+'_'+df['team_name']+'_'+df['year']))
    df.insert(3, "team_id_impute", (df['team_name']+'_'+df['year']))
    return df

rush_summ_conc = drop_non_rbs(rush_summ_conc)


####################################################################################
				###   Read-in and clean all blocking datasets ###
####################################################################################


block_summ_conc = pd.read_csv('./historic_data/pff_data/block_summ_conc_hist.csv')
offense_pass_blocking = pd.read_csv('./historic_data/pff_data/offense_pass_blocking_hist.csv')
offense_run_blocking = pd.read_csv('./historic_data/pff_data/offense_run_blocking_hist.csv')
                                 
block_summ_conc_new = pd.read_csv('./scripts/nfl_all/block_summ_conc_2022.csv')
offense_pass_blocking_new = pd.read_csv('./scripts/nfl_all/offense_pass_blocking_2022.csv')
offense_run_blocking_new = pd.read_csv('./scripts/nfl_all/offense_run_blocking_2022.csv')                                 

block_summ_conc = pd.concat([block_summ_conc, block_summ_conc_new], axis=0).reset_index(drop=True)
offense_pass_blocking = pd.concat([offense_pass_blocking, offense_pass_blocking_new], axis=0).reset_index(drop=True)
offense_run_blocking = pd.concat([offense_run_blocking, offense_run_blocking_new], axis=0).reset_index(drop=True)

def drop_non_ols(df):
    df=df.rename(columns={"player_id": "numeric_id"})
    df = df[df['position'].notna()]
    df= df[df.position.str.match('T|C|G|TE')]
    df['position']=df['position'].astype(str).str.lower()
    df['team_name']=df['team_name'].astype(str).str.lower()       
    df['player']=df['player'].str.replace('[^a-zA-Z0-9]', '').str.lower()
    df['team_name']=df['team_name'].str.lower()
    df['team_name']=df['team_name'].replace("oak","lv")
    df['year'] = df['year'].astype(str)
    df['week'] = df['week'].astype(str)
        ##  pass team name through dictionary to clean ##
    df['team_name'] = df['team_name'].map(utils.cleaning_dicts.clean_team_pff).fillna(df['team_name'])
    df['position'] = df['position'].map(utils.cleaning_dicts.pos_dict).fillna(df['position'])

    
    df.insert(0, "p_id", (df['player']+'_'+df['team_name']+'_'+df['year']+'_'+df['week']))
    df.insert(1, "unique_team_id", (df['team_name']+'_'+df['year']+'_'+df['week']))
    df.insert(2, "player_team_id", (df['player']+'_'+df['team_name']+'_'+df['year']))
    df.insert(3, "team_id_impute", (df['team_name']+'_'+df['year']))
    return df


block_summ_conc	= drop_non_ols(block_summ_conc)
offense_pass_blocking = drop_non_ols(offense_pass_blocking)
offense_run_blocking = drop_non_ols(offense_run_blocking)



####################################################################################
				###   Read-in and clean all defensive datasets ###
####################################################################################

def_summ_conc = pd.read_csv('./historic_data/pff_data/def_summ_conc_hist.csv')
pass_rush_summary = pd.read_csv('./historic_data/pff_data/pass_rush_summary_hist.csv')
run_defense_summary = pd.read_csv('./historic_data/pff_data/run_defense_summary_hist.csv')
defense_coverage_scheme = pd.read_csv('./historic_data/pff_data/defense_coverage_scheme_hist.csv')
defense_coverage_summary = pd.read_csv('./historic_data/pff_data/defense_coverage_summary_hist.csv')
slot_coverage = pd.read_csv('./historic_data/pff_data/slot_coverage_hist.csv')
                                 
def_summ_conc_new = pd.read_csv('./scripts/nfl_all/def_summ_conc_2022.csv')
pass_rush_summary_new = pd.read_csv('./scripts/nfl_all/pass_rush_summary_2022.csv')
run_defense_summary_new = pd.read_csv('./scripts/nfl_all/run_defense_summary_2022.csv')
defense_coverage_scheme_new = pd.read_csv('./scripts/nfl_all/defense_coverage_scheme_2022.csv')
defense_coverage_summary_new = pd.read_csv('./scripts/nfl_all/defense_coverage_summary_2022.csv')
slot_coverage_new = pd.read_csv('./scripts/nfl_all/slot_coverage_2022.csv')

def_summ_conc = pd.concat([def_summ_conc, def_summ_conc_new], axis=0).reset_index(drop=True)
pass_rush_summary = pd.concat([pass_rush_summary, pass_rush_summary_new], axis=0).reset_index(drop=True)
run_defense_summary = pd.concat([run_defense_summary, run_defense_summary_new], axis=0).reset_index(drop=True)
defense_coverage_scheme = pd.concat([defense_coverage_scheme, defense_coverage_scheme_new], axis=0).reset_index(drop=True)
defense_coverage_summary = pd.concat([defense_coverage_summary, defense_coverage_summary_new], axis=0).reset_index(drop=True)
slot_coverage = pd.concat([slot_coverage, slot_coverage_new], axis=0).reset_index(drop=True)
                                 
def drop_non_def(df):
    df=df.rename(columns={"player_id": "numeric_id"})
    df['position']=df['position'].astype(str).str.lower()
    df['team_name']=df['team_name'].astype(str).str.lower()       
    df['player']=df['player'].str.replace('[^a-zA-Z0-9]', '').str.lower()
    df['team_name']=df['team_name'].str.lower()
    df['team_name']=df['team_name'].replace("oak","lv")
    df['year'] = df['year'].astype(str)
    df['week'] = df['week'].astype(str)
        ##  pass team name through dictionary to clean ##
    df['team_name'] = df['team_name'].map(utils.cleaning_dicts.clean_team_pff).fillna(df['team_name'])
    df['position'] = df['position'].map(utils.cleaning_dicts.pos_dict).fillna(df['position'])

    
    df.insert(0, "p_id", (df['player']+'_'+df['team_name']+'_'+df['year']+'_'+df['week']))
    df.insert(1, "unique_team_id", (df['team_name']+'_'+df['year']+'_'+df['week']))
    df.insert(2, "player_team_id", (df['player']+'_'+df['team_name']+'_'+df['year']))
    df.insert(3, "team_id_impute", (df['team_name']+'_'+df['year']))
    return df

def_summ_conc = drop_non_def(def_summ_conc)
pass_rush_summary = drop_non_def(pass_rush_summary)
run_defense_summary = drop_non_def(run_defense_summary)
defense_coverage_scheme = drop_non_def(defense_coverage_scheme)
defense_coverage_summary = drop_non_def(defense_coverage_summary)
slot_coverage = drop_non_def(slot_coverage)

def_summ_conc=def_summ_conc[def_summ_conc['position'].isin(["ed","lb","di","s","cb"])]
pass_rush_summary=pass_rush_summary[pass_rush_summary['position'].isin(["ed","lb","di","s"])]
run_defense_summary=run_defense_summary[run_defense_summary['position'].isin(["ed","lb","di","s","cb"])]
defense_coverage_scheme=defense_coverage_scheme[defense_coverage_scheme['position'].isin(["lb","s","cb"])]
defense_coverage_summary=defense_coverage_summary[defense_coverage_summary['position'].isin(["lb","s","cb"])]
slot_coverage=slot_coverage[slot_coverage['position'].isin(["lb","s","cb"])]

####################################################################################
				###   Read-in and clean all special teams datasets ###
####################################################################################	

st_kickers = pd.read_csv('./historic_data/pff_data/st_kickers_hist.csv')
st_punters = pd.read_csv('./historic_data/pff_data/st_punters_hist.csv')

st_kickers_new = pd.read_csv('./scripts/nfl_all/st_kickers_2022.csv')
st_punters_new = pd.read_csv('./scripts/nfl_all/st_punters_2022.csv')                                 
                                 
                                 
st_kickers = pd.concat([st_kickers, st_kickers_new], axis=0).reset_index(drop=True)
st_punters = pd.concat([st_punters, st_punters_new], axis=0).reset_index(drop=True)
                                 
def clean_spec(df):
    df=df.rename(columns={"player_id": "numeric_id"})
    df['position']=df['position'].astype(str).str.lower()
    df['team_name']=df['team_name'].astype(str).str.lower()       
    df['player']=df['player'].str.replace('[^a-zA-Z0-9]', '').str.lower()
    df['team_name']=df['team_name'].str.lower()
    df['team_name']=df['team_name'].replace("oak","lv")
    df['year'] = df['year'].astype(str)
    df['week'] = df['week'].astype(str)
        ##  pass team name through dictionary to clean ##
    df['team_name'] = df['team_name'].map(utils.cleaning_dicts.clean_team_pff).fillna(df['team_name'])
    df['position'] = df['position'].map(utils.cleaning_dicts.pos_dict).fillna(df['position'])

    
    df.insert(0, "p_id", (df['player']+'_'+df['team_name']+'_'+df['year']+'_'+df['week']))
    df.insert(1, "unique_team_id", (df['team_name']+'_'+df['year']+'_'+df['week']))
    df.insert(2, "player_team_id", (df['player']+'_'+df['team_name']+'_'+df['year']))
    df.insert(3, "team_id_impute", (df['team_name']+'_'+df['year']))
    return df

st_kickers =clean_spec(st_kickers)
st_punters = clean_spec(st_punters)



####################################################################################
####################################################################################
####################################################################################


# ### Impute all missing values in pff dataframe - NEED TO UPDATE

# In[5]:

#get_ipython().run_cell_magic('time', '', "\ndef impute(df):\n    df = df.apply(pd.to_numeric, errors='ignore')\n    df.reset_index(inplace=True, drop=True)\n    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()\n    df[num_cols]= df.groupby(df['team_id_impute'])[num_cols].fillna(df.mean()).reset_index(level=0, drop=True)\n    return df\n\npassing_depth = impute(passing_depth)\npassing_allowed_pressure = impute(passing_allowed_pressure)\npassing_pressure = impute(passing_pressure)\npassing_concept = impute(passing_concept)\ntime_in_pocket = impute(time_in_pocket)\npassing_summ_conc = impute(passing_summ_conc)\n\nrec_summ_conc = impute(rec_summ_conc)\nreceiving_concept = impute(receiving_concept)\nreceiving_depth = impute(receiving_depth)\nreceiving_scheme = impute(receiving_scheme)\n\nrush_summ_conc = impute(rush_summ_conc)\n\nblock_summ_conc = impute(block_summ_conc)\noffense_pass_blocking = impute(offense_pass_blocking)\noffense_run_blocking = impute(offense_run_blocking)\n\ndef_summ_conc = impute(def_summ_conc)\npass_rush_summary = impute(pass_rush_summary)\nrun_defense_summary = impute(run_defense_summary)\ndefense_coverage_scheme = impute(defense_coverage_scheme)\ndefense_coverage_summary = impute(defense_coverage_summary)\nslot_coverage = impute(slot_coverage)\n\nst_kickers = impute(st_kickers)\nst_punters = impute(st_punters)")


# ### Add prefixes to all columns.  Creating column names structured as "source-dataset_column-name"

# In[6]:

####################################################################################
								###   add prefixes ###
####################################################################################	

def create_prefix(prefix=None, df=None):
    id = df[['p_id','player_team_id','unique_team_id','team_id_impute','player','numeric_id','position','team_name','year','week']]
    temp = df.drop(['p_id','player','player_team_id','unique_team_id','player','team_id_impute','numeric_id','position','team_name','unique_team_id','numeric_id','position','team_name','year','week'], axis=1)
    temp = temp.add_prefix(prefix)
    id = pd.concat([id, temp], axis=1)
    return id

def id_prefix(prefix=None, df=None):
    id = df[['p_id','player','player_team_id','unique_team_id','team_id_impute','numeric_id','position','team_name','year','week']]
    temp = df.drop(['p_id','player','player_team_id','unique_team_id','team_id_impute','numeric_id','position','team_name','year','week'], axis=1)
    temp = temp.add_prefix(prefix)
    id = pd.concat([id, temp], axis=1)
    return id

passing_summ_conc = id_prefix(prefix="pass_summary_", df=passing_summ_conc)
rush_summ_conc = id_prefix(prefix="rush_summary_", df=rush_summ_conc)
rec_summ_conc = id_prefix(prefix="rec_summary_", df=rec_summ_conc)
block_summ_conc = id_prefix(prefix="block_summary_", df=block_summ_conc)
def_summ_conc = id_prefix(prefix="def_summary_", df=def_summ_conc)
st_kickers = id_prefix(prefix="kicking_", df=st_kickers)
st_punters = id_prefix(prefix="punting_", df=st_punters)


passing_depth = create_prefix(prefix="pass_depth_", df=passing_depth)
passing_allowed_pressure = create_prefix(prefix="pressure_source_", df=passing_allowed_pressure)
passing_pressure = create_prefix(prefix="pass_under_pressure_", df=passing_pressure)
passing_concept = create_prefix(prefix="pass_concept_", df=passing_concept)
time_in_pocket = create_prefix(prefix="pass_time_", df=time_in_pocket)


receiving_concept = create_prefix(prefix="rec_concept_", df=receiving_concept)
receiving_depth = create_prefix(prefix="rec_depth_", df=receiving_depth)
receiving_scheme = create_prefix(prefix="rec_scheme_", df=receiving_scheme)

offense_pass_blocking = create_prefix(prefix="pass_block_", df=offense_pass_blocking)
offense_run_blocking = create_prefix(prefix="run_block_", df=offense_run_blocking)


pass_rush_summary = create_prefix(prefix="pass_rush_", df=pass_rush_summary)
run_defense_summary = create_prefix(prefix="run_defense_", df=run_defense_summary)
defense_coverage_scheme = create_prefix(prefix="def_coverage_scheme_", df=defense_coverage_scheme)
defense_coverage_summary = create_prefix(prefix="def_coverage_summary_", df=defense_coverage_summary)
slot_coverage = create_prefix(prefix="def_slot_coverage_", df=slot_coverage)


# ### Read in weather data and clean raiders name - merged onto spreads data below ###

# In[7]:

### read in weather data###
weather = pd.read_csv('./current_data/week_'+cur_week_str+'/weather_hist_all.csv')

def raiders(df):
    if 'oak' in str(df.away_matchup_id) and '2020' in str(df.away_matchup_id):
        return df.away_matchup_id.replace("oak","lv")
    if 'oak' in str(df.away_matchup_id) and '2021' in str(df.away_matchup_id):
        return df.away_matchup_id.replace("oak","lv")
    if 'oak' in str(df.away_matchup_id) and '2022' in str(df.away_matchup_id):
        return df.away_matchup_id.replace("oak","lv")
    else:
        return df.away_matchup_id
weather['away_matchup_id'] = weather.apply(lambda df: raiders(df), axis=1)


# ### Create spreads data ###

# In[8]:

####################################################################################
				###   spreads data cleaning and engineering ###
####################################################################################

spreads = pd.read_csv('./current_data/week_'+cur_week_str+'/spreadsw'+cur_week_str+'.csv')

new_acc = {'oak':'lv',
          'sd':'lac',
          'stl':'lar'}  

spreads['team_home_abb'] = spreads['team_home_abb'].map(new_acc).fillna(spreads['team_home_abb'])
spreads['away_team_abb'] = spreads['away_team_abb'].map(new_acc).fillna(spreads['away_team_abb']) 

spreads = spreads[spreads['schedule_season']>=2014]
spreads = spreads[['schedule_season','schedule_week','team_home_abb','score_home','score_away','away_team_abb','team_favorite_id','spread_favorite','over_under_line','starting_spread', 'Total Score Open',
       'fav_team_open', 'fav_team_cur', 'remain_fav', 'spread_movement','ou_movement', 'strong_movement', 'fav_team_stronger']]
spreads['team_home_abb'] = spreads['team_home_abb'].astype(str)
spreads['team_favorite_id'] = spreads['team_favorite_id'].astype(str)
spreads['over_under_line'] = spreads['over_under_line'].astype(float)


def fav_spread(nData):
    if nData['team_home_abb'] == nData['team_favorite_id']:
        return nData['spread_favorite']
    elif nData['away_team_abb'] == nData['team_favorite_id']:
        return nData['spread_favorite']
    else:
        pass
spreads['fav_spread'] = spreads.apply(lambda nData: fav_spread(nData), axis=1)

def nonfav_spread(nData):
    if nData['team_home_abb'] != nData['team_favorite_id']:
        return nData['team_home_abb']
    elif nData['away_team_abb'] != nData['team_favorite_id']:
        return nData['away_team_abb']
    else:
        pass
spreads['team_notfav_id'] = spreads.apply(lambda nData: nonfav_spread(nData), axis=1)

def cover_or_not(nData):    
    if nData['team_home_abb'] == nData['team_favorite_id']:
        if ((nData['score_home']-nData['score_away']))+nData['spread_favorite'] > 0:
            return 'Cover'
        elif ((nData['score_home']-nData['score_away']))+nData['spread_favorite'] == 0:            
            return 'Push'       
        else:            
            return 'No Cover'
    elif nData['away_team_abb'] == nData['team_favorite_id']:        
        if ((nData['score_away']-nData['score_home']))+nData['spread_favorite'] > 0:            
            return 'Cover'        
        elif ((nData['score_away']-nData['score_home']))+nData['spread_favorite'] == 0:            
            return 'Push'        
        else:            
            return 'No Cover'
spreads['fav_cover'] = spreads.apply(lambda nData: cover_or_not(nData), axis=1)

def OU_or_not(nData):    
    if (nData['score_home']+nData['score_away']) > nData['over_under_line']:        
        return 'Over'    
    elif (nData['score_home']-nData['score_away']) == nData['over_under_line']:        
        return 'Push'    
    else:        
        return 'Under'
spreads['over_under_result'] = spreads.apply(lambda nData: OU_or_not(nData), axis=1)



spreads['schedule_season'] = spreads['schedule_season'].apply(int)    
spreads['schedule_week'] = spreads['schedule_week'].apply(int)  
data = spreads.sort_values(by=["team_home_abb","schedule_season","schedule_week"], ascending=[True, True, True])

def clean_spreads(df):
    ##  basic scrubbing to clean data ##    
    df['schedule_season'] = df['schedule_season'].apply(str)    
    df['schedule_week'] = df['schedule_week'].apply(str)        
    df=df.apply(lambda x: x.astype(str).str.lower())    
    #df['schedule_week']=df['schedule_week'].astype(str).str[:-2].astype(object)    
    #df['schedule_season'] = df['schedule_season'].astype(str).str[:-2].astype(object)  
    df['team_home_abb'] = df['team_home_abb'].map(new_acc).fillna(df['team_home_abb'])
    df['away_team_abb'] = df['away_team_abb'].map(new_acc).fillna(df['away_team_abb'])
    
    ##  create our unique ids  ##
    df.insert(0, "home_matchup_id", (df['team_home_abb']+'vs'+df['away_team_abb']+'_'+df['schedule_season']+'_'+df['schedule_week']))
    df.insert(1, "away_matchup_id", (df['away_team_abb']+'@'+df['team_home_abb']+'_'+df['schedule_season']+'_'+df['schedule_week']))
    df.insert(2, "home_id", (df['team_home_abb']+'_'+df['schedule_season']+'_'+df['schedule_week']))
    df.insert(3, "away_id", (df['away_team_abb']+'_'+df['schedule_season']+'_'+df['schedule_week']))
    return df
    
data = clean_spreads(data)

data = pd.merge(data, weather, on='away_matchup_id', how='left')


sh = data
sa = data

sh = sh.rename(columns={'home_id':'team_id'})
sh.drop('away_id', axis=1, inplace=True)

sa = sa.rename(columns={'away_id':'team_id'})
sa.drop('home_id', axis=1, inplace=True)

spread_comb = pd.concat([sh, sa], axis=0).reset_index(drop=True)

spread_comb['team_abb'] = spread_comb['team_id'].astype(str).str[:3]
spread_comb['team_abb'] = spread_comb['team_abb'].str.replace("_","")

def hora1(nData):
    if nData['team_favorite_id'] == nData['team_home_abb']:
        return 1
    elif nData['team_notfav_id'] == nData['team_home_abb']:
        return 1
    else:
        return 0
spread_comb['homeoraway'] = spread_comb.apply(lambda nData: hora1(nData), axis=1)

def hora(nData):
    if nData['team_favorite_id'] == nData['away_team_abb']:
        return 1
    else:
        return 0
spread_comb['fav_homeoraway'] = spread_comb.apply(lambda nData: hora(nData), axis=1)
#sh['fav_homeoraway'] = sh.apply(lambda nData: hora(nData), axis=1)

def ws(nData):
    if (nData['fav_homeoraway'] == 0) & (nData['fav_cover'] == 'cover'):
        return 1
    elif (nData['fav_homeoraway'] == 1) & (nData['fav_cover'] == 'no cover'):
        return 1
    else:
        return 0

def ls(nData):    
    if (nData['fav_homeoraway'] == 0) & (nData['fav_cover'] == 'no cover'):
        return 1
    elif (nData['fav_homeoraway'] == 1) & (nData['fav_cover'] == 'cover'):
        return 1
    else:
        return 0

spread_comb['ats_w'] = spread_comb.apply(lambda nData: ws(nData), axis=1)
spread_comb['ats_l'] = spread_comb.apply(lambda nData: ls(nData), axis=1)


# In[9]:

spread_comb.columns


# ## Create Football Outsiders rolling function

# In[10]:



def rolling_fo(data=None, roll_value=None, roll_type=None):
    
    """
        Args:
        data: input pandas dataframe to be rolled
        roll_value: input the number, default is three ## we will need to modify the function if we want more ##
        roll_type: 'mean','std', or 'var' are the only options at the point
        ## assign mean for a given team & year as opposed to the entire dataset
   
    """
    
    data = data.sort_values(by=["team","year","week"], ascending=[True, True, True])
    #data=data.fillna(data.mean())
    num_cols = ['total_dvoa', 'off_dvoa','off_pass_dvoa', 'off_rush_dvoa', 'def_dvoa', 'def_pass_dvoa','def_rush_dvoa', 'special_teams_dvoa']
    ids = data[['team_id', 'year', 'team', 'week', 'opp']].reset_index(drop=True)
   
    if roll_type == 'mean':
        roll3 = data.groupby(['team','year'])[num_cols].apply(lambda x : x.shift().rolling(roll_value).mean())
        roll2 = data.groupby(['team','year'])[num_cols].apply(lambda x : x.shift().rolling(roll_value-1).mean())
        roll1 = data.groupby(['team','year'])[num_cols].apply(lambda x : x.shift().rolling(roll_value-2).mean())
        roll3 = pd.DataFrame(roll3.combine_first(roll2).combine_first(roll1)).reset_index(drop=True)
        df = pd.concat([ids, roll3], axis=1)
    return df


# ## Read in historic weekly football outsiders data and create the current week rows for each team

# In[11]:

##Create the current weeks fo team_ids/rows to roll into##
fo_data = pd.read_csv("./current_data/week_"+cur_week_str+"/fo_weekly_update.csv")
fo_data_new = fo_data[~fo_data['week'].isnull()]
fo_data_new['week']=fo_data_new['week'].apply(int)
fo_data_new=fo_data_new.drop_duplicates(subset=['team','year'], keep='last').assign(week=cur_week_str)
if cur_week_int >= 10:
    fo_data_new['team_id'] = fo_data_new['team_id'].str[:-2]
    fo_data_new['team_id']=fo_data_new['team_id'].str.replace("2022_", str("2022_"+cur_week_str))
else:
    fo_data_new['team_id'] = fo_data_new['team_id'].str[:-1]
    fo_data_new['team_id']=fo_data_new['team_id'].str.replace("2022_", str("2022_"+cur_week_str))
    
fo_data_new = fo_data_new.sort_values(by=["team","week"], ascending=[True, False])
fo_data_new[fo_data_new.columns[4:]] = np.nan
fo_data_new.head().T



# ## Now read in the historic FO data and concat all of them together for our rolling function

# In[12]:

fo_data_2022 = pd.read_csv("./historic_data/fo_data/fo_weekly_hist.csv")
fo_data = pd.read_csv("./current_data/week_"+cur_week_str+"/fo_weekly_update.csv")

fo = pd.concat([fo_data_2022, fo_data, fo_data_new], axis=0).reset_index(drop=True)

fo['team'] = fo['team'].map(new_acc).fillna(fo['team'])
fo['opp'] = fo['opp'].map(new_acc).fillna(fo['opp']) 

fo['team'] = fo['team'].map(utils.cleaning_dicts.clean_team_fo).fillna(fo['team'])
fo['opp'] = fo['opp'].map(utils.cleaning_dicts.clean_team_fo).fillna(fo['opp'])

##combine our current season fo data with the new week 4 rows we just made##
fo_roll = rolling_fo(data=fo, roll_value=3, roll_type='mean')
fo_roll = fo_roll.rename(columns={'team_id': 'unique_team_id'})

fo_roll['unique_team_id']=fo_roll['unique_team_id'].str.replace('sd_','lac_')
fo_roll['unique_team_id']=fo_roll['unique_team_id'].str.replace('oak_','lv_')
fo_roll.drop(['year','team','week','opp'], axis=1, inplace=True)

fo_roll.head().T


# ### PFF team_game_summaries (tgs) clean and create current week rows

# In[13]:

tgs_new_week = pd.read_csv("./current_data/week_"+cur_week_str+"/team_game_summaries_w"+cur_week_str+".csv")

tgs_new_week = tgs_new_week[~tgs_new_week['week'].isnull()]
tgs_new_week=tgs_new_week.drop_duplicates(subset=['team','year'], keep='last').assign(week=cur_week_str)

tgs_new_week['team_name'] = tgs_new_week['team'].map(utils.cleaning_dicts.clean_team_pff_full).fillna(tgs_new_week['team'])
tgs_new_week['opponent_name'] = tgs_new_week['opponent'].map(utils.cleaning_dicts.clean_team_pff_opp).fillna(tgs_new_week['opponent'])

tgs_new_week['home_or_away']=tgs_new_week['home_or_away'].astype(str)

def home_team(nData):
    if str('@') in nData['home_or_away']:
        return nData['opponent_name']
    else:
        return nData['team_name']

tgs_new_week['home_team'] = tgs_new_week.apply(lambda nData: home_team(nData), axis=1)

def away_team(nData):
    if str('@') in nData['home_or_away']:
        return nData['team_name']
    else:
        return nData['opponent_name']
    
tgs_new_week['away_team'] = tgs_new_week.apply(lambda nData: away_team(nData), axis=1)

def clean_pff_team_summ(df):
##  basic scrubbing to clean data ##

    df['year'] = df['year'].astype(str)
    df['week'] = df['week'].astype(str)
    df['home_or_away']=np.where(df['home_or_away'] == "@", 1, 0)
    df['wl_int'] = np.where(df['wl'] == "W", 1, 0)
    df=df.replace('-','', regex=True)
    df=df.replace(' ','', regex=True)
    
    df['team_name'] = df['team_name'].map(new_acc).fillna(df['team_name'])
    df['opponent_name'] = df['opponent_name'].map(new_acc).fillna(df['opponent_name'])


    ##  create our unique ids  ##
    df.insert(0, "unique_team_id", (df['team_name']+'_'+df['year']+'_'+df['week']))
    df.insert(1, "team_id_impute", (df['team_name']+'_'+df['year']))
    df.insert(2, "opponent_id", (df['opponent_name']+'_'+df['year']+'_'+df['week']))
    df.insert(3, "home_matchup_id", (df['home_team']+'vs'+df['away_team']+'_'+df['year']+'_'+df['week']))
    
    return df

tgs_new_week = clean_pff_team_summ(tgs_new_week)
tgs_new_week['wl_int'] = ''
tgs_new_week = tgs_new_week.sort_values(by=["team_name","week"], ascending=[True, False])


# ## Now read in historic tgs data and clean

# In[14]:

tgs_data_2022 = pd.read_csv("./historic_data/pff_data/team_game_summaries_historic.csv")
tgs_data_cur = pd.read_csv("./current_data/week_"+cur_week_str+"/team_game_summaries_w"+cur_week_str+".csv")
tgs = pd.concat([tgs_data_2022, tgs_data_cur], axis=0)

tgs = tgs[tgs['year'] >= 2014]


tgs['team_name'] = tgs['team'].map(utils.cleaning_dicts.clean_team_pff_full).fillna(tgs['team'])
tgs['opponent_name'] = tgs['opponent'].map(utils.cleaning_dicts.clean_team_pff_opp).fillna(tgs['opponent'])

##adding just incase accronyms have changed
tgs['team_name'] = tgs['team_name'].map(new_acc).fillna(tgs['team_name'])
tgs['opponent_name'] = tgs['opponent_name'].map(new_acc).fillna(tgs['opponent_name']) 

tgs['home_or_away']=tgs['home_or_away'].astype(str)

def home_team(nData):
    if str('@') in nData['home_or_away']:
        return nData['opponent_name']
    else:
        return nData['team_name']

tgs['home_team'] = tgs.apply(lambda nData: home_team(nData), axis=1)

def away_team(nData):
    if str('@') in nData['home_or_away']:
        return nData['team_name']
    else:
        return nData['opponent_name']
    
tgs['away_team'] = tgs.apply(lambda nData: away_team(nData), axis=1)

def clean_pff_team_summ(df):
##  basic scrubbing to clean data ##

    df['year'] = df['year'].astype(str)
    df['week'] = df['week'].astype(str)
    df['home_or_away']=np.where(df['home_or_away'] == "@", 1, 0)
    df['wl_int'] = np.where(df['wl'] == "W", 1, 0)
    df=df.replace('-','', regex=True)
    df=df.replace(' ','', regex=True)
    
    df['team_name'] = df['team_name'].map(new_acc).fillna(df['team_name'])
    df['opponent_name'] = df['opponent_name'].map(new_acc).fillna(df['opponent_name'])


    ##  create our unique ids  ##
    df.insert(0, "unique_team_id", (df['team_name']+'_'+df['year']+'_'+df['week']))
    df.insert(1, "team_id_impute", (df['team_name']+'_'+df['year']))
    df.insert(2, "opponent_id", (df['opponent_name']+'_'+df['year']+'_'+df['week']))
    df.insert(3, "home_matchup_id", (df['home_team']+'vs'+df['away_team']+'_'+df['year']+'_'+df['week']))
    
    ##Impute missing special teams data added after 2014##
    df = df.apply(pd.to_numeric, errors='ignore')
    df.reset_index(inplace=True, drop=True)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols]= df.groupby(df['team_name'])[num_cols].fillna(df.mean()).reset_index(level=0, drop=True)
   
    return df


   
tgs_clean = clean_pff_team_summ(tgs)


tgs_clean = pd.concat([tgs_clean, tgs_new_week], axis=0).reset_index(drop=True)
tgs_clean['year']=tgs_clean['year'].apply(int)
tgs_clean['week']=tgs_clean['week'].apply(int)
tgs_clean['special_teams']=tgs_clean['special_teams'].apply(float)
tgs_clean = tgs_clean.sort_values(by=["team_name","year","week"], ascending=[True, True, True])

tgs_clean = tgs_clean[['unique_team_id','team_id_impute', 'home_matchup_id','opponent_id','wl','pf','pa','team_name','opponent_name','year','week','overall_performance', 'offense', 'pass',
       'pass_blocking', 'receiving', 'rushing', 'run_blocking', 'defense',
       'rush_defense', 'tackling', 'pass_rush', 'coverage', 'special_teams']]

tgs_clean.drop_duplicates(inplace=True)

#tgs_clean.to_csv("C:/football2022/tgsv1.csv")




# In[15]:

tgs_clean.sort_values(['team_name','week','year'], ascending=True, inplace=True)

alpha=.5

tm1=tgs_clean.copy()
tm1=tm1.add_prefix('tm1_')

tm2=tgs_clean.copy()
tm2=tm2.add_prefix('tm2_')

opps1=pd.merge(tm1,tm2,left_on='tm1_opponent_id', right_on='tm2_unique_team_id', how='left')  


def adjust_score(datain,tm1_field,tm2_field):
#    datain['a0_'+tm1_field+'_'+tm2_field]=( .4*(datain[tm1_field]-datain[tm2_field]) + datain[tm2_field] ) / .4
    datain['a1_'+tm1_field+'_'+tm2_field]=( .5*(datain[tm1_field]-datain[tm2_field]) + datain[tm2_field] ) / .5
    datain['a2_'+tm1_field+'_'+tm2_field]=datain[tm1_field] - datain[tm2_field]
    datain['a0_'+tm1_field+'_'+tm2_field]=datain[tm1_field] / datain[tm2_field]
 
perfs=['overall_performance', 'offense', 'pass', 'pass_blocking',
       'receiving', 'rushing', 'run_blocking', 'defense', 'rush_defense',
       'tackling', 'pass_rush', 'coverage', 'special_teams' ]

capture_fields=[]
for p in perfs:
    for q in perfs:
        adjust_score(opps1,'tm1_'+p,'tm2_'+q)
        
        capture_fields.append(['a0_'+'tm1_'+p+'_'+'tm2_'+q,'a1_'+'tm1_'+p+'_'+'tm2_'+q,'a2_'+'tm1_'+p+'_'+'tm2_'+q])

capt_flat_list = ['tm1_home_matchup_id']+[item for sublist in capture_fields for item in sublist]
keepies=[item for sublist in capture_fields for item in sublist]
opps2=opps1[capt_flat_list]


tgs_clean=pd.merge(tgs_clean,opps2,left_on='home_matchup_id', right_on='tm1_home_matchup_id', how='left') 



    
    
    









# ### Create tgs rolling mean function and combine all tgs datasets together and pass through the rolling function

# In[16]:

def rolling_tgs(data=None, roll_value=None, roll_type=None):
    
    """
        Args:
        data: input pandas dataframe to be rolled
        roll_value: input the number, default is three ## we will need to modify the function if we want more ##
        roll_type: 'mean','std', or 'var' are the only options at the point
        ## assign mean for a given team & year as opposed to the entire dataset
   
    """
    
    data = data.sort_values(by=["team_name","year","week"], ascending=[True, True, True])
    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    ids = pd.DataFrame(data.select_dtypes(exclude=[np.number])).reset_index(drop=True)
   
    if roll_type == 'mean':
        roll3 = data.groupby(data['team_id_impute'])[num_cols].apply(lambda x : x.shift().rolling(roll_value).mean())
        roll2 = data.groupby(data['team_id_impute'])[num_cols].apply(lambda x : x.shift().rolling(roll_value-1).mean())
        roll1 = data.groupby(data['team_id_impute'])[num_cols].apply(lambda x : x.shift().rolling(roll_value-2).mean())
        roll3 = pd.DataFrame(roll3.combine_first(roll2).combine_first(roll1)).reset_index(drop=True)
        df = pd.concat([ids, roll3], axis=1)
    return df
        
tgs_roll = rolling_tgs(data=tgs_clean, roll_value=3, roll_type='mean')

tgs_roll = tgs_roll[['unique_team_id','wl','pf','pa','overall_performance', 'offense', 'pass',
       'pass_blocking', 'receiving', 'rushing', 'run_blocking', 'defense',
       'rush_defense', 'tackling', 'pass_rush', 'coverage', 'special_teams']+keepies]

tgs_roll = tgs_roll.rename(columns={c: c+'_tgs' for c in tgs_roll.columns if c not in ['unique_team_id','wl','pf','pa']})

tgs_roll.rename(columns={'unique_team_id_tgs_pff':'unique_team_id'}, inplace=True)


# ## Read in all the pff current week datasets and prep for rolling function

# In[17]:

passing_depth_new = pd.read_csv("./current_data/week_"+cur_week_str+"/passing_depth_new_pp_w"+cur_week_str+".csv")
passing_allowed_pressure_new = pd.read_csv('./current_data/week_'+cur_week_str+'/passing_allowed_pressure_new_pp_w'+cur_week_str+".csv")
passing_pressure_new = pd.read_csv('./current_data/week_'+cur_week_str+'/passing_pressure_new_pp_w'+cur_week_str+".csv")
passing_concept_new = pd.read_csv('./current_data/week_'+cur_week_str+'/passing_concept_new_pp_w'+cur_week_str+".csv")
time_in_pocket_new = pd.read_csv('./current_data/week_'+cur_week_str+'/time_in_pocket_new_pp_w'+cur_week_str+".csv")
passing_summ_conc_new = pd.read_csv('./current_data/week_'+cur_week_str+'/passing_summ_conc_new_pp_w'+cur_week_str+".csv")


rec_summ_conc_new = pd.read_csv('./current_data/week_'+cur_week_str+'/rec_summ_conc_pp_w'+cur_week_str+".csv")
receiving_concept_new = pd.read_csv('./current_data/week_'+cur_week_str+'/receiving_concept_pp_w'+cur_week_str+".csv")
receiving_depth_new = pd.read_csv('./current_data/week_'+cur_week_str+'/receiving_depth_pp_w'+cur_week_str+".csv")
receiving_scheme_new = pd.read_csv('./current_data/week_'+cur_week_str+'/receiving_scheme_pp_w'+cur_week_str+".csv")

rush_summ_conc_new = pd.read_csv('./current_data/week_'+cur_week_str+'/rush_summ_conc_pp_w'+cur_week_str+".csv")

block_summ_conc_new = pd.read_csv('./current_data/week_'+cur_week_str+'/block_summ_conc_pp_w'+cur_week_str+".csv")
offense_pass_blocking_new = pd.read_csv('./current_data/week_'+cur_week_str+'/offense_pass_blocking_pp_w'+cur_week_str+".csv")
offense_run_blocking_new = pd.read_csv('./current_data/week_'+cur_week_str+'/offense_run_blocking_pp_w'+cur_week_str+".csv")

def_summ_conc_new = pd.read_csv('./current_data/week_'+cur_week_str+'/def_summ_conc_pp_w'+cur_week_str+".csv")
pass_rush_summary_new = pd.read_csv('./current_data/week_'+cur_week_str+'/pass_rush_summary_pp_w'+cur_week_str+".csv")
run_defense_summary_new = pd.read_csv('./current_data/week_'+cur_week_str+'/run_defense_summary_pp_w'+cur_week_str+".csv")
defense_coverage_scheme_new = pd.read_csv('./current_data/week_'+cur_week_str+'/defense_coverage_scheme_pp_w'+cur_week_str+".csv")
defense_coverage_summary_new = pd.read_csv('./current_data/week_'+cur_week_str+'/defense_coverage_summary_pp_w'+cur_week_str+".csv")
slot_coverage_new = pd.read_csv('./current_data/week_'+cur_week_str+'/slot_coverage_pp_w'+cur_week_str+".csv")

st_kickers_new = pd.read_csv('./current_data/week_'+cur_week_str+'/st_kickers_pp_w'+cur_week_str+".csv")
st_punters_new = pd.read_csv('./current_data/week_'+cur_week_str+'/st_punters_no_inj_pp_w'+cur_week_str+".csv")


passing_depth_new['week'] = cur_week_str 
passing_allowed_pressure_new['week'] = cur_week_str 
passing_pressure_new['week'] = cur_week_str 
passing_concept_new['week'] = cur_week_str 
time_in_pocket_new['week'] = cur_week_str 
passing_summ_conc_new['week'] = cur_week_str 
rec_summ_conc_new['week'] = cur_week_str 
receiving_concept_new['week'] = cur_week_str
receiving_depth_new['week'] = cur_week_str 
receiving_scheme_new['week'] = cur_week_str 
rush_summ_conc_new['week'] = cur_week_str
block_summ_conc_new['week'] = cur_week_str 
offense_pass_blocking_new['week'] = cur_week_str 
offense_run_blocking_new['week'] = cur_week_str 
def_summ_conc_new['week'] = cur_week_str 
pass_rush_summary_new['week'] = cur_week_str 
run_defense_summary_new['week'] = cur_week_str 
defense_coverage_scheme_new['week'] = cur_week_str 
defense_coverage_summary_new['week'] = cur_week_str 
slot_coverage_new['week'] = cur_week_str 
st_kickers_new['week'] = cur_week_str 
st_punters_new['week'] = cur_week_str


# In[18]:

def_summ_conc_new.tail()


# ### Add the prefixes like we did for the pff datasets above

# In[19]:

####################################################################################
								###   add prefixes ###
####################################################################################	

def create_prefix(prefix=None, df=None):
    id = df[['p_id','player_team_id','unique_team_id','team_id_impute','player','numeric_id','position','team_name','year','week']]
    temp = df.drop(['p_id','player','player_team_id','unique_team_id','plyr_number','player','team_id_impute','numeric_id','position','team_name','unique_team_id','numeric_id','position','team_name','year','week','plyr_number'], axis=1)
    temp = temp.add_prefix(prefix)
    id = pd.concat([id, temp], axis=1)
    return id

def id_prefix(prefix=None, df=None):
    id = df[['p_id','player','player_team_id','unique_team_id','team_id_impute','numeric_id','position','team_name','year','week']]
    temp = df.drop(['p_id','player','player_team_id','unique_team_id','plyr_number','team_id_impute','numeric_id','position','team_name','year','week','plyr_number'], axis=1)
    temp = temp.add_prefix(prefix)
    id = pd.concat([id, temp], axis=1)
    return id

passing_summ_conc_new = id_prefix(prefix="pass_summary_", df=passing_summ_conc_new)
rush_summ_conc_new  = id_prefix(prefix="rush_summary_", df=rush_summ_conc_new)
rec_summ_conc_new  = id_prefix(prefix="rec_summary_", df=rec_summ_conc_new)
block_summ_conc_new  = id_prefix(prefix="block_summary_", df=block_summ_conc_new)
def_summ_conc_new  = id_prefix(prefix="def_summary_", df=def_summ_conc_new)
st_kickers_new  = id_prefix(prefix="kicking_", df=st_kickers_new)
st_punters_new  = id_prefix(prefix="punting_", df=st_punters_new)


passing_depth_new = create_prefix(prefix="pass_depth_", df=passing_depth_new)
passing_allowed_pressure_new = create_prefix(prefix="pressure_source_", df=passing_allowed_pressure_new)
passing_pressure_new = create_prefix(prefix="pass_under_pressure_", df=passing_pressure_new)
passing_concept_new = create_prefix(prefix="pass_concept_", df=passing_concept_new)
time_in_pocket_new = create_prefix(prefix="pass_time_", df=time_in_pocket_new)


receiving_concept_new = create_prefix(prefix="rec_concept_", df=receiving_concept_new)
receiving_depth_new = create_prefix(prefix="rec_depth_", df=receiving_depth_new)
receiving_scheme_new = create_prefix(prefix="rec_scheme_", df=receiving_scheme_new)

offense_pass_blocking_new = create_prefix(prefix="pass_block_", df=offense_pass_blocking_new)
offense_run_blocking_new = create_prefix(prefix="run_block_", df=offense_run_blocking_new)


pass_rush_summary_new = create_prefix(prefix="pass_rush_", df=pass_rush_summary_new)
run_defense_summary_new = create_prefix(prefix="run_defense_", df=run_defense_summary_new)
defense_coverage_scheme_new = create_prefix(prefix="def_coverage_scheme_", df=defense_coverage_scheme_new)
defense_coverage_summary_new = create_prefix(prefix="def_coverage_summary_", df=defense_coverage_summary_new)
slot_coverage_new= create_prefix(prefix="def_slot_coverage_", df=slot_coverage_new)


# In[20]:

def_summ_conc_new.tail()


# ### Bring the historic and new player pool data together

# In[21]:

passing_depth = pd.concat([passing_depth, passing_depth_new], axis=0)
passing_allowed_pressure = pd.concat([passing_allowed_pressure, passing_allowed_pressure_new], axis=0)
passing_pressure = pd.concat([passing_pressure, passing_pressure_new], axis=0)
passing_concept = pd.concat([passing_concept, passing_concept_new], axis=0)
time_in_pocket = pd.concat([time_in_pocket, time_in_pocket_new], axis=0)
passing_summ_conc = pd.concat([passing_summ_conc, passing_summ_conc_new], axis=0)


rec_summ_conc = pd.concat([rec_summ_conc, rec_summ_conc_new], axis=0)
receiving_concept = pd.concat([receiving_concept, receiving_concept_new], axis=0)
receiving_depth = pd.concat([receiving_depth, receiving_depth_new], axis=0)
receiving_scheme = pd.concat([receiving_scheme, receiving_scheme_new], axis=0)

rush_summ_conc = pd.concat([rush_summ_conc, rush_summ_conc_new], axis=0)

block_summ_conc = pd.concat([block_summ_conc, block_summ_conc_new], axis=0)
offense_pass_blocking = pd.concat([offense_pass_blocking, offense_pass_blocking_new], axis=0)
offense_run_blocking = pd.concat([offense_run_blocking, offense_run_blocking_new], axis=0)

def_summ_conc = pd.concat([def_summ_conc, def_summ_conc_new], axis=0)
pass_rush_summary = pd.concat([pass_rush_summary, pass_rush_summary_new], axis=0)
run_defense_summary = pd.concat([run_defense_summary, run_defense_summary_new], axis=0)
defense_coverage_scheme = pd.concat([defense_coverage_scheme, defense_coverage_scheme_new], axis=0)
defense_coverage_summary = pd.concat([defense_coverage_summary, defense_coverage_summary_new], axis=0)
slot_coverage = pd.concat([slot_coverage, slot_coverage_new], axis=0)

st_kickers = pd.concat([st_kickers, st_kickers_new], axis=0)
st_punters = pd.concat([st_punters, st_punters_new], axis=0)


### after the concat cell ###
passing_depth.drop_duplicates(subset='p_id', inplace=True)
passing_allowed_pressure.drop_duplicates(subset='p_id', inplace=True)
passing_pressure.drop_duplicates(subset='p_id', inplace=True)
passing_concept.drop_duplicates(subset='p_id', inplace=True)
time_in_pocket.drop_duplicates(subset='p_id', inplace=True)
passing_summ_conc.drop_duplicates(subset='p_id', inplace=True)


rec_summ_conc.drop_duplicates(subset='p_id', inplace=True)
receiving_concept.drop_duplicates(subset='p_id', inplace=True)
receiving_depth.drop_duplicates(subset='p_id', inplace=True)
receiving_scheme.drop_duplicates(subset='p_id', inplace=True)

rush_summ_conc.drop_duplicates(subset='p_id', inplace=True)

block_summ_conc.drop_duplicates(subset='p_id', inplace=True)
offense_pass_blocking.drop_duplicates(subset='p_id', inplace=True)
offense_run_blocking.drop_duplicates(subset='p_id', inplace=True)

def_summ_conc.drop_duplicates(subset='p_id', inplace=True)
pass_rush_summary.drop_duplicates(subset='p_id', inplace=True)
run_defense_summary.drop_duplicates(subset='p_id', inplace=True)
defense_coverage_scheme.drop_duplicates(subset='p_id', inplace=True)
defense_coverage_summary.drop_duplicates(subset='p_id', inplace=True)
slot_coverage.drop_duplicates(subset='p_id', inplace=True)

st_kickers.drop_duplicates(subset='p_id', inplace=True)
st_punters.drop_duplicates(subset='p_id', inplace=True)


# In[22]:

inj=pd.read_csv('./misc_files/pfr_injury.csv')

rec = rec_summ_conc
rec=rec[['p_id','player_team_id','numeric_id', 'position', 'team_name', 'year', 'week','rec_summary_grades_offense', 'rec_summary_pass_plays']]
rec.columns= ['p_id','player_team_id','numeric_id', 'position', 'team_name', 'year', 'week','pff_grade', 'plays']

rush = rush_summ_conc
rush=rush[['p_id','player_team_id','numeric_id', 'position', 'team_name', 'year', 'week','rush_summary_grades_offense', 'rush_summary_run_plays']]
rush.columns= ['p_id','player_team_id','numeric_id', 'position', 'team_name', 'year', 'week','pff_grade', 'plays']

blk = block_summ_conc
blk=blk[['p_id','player_team_id','numeric_id', 'position', 'team_name', 'year', 'week','block_summary_grades_offense', 'block_summary_snap_counts_block']]
blk.columns= ['p_id','player_team_id','numeric_id', 'position', 'team_name', 'year', 'week','pff_grade', 'plays']

defns = def_summ_conc
defns=defns[['p_id','player_team_id','numeric_id', 'position', 'team_name', 'year', 'week','def_summary_grades_defense', 'def_summary_snap_counts_defense']]
defns.columns= ['p_id','player_team_id','numeric_id', 'position', 'team_name', 'year', 'week','pff_grade', 'plays']

passing = passing_summ_conc
passing=passing[['p_id','player_team_id','numeric_id', 'position', 'team_name', 'year', 'week','pass_summary_grades_offense', 'pass_summary_passing_snaps']]
passing.columns= ['p_id','player_team_id','numeric_id', 'position', 'team_name', 'year', 'week','pff_grade', 'plays']

comb = pd.concat([rec,rush,blk,defns,passing], axis=0)
comb = comb.sort_values(by=["player_team_id","team_name","year","week","plays"], ascending=[True, True, True, True,True])
comb_grp=comb.groupby('p_id').agg({'pff_grade':'mean', 'plays':'sum'}).reset_index(drop=False)
comb.drop(['pff_grade','plays'], axis=1, inplace=True)
comb = pd.merge(comb_grp, comb, on='p_id', how='left')

pos_dict={
'di':'dl',
'ed':'dl',
'cb':'db',
's':'db',
't':'ol',
'g':'ol',
'fb':'hb'}

comb['position'] = comb['position'].map(pos_dict).fillna(comb['position'])

comb=comb.groupby(['player_team_id','position']).agg({'pff_grade':'mean', 'plays':'mean'}).reset_index(drop=False)

comb=comb[['player_team_id','position','pff_grade','plays']]


# In[23]:

inj = pd.merge(inj, comb, left_on='player_id', right_on='player_team_id', how='left')
inj.head()


# In[24]:


tmp=[]

for i in inj['unique_id']:
    t = i.split('_', 1)[1]
    tmp.append(t)
new = pd.DataFrame(tmp, columns=['team_id'])
inj.reset_index(drop=True, inplace=True)
inj = pd.concat([new, inj], axis=1)

inj = inj.loc[inj['position'].notnull()]
g = inj.groupby(['team_id','position']).mean().reset_index(drop=False)
d = {'score':'inj_count', 'pff_grade':'inj_grade','plays':'inj_plays'}
g=inj.groupby(['team_id','position']).agg({'score':'count', 'pff_grade':'mean','plays':'mean'}).rename(columns=d).reset_index(drop=False)

inj_final = g.pivot_table(['inj_count', 'inj_plays','inj_grade'], ['team_id'], 'position')

inj_final.columns = ['_'.join(col) for col in inj_final.columns]
inj_final=inj_final.fillna(0).reset_index(drop=False)


# In[25]:

inj_final=inj_final.rename(columns={"team_id": "unique_team_id"})
spread_id = spread_comb[['team_id']]
spread_id.columns = ['unique_team_id']
inj_final = pd.concat([inj_final, spread_id], axis=0)
inj_final.drop_duplicates(subset='unique_team_id', keep="first")
inj_final=inj_final.fillna(0)
inj_final.tail()


# In[26]:

inj_final.head()


# ## Create rolling function and pass pff datasets through

# In[27]:

get_ipython().run_cell_magic('time', '', '\ndef rolling(data=None, roll_value=None, roll_type=None):\n    \n    """\n        Args:\n        data: input pandas dataframe to be rolled\n        roll_value: input the number, default is three ## we will need to modify the function if we want more ##\n        roll_type: \'mean\',\'std\', or \'var\' are the only options at the point\n        ## assign mean for a given team & year as opposed to the entire dataset\n   \n    """\n    \n    data = data.sort_values(by=["player","team_name","year","week"], ascending=[True, True, True, True])\n    data[\'week\']=data[\'week\'].apply(str)\n    data[\'year\']=data[\'year\'].apply(str)\n    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()\n    ids = pd.DataFrame(data.select_dtypes(exclude=[np.number])).reset_index(drop=True)\n   \n    if roll_type == \'mean\':\n        #roll5 = data.groupby(data[\'player_id\'])[num_cols].apply(lambda x : x.shift().rolling(roll_value).mean())\n        #roll4 = data.groupby(data[\'player_id\'])[num_cols].apply(lambda x : x.shift().rolling(roll_value).mean())\n        roll3 = data.groupby(data["player_team_id"])[num_cols].apply(lambda x : x.shift().rolling(roll_value).mean())\n        roll2 = data.groupby(data["player_team_id"])[num_cols].apply(lambda x : x.shift().rolling(roll_value-1).mean())\n        roll1 = data.groupby(data["player_team_id"])[num_cols].apply(lambda x : x.shift().rolling(roll_value-2).mean())\n        roll3 = pd.DataFrame(roll3.combine_first(roll2).combine_first(roll1)).reset_index(drop=True)\n        df = pd.concat([ids, roll3], axis=1)\n    return df\n   \npassing_depth_roll = rolling(data=passing_depth, roll_value=3, roll_type=\'mean\')\npassing_allowed_pressure_roll = rolling(data=passing_allowed_pressure, roll_value=3, roll_type=\'mean\')\npassing_pressure_roll = rolling(data=passing_pressure, roll_value=3, roll_type=\'mean\')\npassing_concept_roll = rolling(data=passing_concept, roll_value=3, roll_type=\'mean\')\ntime_in_pocket_roll = rolling(data=time_in_pocket, roll_value=3, roll_type=\'mean\')\npassing_summ_conc_roll = rolling(data=passing_summ_conc, roll_value=3, roll_type=\'mean\')\n\n\nrec_summ_conc_roll = rolling(data=rec_summ_conc, roll_value=3, roll_type=\'mean\')\nreceiving_concept_roll =rolling(data=receiving_concept, roll_value=3, roll_type=\'mean\')\nreceiving_depth_roll = rolling(data=receiving_depth, roll_value=3, roll_type=\'mean\')\nreceiving_scheme_roll = rolling(data=receiving_scheme, roll_value=3, roll_type=\'mean\')\n\nrush_summ_conc_roll = rolling(data=rush_summ_conc, roll_value=3, roll_type=\'mean\')\n\nblock_summ_conc_roll = rolling(data=block_summ_conc, roll_value=3, roll_type=\'mean\')\noffense_pass_blocking_roll = rolling(data=offense_pass_blocking, roll_value=3, roll_type=\'mean\')\noffense_run_blocking_roll = rolling(data=offense_run_blocking, roll_value=3, roll_type=\'mean\')\n\ndef_summ_conc_roll = rolling(data=def_summ_conc, roll_value=3, roll_type=\'mean\')\npass_rush_summary_roll = rolling(data=pass_rush_summary, roll_value=3, roll_type=\'mean\')\nrun_defense_summary_roll = rolling(data=run_defense_summary, roll_value=3, roll_type=\'mean\')\ndefense_coverage_scheme_roll = rolling(data=defense_coverage_scheme, roll_value=3, roll_type=\'mean\')\ndefense_coverage_summary_roll = rolling(data=defense_coverage_summary, roll_value=3, roll_type=\'mean\')\nslot_coverage_roll = rolling(data=slot_coverage, roll_value=3, roll_type=\'mean\')\n\nst_kickers_roll = rolling(data=st_kickers, roll_value=3, roll_type=\'mean\')\nst_punters_roll = rolling(data=st_punters, roll_value=3, roll_type=\'mean\')')


# ## TO DO: Create better imputation function before weighting team_position_group functions

# def filter_fillna(df=None, position=None, min_Var=None):
#     sub= df[df['position'].str.match(position)]
#     sub_limit = sub[(sub[min_Var] <=5) & (sub[min_Var] >=1)]
#     buckup_df = pd.DataFrame(sub_limit.median()).T
#     num_cols = sub.select_dtypes(include=[np.number]).columns.tolist()
#     msk = sub.isnull()
#     tmp = sub[num_cols].mask(msk, buckup_df[num_cols])
#     tmp = np.where(msk[num_cols], buckup_df[num_cols], tmp[num_cols])
#     tmp = pd.DataFrame(tmp, columns=buckup_df.columns)
#     ids = pd.DataFrame(sub.select_dtypes(exclude=[np.number])).reset_index(drop=True)
#     mrg = pd.concat([ids, tmp], axis=1)
#     return mrg
#     

# In[28]:

get_ipython().run_cell_magic('time', '', "\ndef impute(df):\n    df = df.apply(pd.to_numeric, errors='ignore')\n    df.reset_index(inplace=True, drop=True)\n    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()\n    df[num_cols]= df.groupby(df['team_id_impute'])[num_cols].fillna(df.median()).reset_index(level=0, drop=True)\n    return df\n\npassing_depth_roll = impute(passing_depth_roll)\npassing_allowed_pressure_roll = impute(passing_allowed_pressure_roll)\npassing_pressure_roll = impute(passing_pressure_roll)\npassing_concept_roll = impute(passing_concept_roll)\ntime_in_pocket_roll = impute(time_in_pocket_roll)\npassing_summ_conc_roll = impute(passing_summ_conc_roll)\n\nrec_summ_conc_roll = impute(rec_summ_conc_roll)\nreceiving_concept_roll = impute(receiving_concept_roll)\nreceiving_depth_roll = impute(receiving_depth_roll)\nreceiving_scheme_roll = impute(receiving_scheme_roll)\n\nrush_summ_conc_roll = impute(rush_summ_conc_roll)\n\nblock_summ_conc_roll = impute(block_summ_conc_roll)\noffense_pass_blocking_roll = impute(offense_pass_blocking_roll)\noffense_run_blocking_roll = impute(offense_run_blocking_roll)\n\ndef_summ_conc_roll = impute(def_summ_conc_roll)\npass_rush_summary_roll = impute(pass_rush_summary_roll)\nrun_defense_summary_roll = impute(run_defense_summary_roll)\ndefense_coverage_scheme_roll = impute(defense_coverage_scheme_roll)\ndefense_coverage_summary_roll = impute(defense_coverage_summary_roll)\nslot_coverage_roll = impute(slot_coverage_roll)\n\nst_kickers_roll = impute(st_kickers_roll)\nst_punters_roll = impute(st_punters_roll)")


# In[29]:

biometrics = pd.read_csv('./other_data/2022_imputed_combine.csv')
biometrics=biometrics[['player_team_id','position','height_clean','weight_clean', 'speed_clean',
'hand_size', 'arm_length', 'bench','vertical', 'broad_jump', 'shuttle', '3cone', 'explosive', 'size_speed','draft_yr', 'round', 'selection']]
biometrics['position']=biometrics['position'].apply(str)


qb_bio = biometrics[biometrics['position'].isin(['qb'])]
rb_bio = biometrics[biometrics['position'].isin(['hb','qb','fb','wr'])]
rec_bio = biometrics[biometrics['position'].isin(['wr','te','hb'])]
ol_bio = biometrics[biometrics['position'].isin(['ol','te'])]
def_bio_dl = biometrics[biometrics['position'].isin(['dl'])]
def_bio_db = biometrics[biometrics['position'].isin(['db'])]
def_bio_lb = biometrics[biometrics['position'].isin(['lb'])]
st_bio = biometrics[biometrics['position'].isin(['st'])]

qb_median = qb_bio.groupby(['position']).median().reset_index()
rb_median = rb_bio.groupby(['position']).median().reset_index()
rec_median = rec_bio.groupby(['position']).median().reset_index()
ol_median = ol_bio.groupby(['position']).median().reset_index()
dl_median = def_bio_dl.groupby(['position']).median().reset_index()
db_median = def_bio_db.groupby(['position']).median().reset_index()
lb_median = def_bio_lb.groupby(['position']).median().reset_index()
st_median = st_bio.groupby(['position']).median().reset_index()


qb_bio.drop(['position'], axis=1, inplace=True)
rb_bio.drop(['position'], axis=1, inplace=True)
rec_bio.drop(['position'], axis=1, inplace=True)
ol_bio.drop(['position'], axis=1, inplace=True)
def_bio_dl.drop(['position'], axis=1, inplace=True)
def_bio_db.drop(['position'], axis=1, inplace=True)
def_bio_lb.drop(['position'], axis=1, inplace=True)
st_bio.drop(['position'], axis=1, inplace=True)


## fill in missing bio data with median for position ##
rush_summ_conc_roll['position'] = rush_summ_conc_roll['position'].str.replace('fb','hb')
temp_fillna_df = pd.merge(rush_summ_conc_roll, rb_median, on='position', how='left')
rush_summ_conc_roll = pd.merge(rush_summ_conc_roll, rb_bio, on='player_team_id', how='left')
rush_summ_conc_roll = rush_summ_conc_roll.combine_first(temp_fillna_df)

## fill in missing bio data with median for position- qb ##
temp_fillna_df = pd.merge(passing_summ_conc_roll, qb_median, on='position', how='left')
passing_summ_conc_roll = pd.merge(passing_summ_conc_roll, qb_bio, on='player_team_id', how='left')
passing_summ_conc_roll = passing_summ_conc_roll.combine_first(temp_fillna_df)


## fill in missing bio data with median for position- rec ##
rec_summ_conc_roll['position'] = rec_summ_conc_roll['position'].str.replace('fb','hb')
temp_fillna_df = pd.merge(rec_summ_conc_roll, rec_median, on='position', how='left')
rec_summ_conc_roll = pd.merge(rec_summ_conc_roll, rec_bio, on='player_team_id', how='left')
rec_summ_conc_roll = rec_summ_conc_roll.combine_first(temp_fillna_df)

## fill in missing bio data with median for position- rec ##
block_summ_conc_roll=block_summ_conc_roll[block_summ_conc_roll['position'] != 'cb']
block_summ_conc_roll['position'] = block_summ_conc_roll['position'].str.replace('t','ol')
block_summ_conc_roll['position'] = block_summ_conc_roll['position'].str.replace('g','ol')
temp_fillna_df = pd.merge(block_summ_conc_roll , ol_median, on='position', how='left')
block_summ_conc_roll = pd.merge(block_summ_conc_roll, ol_bio, on='player_team_id', how='left')
block_summ_conc_roll = block_summ_conc_roll.combine_first(temp_fillna_df)


def_line = def_summ_conc_roll[def_summ_conc_roll['position'].isin(['ed','di'])]
def_line['position'] = def_line['position'].str.replace('ed','dl')
def_line['position'] = def_line['position'].str.replace('di','dl')
temp_fillna_df = pd.merge(def_line, dl_median , on='position', how='left')
def_line = pd.merge(def_line, def_bio_dl, on='player_team_id', how='left')
def_line = def_line.combine_first(temp_fillna_df)

def_line=def_line[['unique_team_id','position','height_clean','weight_clean', 'speed_clean',
'hand_size', 'arm_length', 'bench','vertical', 'broad_jump', 'shuttle', '3cone', 'explosive', 'size_speed','draft_yr', 'round', 'selection']]
def_line = def_line.rename(columns={c: c+'_dls_bio' for c in def_line.columns if c not in ['unique_team_id']})
def_line = def_line.groupby('unique_team_id').mean().reset_index(drop=False)

def_lbs = def_summ_conc_roll[def_summ_conc_roll['position'].isin(['lb'])]
temp_fillna_df = pd.merge(def_lbs, lb_median , on='position', how='left')
def_lbs = pd.merge(def_lbs, def_bio_lb, on='player_team_id', how='left')
def_lbs = def_lbs.combine_first(temp_fillna_df)
def_lbs=def_lbs[['unique_team_id','position','height_clean','weight_clean', 'speed_clean',
'hand_size', 'arm_length', 'bench','vertical', 'broad_jump', 'shuttle', '3cone', 'explosive', 'size_speed','draft_yr', 'round', 'selection']]
def_lbs = def_lbs.rename(columns={c: c+'_lbs_bio' for c in def_lbs.columns if c not in ['unique_team_id']})
def_lbs = def_lbs.groupby('unique_team_id').mean().reset_index(drop=False)

def_db = def_summ_conc_roll[def_summ_conc_roll['position'].isin(['cb','s'])]
def_db['position'] = def_db['position'].str.replace('cb','db')
def_db['position'] = def_db['position'].str.replace('s','db')
temp_fillna_df = pd.merge(def_db, lb_median , on='position', how='left')
def_db = pd.merge(def_db, def_bio_db, on='player_team_id', how='left')
def_db = def_db.combine_first(temp_fillna_df)
def_db=def_db[['unique_team_id','position','height_clean','weight_clean', 'speed_clean',
'hand_size', 'arm_length', 'bench','vertical', 'broad_jump', 'shuttle', '3cone', 'explosive', 'size_speed','draft_yr', 'round', 'selection']]
def_db = def_db.rename(columns={c: c+'_dbs_bio' for c in def_db.columns if c not in ['player_team_id','unique_team_id']})
def_db = def_db.groupby('unique_team_id').mean().reset_index(drop=False)


# # Combine players for each dataset into team_year_week groupings
# 
# ### These next few cells will compute weighted averages based on average snaps played.  The first function will default snaps to 1 if snap value is 0.  The rest of the functions are dataset specific and will compute the weighted averages based on rollup aaverages and snaps played.
# 
# #### For example: Washington had 5 rbs player in the last 3 games.  It doesn't make sense to weight all the players stats into a single average if 3 of those backs only averaged 2 snaps and rushed for 2 yards whereas B. Robinson averages 18 snaps and rushes for 65 yards and Gibson averages 10 snaps for 40 yards.  Therefore we weight each players rolling average based on their rolling snaps played. 

# ## Compute rushing weighted average dataset

# In[30]:

rush_summ_conc_roll.drop_duplicates(subset=['p_id'], keep='first', inplace=True)


## make sure we aren't weighting w/a 0 value (non-designed runs are cancelled ##
def rush_att(nData, var=None):
    if nData[var] == 0:
        return 1
    else:
        return nData[var]

rush_summ_conc_roll['rush_summary_attempts'] = rush_summ_conc_roll.apply(lambda df: rush_att(df, var='rush_summary_attempts'), axis=1)   


def weighted(nData, snap_Var='rush_summary_attempts'):
    data_cols = nData.select_dtypes(include=[np.number])
    num_cols = data_cols[data_cols.columns.drop(list(data_cols.filter(regex='player_game_count|player_id|plyr_number|week|year|team_id')))].columns.tolist()
    return pd.Series(np.average(nData[num_cols], weights=nData[snap_Var], axis=0), num_cols)


rb_stats = rush_summ_conc_roll.groupby('unique_team_id').apply(weighted).reset_index()
rb_stats.tail(n=10)
rb_stats = rb_stats.rename(columns={c: c+'_rush' for c in rb_stats.columns if c not in ['unique_team_id']})


# In[31]:

passing_concept_roll.columns[:50]


# ## Compute Passing weight average datasets

# In[32]:

passing_summ_conc_roll.drop_duplicates(subset=['p_id'], keep='first', inplace=True)

def pass_att(nData, var=None):
    if nData[var] == 0:
        return 1
    else:
        return nData[var]
passing_summ_conc_roll['pass_summary_dropbacks'] = passing_summ_conc_roll.apply(lambda df: pass_att(df, var='pass_summary_dropbacks'), axis=1)
passing_depth_roll['pass_depth_base_dropbacks'] = passing_depth_roll.apply(lambda df: pass_att(df, var='pass_depth_base_dropbacks'), axis=1)  
passing_pressure_roll['pass_under_pressure_base_dropbacks'] = passing_pressure_roll.apply(lambda df: pass_att(df, var='pass_under_pressure_base_dropbacks'), axis=1)  
passing_allowed_pressure_roll['pressure_source_allowed_pressure_dropbacks'] = passing_allowed_pressure_roll.apply(lambda df: pass_att(df, var='pressure_source_allowed_pressure_dropbacks'), axis=1)  
passing_concept_roll['pass_concept_dropbacks'] = passing_concept_roll.apply(lambda df: pass_att(df, var='pass_concept_dropbacks'), axis=1)  
time_in_pocket_roll['pass_time_dropbacks'] = time_in_pocket_roll.apply(lambda df: pass_att(df, var='pass_time_dropbacks'), axis=1)     


def weighted(nData, snap_Var='pass_summary_dropbacks'):
    data_cols = nData.select_dtypes(include=[np.number])
    num_cols = data_cols[data_cols.columns.drop(list(data_cols.filter(regex='player_game_count|player_id|plyr_number|week|year|team_id')))].columns.tolist()
    return pd.Series(np.average(nData[num_cols], weights=nData[snap_Var], axis=0), num_cols)
qb_stats = passing_summ_conc_roll.groupby('unique_team_id').apply(weighted).reset_index()

def weighted(nData, snap_Var='pass_depth_base_dropbacks'):
    data_cols = nData.select_dtypes(include=[np.number])
    num_cols = data_cols[data_cols.columns.drop(list(data_cols.filter(regex='player_game_count|player_id|plyr_number|week|year|team_id')))].columns.tolist()
    return pd.Series(np.average(nData[num_cols], weights=nData[snap_Var], axis=0), num_cols)
pass_depth_stats = passing_depth_roll.groupby('unique_team_id').apply(weighted).reset_index()
pass_depth_stats = pass_depth_stats.rename(columns={c: c+'_passdepth' for c in pass_depth_stats.columns if c not in ['unique_team_id']})


def weighted(nData, snap_Var='pressure_source_allowed_pressure_dropbacks'):
    data_cols = nData.select_dtypes(include=[np.number])
    num_cols = data_cols[data_cols.columns.drop(list(data_cols.filter(regex='player_game_count|player_id|plyr_number|week|year|team_id')))].columns.tolist()
    return pd.Series(np.average(nData[num_cols], weights=nData[snap_Var], axis=0), num_cols)
passing_allowed_pressure_stats = passing_allowed_pressure_roll.groupby('unique_team_id').apply(weighted).reset_index()
passing_allowed_pressure_stats = passing_allowed_pressure_stats.rename(columns={c: c+'_pass_allow_pressure' for c in passing_allowed_pressure_stats.columns if c not in ['unique_team_id']})

def weighted(nData, snap_Var='pass_under_pressure_base_dropbacks'):
    data_cols = nData.select_dtypes(include=[np.number])
    num_cols = data_cols[data_cols.columns.drop(list(data_cols.filter(regex='player_game_count|player_id|plyr_number|week|year|team_id')))].columns.tolist()
    return pd.Series(np.average(nData[num_cols], weights=nData[snap_Var], axis=0), num_cols)
passing_pressure_stats = passing_pressure_roll.groupby('unique_team_id').apply(weighted).reset_index()
passing_pressure_stats = passing_pressure_stats.rename(columns={c: c+'_pass_pressure' for c in passing_pressure_stats.columns if c not in ['unique_team_id']})


def weighted(nData, snap_Var='pass_concept_dropbacks'):
    data_cols = nData.select_dtypes(include=[np.number])
    num_cols = data_cols[data_cols.columns.drop(list(data_cols.filter(regex='player_game_count|player_id|plyr_number|week|year|team_id')))].columns.tolist()
    return pd.Series(np.average(nData[num_cols], weights=nData[snap_Var], axis=0), num_cols)
passing_concept_stats = passing_concept_roll.groupby('unique_team_id').apply(weighted).reset_index()
passing_concept_stats = passing_concept_stats.rename(columns={c: c+'_pass_conc' for c in passing_concept_stats.columns if c not in ['unique_team_id']})


def weighted(nData, snap_Var='pass_time_dropbacks'):
    data_cols = nData.select_dtypes(include=[np.number])
    num_cols = data_cols[data_cols.columns.drop(list(data_cols.filter(regex='player_game_count|player_id|plyr_number|week|year|team_id')))].columns.tolist()
    return pd.Series(np.average(nData[num_cols], weights=nData[snap_Var], axis=0), num_cols)
time_in_pocket_stats = time_in_pocket_roll.groupby('unique_team_id').apply(weighted).reset_index()
time_in_pocket_stats = time_in_pocket_stats.rename(columns={c: c+'_time_pocket' for c in time_in_pocket_stats.columns if c not in ['unique_team_id']})


qb_stats = qb_stats.rename(columns={c: c+'_passing' for c in qb_stats.columns if c not in ['unique_team_id']})


# In[33]:

qb_stats.head()


# ## Compute receiver weighted average datasets

# In[34]:

rec_summ_conc_roll.drop_duplicates(subset=['p_id'], keep='first', inplace=True)


## make sure we aren't weighting w/a 0 value (non-designed runs are cancelled ##
def rec_att(nData, var=None):
    if nData[var] == 0:
        return 1
    else:
        return nData[var]

rec_summ_conc_roll['rec_summary_targets'] = rec_summ_conc_roll.apply(lambda df: rec_att(df, var='rec_summary_targets'), axis=1)   
receiving_concept_roll['rec_concept_base_targets'] = receiving_concept_roll.apply(lambda df: rec_att(df, var='rec_concept_base_targets'), axis=1) 
receiving_depth_roll['rec_depth_base_targets'] = receiving_depth_roll.apply(lambda df: rec_att(df, var='rec_depth_base_targets'), axis=1) 
receiving_scheme_roll['rec_scheme_base_targets'] = receiving_scheme_roll.apply(lambda df: rec_att(df, var='rec_scheme_base_targets'), axis=1) 


def weighted(nData, snap_Var='rec_summary_targets'):
    data_cols = nData.select_dtypes(include=[np.number])
    num_cols = data_cols[data_cols.columns.drop(list(data_cols.filter(regex='player_game_count|player_id|plyr_number|week|year|team_id')))].columns.tolist()
    return pd.Series(np.average(nData[num_cols], weights=nData[snap_Var], axis=0), num_cols)

rec_stats = rec_summ_conc_roll.groupby('unique_team_id').apply(weighted).reset_index()

def weighted(nData, snap_Var='rec_concept_base_targets'):
    data_cols = nData.select_dtypes(include=[np.number])
    num_cols = data_cols[data_cols.columns.drop(list(data_cols.filter(regex='player_game_count|player_id|plyr_number|week|year|team_id')))].columns.tolist()
    return pd.Series(np.average(nData[num_cols], weights=nData[snap_Var], axis=0), num_cols)
	
receiving_concept = receiving_concept_roll.groupby('unique_team_id').apply(weighted).reset_index()
receiving_concept = receiving_concept.rename(columns={c: c+'_rec_concept' for c in receiving_concept.columns if c not in ['unique_team_id']})


def weighted(nData, snap_Var='rec_depth_base_targets'):
    data_cols = nData.select_dtypes(include=[np.number])
    num_cols = data_cols[data_cols.columns.drop(list(data_cols.filter(regex='player_game_count|player_id|plyr_number|week|year|team_id')))].columns.tolist()
    return pd.Series(np.average(nData[num_cols], weights=nData[snap_Var], axis=0), num_cols)
	
receiving_depth = receiving_depth_roll.groupby('unique_team_id').apply(weighted).reset_index()
receiving_depth = receiving_depth.rename(columns={c: c+'_rec_depth' for c in receiving_depth.columns if c not in ['unique_team_id']})


def weighted(nData, snap_Var='rec_scheme_base_targets'):
    data_cols = nData.select_dtypes(include=[np.number])
    num_cols = data_cols[data_cols.columns.drop(list(data_cols.filter(regex='player_game_count|player_id|plyr_number|week|year|team_id')))].columns.tolist()
    return pd.Series(np.average(nData[num_cols], weights=nData[snap_Var], axis=0), num_cols)
	
receiving_scheme = receiving_scheme_roll.groupby('unique_team_id').apply(weighted).reset_index()
receiving_scheme = receiving_scheme.rename(columns={c: c+'_rec_schem' for c in receiving_scheme.columns if c not in ['unique_team_id']})

rec_stats = rec_stats.rename(columns={c: c+'_rec' for c in rec_stats.columns if c not in ['unique_team_id']})


# ## Compute OL weighted average dataset

# In[35]:

block_summ_conc_roll.drop_duplicates(subset=['p_id'], keep='first', inplace=True)

def snap_fix(nData, var=None):
    if nData[var] == 0:
        return 1
    else:
        return nData[var]

block_summ_conc_roll['block_summary_snap_counts_offense'] = block_summ_conc_roll.apply(lambda df: snap_fix(df, var='block_summary_snap_counts_offense'), axis=1)
offense_pass_blocking_roll['pass_block_snap_counts_pass_block'] = offense_pass_blocking_roll.apply(lambda df: snap_fix(df, var='pass_block_snap_counts_pass_block'), axis=1) 
offense_run_blocking_roll['run_block_snap_counts_run_block'] = offense_run_blocking_roll.apply(lambda df: snap_fix(df, var='run_block_snap_counts_run_block'), axis=1) 


def weighted(nData, snap_Var='block_summary_snap_counts_offense'):
    data_cols = nData.select_dtypes(include=[np.number])
    num_cols = data_cols[data_cols.columns.drop(list(data_cols.filter(regex='player_game_count|player_id|plyr_number|week|year|team_id')))].columns.tolist()
    return pd.Series(np.average(nData[num_cols], weights=nData[snap_Var], axis=0), num_cols)
ol_stats = block_summ_conc_roll.groupby('unique_team_id').apply(weighted).reset_index()

def weighted(nData, snap_Var='pass_block_snap_counts_pass_block'):
    data_cols = nData.select_dtypes(include=[np.number])
    num_cols = data_cols[data_cols.columns.drop(list(data_cols.filter(regex='player_game_count|player_id|plyr_number|week|year|team_id')))].columns.tolist()
    return pd.Series(np.average(nData[num_cols], weights=nData[snap_Var], axis=0), num_cols)
offense_pass_blocking_stats = offense_pass_blocking_roll.groupby('unique_team_id').apply(weighted).reset_index()
offense_pass_blocking_stats = offense_pass_blocking_stats.rename(columns={c: c+'_pass_block' for c in offense_pass_blocking_stats.columns if c not in ['unique_team_id']})

def weighted(nData, snap_Var='run_block_snap_counts_run_block'):
    data_cols = nData.select_dtypes(include=[np.number])
    num_cols = data_cols[data_cols.columns.drop(list(data_cols.filter(regex='player_game_count|player_id|plyr_number|week|year|team_id')))].columns.tolist()
    return pd.Series(np.average(nData[num_cols], weights=nData[snap_Var], axis=0), num_cols)
offense_run_blocking_stats = offense_run_blocking_roll.groupby('unique_team_id').apply(weighted).reset_index()
offense_run_blocking_stats = offense_run_blocking_stats.rename(columns={c: c+'_run_block' for c in offense_run_blocking_stats.columns if c not in ['unique_team_id']})

ol_stats = ol_stats.rename(columns={c: c+'_block' for c in ol_stats.columns if c not in ['unique_team_id']})


# ## Compute defensive weighted averages datasets

# In[36]:

def_line.columns


# In[37]:

def_summ_conc_roll.drop_duplicates(subset=['p_id'], keep='first', inplace=True)



def snap_fixs(nData, var=None):
    if nData[var] == 0:
        return 1
    else:
        return nData[var]

def_summ_conc_roll['def_summary_snap_counts_defense'] = def_summ_conc_roll.apply(lambda df: snap_fixs(df, var='def_summary_snap_counts_defense'), axis=1) 
def_summ_conc_roll['def_summary_snap_counts_run_defense'] = def_summ_conc_roll.apply(lambda df: snap_fixs(df, var='def_summary_snap_counts_run_defense'), axis=1) 
def_summ_conc_roll['def_summary_snap_counts_pass_rush'] = def_summ_conc_roll.apply(lambda df: snap_fixs(df, var='def_summary_snap_counts_pass_rush'), axis=1) 
def_summ_conc_roll['def_summary_snap_counts_coverage'] = def_summ_conc_roll.apply(lambda df: snap_fixs(df, var='def_summary_snap_counts_coverage'), axis=1) 


pass_rush_summary_roll['pass_rush_snap_counts_pass_play'] = pass_rush_summary_roll.apply(lambda df: snap_fixs(df, var='pass_rush_snap_counts_pass_play'), axis=1)
run_defense_summary_roll['run_defense_snap_counts_run'] = run_defense_summary_roll.apply(lambda df: snap_fixs(df, var='run_defense_snap_counts_run'), axis=1)
defense_coverage_scheme_roll['def_coverage_scheme_base_snap_counts_coverage'] = defense_coverage_scheme_roll.apply(lambda df: snap_fixs(df, var='def_coverage_scheme_base_snap_counts_coverage'), axis=1)
defense_coverage_summary_roll['def_coverage_summary_coverage_snaps_per_target'] = defense_coverage_summary_roll.apply(lambda df: snap_fixs(df, var='def_coverage_summary_coverage_snaps_per_target'), axis=1)
slot_coverage_roll['def_slot_coverage_coverage_snaps'] = slot_coverage_roll.apply(lambda df: snap_fixs(df, var='def_slot_coverage_coverage_snaps'), axis=1)




## Subset into defense positional groups ##
def_rundef = def_summ_conc_roll[def_summ_conc_roll['position'].isin(['ed','di','lb'])]
def_passrush = def_summ_conc_roll[def_summ_conc_roll['position'].isin(['lb','ed','di'])]
def_cov = def_summ_conc_roll[def_summ_conc_roll['position'].isin(['lb','cb','s'])]

# def_rundef['position'] = def_rundef['position'].str.replace('di','dl')
# def_rundef['position'] = def_rundef['position'].str.replace('ed','dl')
# temp_fillna_df = pd.merge(def_rundef , dl_median, on='position', how='left')
# def_rundef = pd.merge(def_rundef, def_bio_dl, on='player_team_id', how='left')
# def_rundef = def_rundef.combine_first(temp_fillna_df); def_rundef.head()

# def_passrush['position'] = def_passrush['position'].str.replace('di','dl')
# def_passrush['position'] = def_passrush['position'].str.replace('ed','dl')
# temp_fillna_df = pd.merge(def_passrush , dl_median, on='position', how='left')
# def_passrush = pd.merge(def_passrush, def_bio_dl, on='player_team_id', how='left')
# def_passrush = def_passrush.combine_first(temp_fillna_df); def_passrush.head()


def weighted(nData, snap_Var='def_summary_snap_counts_defense'):
    data_cols = nData.select_dtypes(include=[np.number])
    num_cols = data_cols[data_cols.columns.drop(list(data_cols.filter(regex='player_game_count|player_id|plyr_number|week|year|team_id')))].columns.tolist()
    return pd.Series(np.average(nData[num_cols], weights=nData[snap_Var], axis=0), num_cols)
def_stats = def_summ_conc_roll.groupby('unique_team_id').apply(weighted).reset_index()
def_stats = def_stats.rename(columns={c: c+'_def_stats' for c in def_stats.columns if c not in ['unique_team_id']})

def weighted(nData, snap_Var='def_summary_snap_counts_run_defense'):
    data_cols = nData.select_dtypes(include=[np.number])
    num_cols = data_cols[data_cols.columns.drop(list(data_cols.filter(regex='player_game_count|player_id|plyr_number|week|year|team_id')))].columns.tolist()
    return pd.Series(np.average(nData[num_cols], weights=nData[snap_Var], axis=0), num_cols)
def_rundef = def_rundef.groupby('unique_team_id').apply(weighted).reset_index()
def_rundef = def_rundef.rename(columns={c: c+'_run_def' for c in def_rundef.columns if c not in ['unique_team_id']})

def weighted(nData, snap_Var='def_summary_snap_counts_pass_rush'):
    data_cols = nData.select_dtypes(include=[np.number])
    num_cols = data_cols[data_cols.columns.drop(list(data_cols.filter(regex='player_game_count|player_id|plyr_number|week|year|team_id')))].columns.tolist()
    return pd.Series(np.average(nData[num_cols], weights=nData[snap_Var], axis=0), num_cols)
def_passrush = def_passrush.groupby('unique_team_id').apply(weighted).reset_index()
def_passrush = def_passrush.rename(columns={c: c+'_passrush' for c in def_passrush.columns if c not in ['unique_team_id']})


def weighted(nData, snap_Var='def_summary_snap_counts_coverage'):
    data_cols = nData.select_dtypes(include=[np.number])
    num_cols = data_cols[data_cols.columns.drop(list(data_cols.filter(regex='player_game_count|player_id|plyr_number|week|year|team_id')))].columns.tolist()
    return pd.Series(np.average(nData[num_cols], weights=nData[snap_Var], axis=0), num_cols)
def_cov = def_cov.groupby('unique_team_id').apply(weighted).reset_index()
def_cov = def_cov.rename(columns={c: c+'_def_cov' for c in def_cov.columns if c not in ['unique_team_id']})





def weighted(nData, snap_Var='pass_rush_snap_counts_pass_play'):
    data_cols = nData.select_dtypes(include=[np.number])
    num_cols = data_cols[data_cols.columns.drop(list(data_cols.filter(regex='player_game_count|player_id|plyr_number|week|year|team_id')))].columns.tolist()
    return pd.Series(np.average(nData[num_cols], weights=nData[snap_Var], axis=0), num_cols)
pass_rush_stats = pass_rush_summary_roll.groupby('unique_team_id').apply(weighted).reset_index()
pass_rush_stats = pass_rush_stats.rename(columns={c: c+'_pass_rush_summ' for c in pass_rush_stats.columns if c not in ['unique_team_id']})


def weighted(nData, snap_Var='run_defense_snap_counts_run'):
    data_cols = nData.select_dtypes(include=[np.number])
    num_cols = data_cols[data_cols.columns.drop(list(data_cols.filter(regex='player_game_count|player_id|plyr_number|week|year|team_id')))].columns.tolist()
    return pd.Series(np.average(nData[num_cols], weights=nData[snap_Var], axis=0), num_cols)
run_defense_stats = run_defense_summary_roll.groupby('unique_team_id').apply(weighted).reset_index()
run_defense_stats = run_defense_stats.rename(columns={c: c+'_run_def_summ' for c in run_defense_stats.columns if c not in ['unique_team_id']})


def weighted(nData, snap_Var='def_coverage_summary_coverage_snaps_per_target'):
    data_cols = nData.select_dtypes(include=[np.number])
    num_cols = data_cols[data_cols.columns.drop(list(data_cols.filter(regex='player_game_count|player_id|plyr_number|week|year|team_id')))].columns.tolist()
    return pd.Series(np.average(nData[num_cols], weights=nData[snap_Var], axis=0), num_cols)
defense_coverage_summary_stats = defense_coverage_summary_roll.groupby('unique_team_id').apply(weighted).reset_index()
defense_coverage_summary_stats = defense_coverage_summary_stats.rename(columns={c: c+'_def_cov_summ' for c in defense_coverage_summary_stats.columns if c not in ['unique_team_id']})

def weighted(nData, snap_Var='def_coverage_scheme_base_snap_counts_coverage'):
    data_cols = nData.select_dtypes(include=[np.number])
    num_cols = data_cols[data_cols.columns.drop(list(data_cols.filter(regex='player_game_count|player_id|plyr_number|week|year|team_id')))].columns.tolist()
    return pd.Series(np.average(nData[num_cols], weights=nData[snap_Var], axis=0), num_cols)
defense_coverage_scheme_stats = defense_coverage_scheme_roll.groupby('unique_team_id').apply(weighted).reset_index()
defense_coverage_scheme_stats = defense_coverage_scheme_stats.rename(columns={c: c+'_def_cov_schem' for c in defense_coverage_scheme_stats.columns if c not in ['unique_team_id']})


def weighted(nData, snap_Var='def_slot_coverage_coverage_snaps'):
    data_cols = nData.select_dtypes(include=[np.number])
    num_cols = data_cols[data_cols.columns.drop(list(data_cols.filter(regex='player_game_count|player_id|plyr_number|week|year|team_id')))].columns.tolist()
    return pd.Series(np.average(nData[num_cols], weights=nData[snap_Var], axis=0), num_cols)
slot_coverage_stats = slot_coverage_roll.groupby('unique_team_id').apply(weighted).reset_index()
slot_coverage_stats = slot_coverage_stats.rename(columns={c: c+'_slot_cov' for c in slot_coverage_stats.columns if c not in ['unique_team_id']})

#def_stats = pd.merge(def_stats, def_rundef, on='unique_team_id', how='inner').merge(def_passrush, on='unique_team_id', how='inner').merge(def_cov, on='unique_team_id', how='inner')
# def_rundef = def_rundef.rename(columns={c: c+'_rundef' for c in def_rundef.columns if c not in ['unique_team_id']})

def_stats = pd.merge(def_stats, def_line, on='unique_team_id', how='left').merge(def_lbs, on='unique_team_id', how='left').merge(def_db, on='unique_team_id', how='left')
def_stats.head()


# ## Compute special teams weighted averages

# In[38]:

# st_bio.columns = [str(col) + '_st' for col in st_bio.columns]
# st_kickers_roll = pd.merge(st_kickers_roll, st_bio, left_on='player_team_id', right_on='unique_id_st', how='left')
# st_kickers_roll.drop_duplicates(subset=['p_id'], keep='first', inplace=True)

def kicks_fix(nData, var=None):
    if nData[var] == 0:
        return 1
    else:
        return nData[var]
		
st_kickers_roll['kicks'] = st_kickers_roll['kicking_pat_attempts']+st_kickers_roll['kicking_total_attempts']
st_kickers_roll ['kicks'] = st_kickers_roll .apply(lambda df: snap_fixs(df, var='kicks'), axis=1)

def weighted(nData, snap_Var='kicks'):
    data_cols = nData.select_dtypes(include=[np.number])
    num_cols = data_cols[data_cols.columns.drop(list(data_cols.filter(regex='player_game_count|player_id|plyr_number|week|year|team_id')))].columns.tolist()
    return pd.Series(np.average(nData[num_cols], weights=nData[snap_Var], axis=0), num_cols)
	
st_kickers = st_kickers_roll.groupby('unique_team_id').apply(weighted).reset_index()


# In[39]:

# st_punters_roll = pd.merge(st_punters_roll, st_bio, left_on='player_team_id', right_on='unique_id_st', how='left')
# st_punters_roll.drop_duplicates(subset=['p_id'], keep='first', inplace=True)

def punts_fix(nData, var=None):
    if nData[var] == 0:
        return 1
    else:
        return nData[var]
		
st_punters_roll['punting_attempts'] = st_punters_roll.apply(lambda df: snap_fixs(df, var='punting_attempts'), axis=1)

def weighted(nData, snap_Var='punting_attempts'):
    data_cols = nData.select_dtypes(include=[np.number])
    num_cols = data_cols[data_cols.columns.drop(list(data_cols.filter(regex='player_game_count|player_id|plyr_number|week|year|team_id')))].columns.tolist()
    return pd.Series(np.average(nData[num_cols], weights=nData[snap_Var], axis=0), num_cols)
	
st_punters = st_punters_roll.groupby('unique_team_id').apply(weighted).reset_index()


# ## Create Modeling File and write out to modeling_data directory

# In[40]:

spread_vars = spread_comb[spread_comb['schedule_week'] != '1']



from functools import reduce


spread_ids = spread_vars[['team_id','home_matchup_id','score_home','score_away']]
spread_ids.columns = ['unique_team_id','home_matchup_id','score_home','score_away']

spread_targs = spread_vars[['team_id',
'schedule_week',
'schedule_season',
'team_favorite_id',
'score_home',
'score_away',
'spread_favorite',
'starting_spread',
'Total Score Open',
'over_under_line',
'fav_cover',
'over_under_result',
'fav_homeoraway',
'remain_fav',
'spread_movement',
"ou_movement",
"strong_movement",
"fav_team_stronger",
"temperature",
"wind_mph",
"dome",
"precip"]]


dfs_list = [spread_ids,
            tgs_roll,
            fo_roll,
            qb_stats,
            rb_stats,
            rec_stats,
            ol_stats,
           def_stats,
           def_rundef,
           def_cov,
           def_passrush,
           st_punters,
           st_kickers]

dfs_team = reduce(lambda  left,right: pd.merge(left,right,on=['unique_team_id'],
                                            how='left'), dfs_list)

def fav_ids(nData):
    if str(nData['team_favorite_id']) in str(nData['team_id']):
        return nData['team_id']
    else:
        pass
spread_targs['fav_team_id'] = spread_targs.apply(lambda nData: fav_ids(nData), axis=1)


favs = spread_targs[~spread_targs['fav_team_id'].isnull()]
not_fav = spread_targs[spread_targs['fav_team_id'].isnull()]

not_fav_df = dfs_team[dfs_team.unique_team_id.isin(not_fav.team_id)]

dfs_team = dfs_team.rename(columns={c: c+'_fav' for c in dfs_team.columns if c not in ['unique_team_id','team_id','schedule_week','schedule_season','home_matchup_id','home_score','away_score','spread_favorite','over_under_line','fav_cover','over_under_result','wl','pf','pa']})
not_fav_df = not_fav_df.rename(columns={c: c+'_dog' for c in not_fav_df.columns if c not in ['unique_team_id','team_id','schedule_week','schedule_season','home_matchup_id','spread_favorite','over_under_line','fav_cover','over_under_result','wl','pf','pa']})

not_fav_df.drop(['unique_team_id','wl','score_away_dog','score_home_dog'], axis=1, inplace=True)


favs = favs[['team_id','schedule_week','schedule_season','spread_favorite','over_under_line','fav_cover','over_under_result']]
#not_fav = not_fav[['team_id','schedule_week','schedule_season','spread_favorite','over_under_line','fav_cover','over_under_result']]


# In[41]:

dfs_team.columns


# ## Merge files and write to modeling_data directory

# In[42]:

fin_df = pd.merge(favs, dfs_team, left_on='team_id', right_on='unique_team_id', how='left').merge(not_fav_df, on='home_matchup_id', how='left')
fin_df=fin_df.round(2)
fin_df.drop_duplicates(subset='home_matchup_id',inplace=True)


# In[43]:

fin_df["overall_performance_tgs_fav_vs_overall_performance_tgs_dog"] = fin_df["overall_performance_tgs_fav"]/fin_df["overall_performance_tgs_dog"]
fin_df["offense_tgs_fav_vs_offense_tgs_dog"] = fin_df["offense_tgs_fav"]/fin_df["offense_tgs_dog"]
fin_df["pass_tgs_fav_vs_pass_tgs_dog"] = fin_df["pass_tgs_fav"]/fin_df["pass_tgs_dog"]
fin_df["pass_blocking_tgs_fav_vs_pass_blocking_tgs_dog"] = fin_df["pass_blocking_tgs_fav"]/fin_df["pass_blocking_tgs_dog"]
fin_df["receiving_tgs_fav_vs_receiving_tgs_dog"] = fin_df["receiving_tgs_fav"]/fin_df["receiving_tgs_dog"]
fin_df["rushing_tgs_fav_vs_rushing_tgs_dog"] = fin_df["rushing_tgs_fav"]/fin_df["rushing_tgs_dog"]
fin_df["run_blocking_tgs_fav_vs_run_blocking_tgs_dog"] = fin_df["run_blocking_tgs_fav"]/fin_df["run_blocking_tgs_dog"]
fin_df["defense_tgs_fav_vs_defense_tgs_dog"] = fin_df["defense_tgs_fav"]/fin_df["defense_tgs_dog"]
fin_df["rush_defense_tgs_fav_vs_rush_defense_tgs_dog"] = fin_df["rush_defense_tgs_fav"]/fin_df["rush_defense_tgs_dog"]
fin_df["tackling_tgs_fav_vs_tackling_tgs_dog"] = fin_df["tackling_tgs_fav"]/fin_df["tackling_tgs_dog"]
fin_df["pass_rush_tgs_fav_vs_pass_rush_tgs_dog"] = fin_df["pass_rush_tgs_fav"]/fin_df["pass_rush_tgs_dog"]
fin_df["coverage_tgs_fav_vs_coverage_tgs_dog"] = fin_df["coverage_tgs_fav"]/fin_df["coverage_tgs_dog"]
fin_df["special_teams_tgs_fav_vs_special_teams_tgs_dog"] = fin_df["special_teams_tgs_fav"]/fin_df["special_teams_tgs_dog"]
fin_df["total_dvoa_fav_vs_total_dvoa_dog"] = fin_df["total_dvoa_fav"]/fin_df["total_dvoa_dog"]
fin_df["off_dvoa_fav_vs_off_dvoa_dog"] = fin_df["off_dvoa_fav"]/fin_df["off_dvoa_dog"]
fin_df["off_pass_dvoa_fav_vs_off_pass_dvoa_dog"] = fin_df["off_pass_dvoa_fav"]/fin_df["off_pass_dvoa_dog"]
fin_df["off_rush_dvoa_fav_vs_off_rush_dvoa_dog"] = fin_df["off_rush_dvoa_fav"]/fin_df["off_rush_dvoa_dog"]
fin_df["def_dvoa_fav_vs_def_dvoa_dog"] = fin_df["def_dvoa_fav"]/fin_df["def_dvoa_dog"]
fin_df["def_pass_dvoa_fav_vs_def_pass_dvoa_dog"] = fin_df["def_pass_dvoa_fav"]/fin_df["def_pass_dvoa_dog"]
fin_df["def_rush_dvoa_fav_vs_def_rush_dvoa_dog"] = fin_df["def_rush_dvoa_fav"]/fin_df["def_rush_dvoa_dog"]
fin_df["special_teams_dvoa_fav_vs_special_teams_dvoa_dog"] = fin_df["special_teams_dvoa_fav"]/fin_df["special_teams_dvoa_dog"]
		
fin_df["offense_tgs_fav_vs_defense_tgs_dog"] = fin_df["offense_tgs_fav"]/fin_df["defense_tgs_dog"]
fin_df["pass_tgs_fav_vs_coverage_tgs_dog"] = fin_df["pass_tgs_fav"]/fin_df["coverage_tgs_dog"]
fin_df["pass_tgs_fav_vs_pass_rush_tgs_dog"] = fin_df["pass_tgs_fav"]/fin_df["pass_rush_tgs_dog"]
fin_df["pass_blocking_tgs_fav_vs_pass_rush_tgs_dog"] = fin_df["pass_blocking_tgs_fav"]/fin_df["pass_rush_tgs_dog"]
fin_df["receiving_tgs_fav_vs_coverage_tgs_dog"] = fin_df["receiving_tgs_fav"]/fin_df["coverage_tgs_dog"]
fin_df["rushing_tgs_fav_vs_rush_defense_tgs_dog"] = fin_df["rushing_tgs_fav"]/fin_df["rush_defense_tgs_dog"]
fin_df["rushing_tgs_fav_vs_tackling_tgs_dog"] = fin_df["rushing_tgs_fav"]/fin_df["tackling_tgs_dog"]
fin_df["run_blocking_tgs_fav_vs_rush_defense_tgs_dog"] = fin_df["run_blocking_tgs_fav"]/fin_df["rush_defense_tgs_dog"]
		
fin_df["defense_tgs_fav_vs_offense_tgs_dog"] = fin_df["defense_tgs_fav"]/fin_df["offense_tgs_dog"]
fin_df["rush_defense_tgs_fav_vs_rushing_tgs_dog"] = fin_df["rush_defense_tgs_fav"]/fin_df["rushing_tgs_dog"]
fin_df["tackling_tgs_fav_vs_offense_tgs_dog"] = fin_df["tackling_tgs_fav"]/fin_df["offense_tgs_dog"]
fin_df["pass_rush_tgs_fav_vs_pass_blocking_tgs_dog"] = fin_df["pass_rush_tgs_fav"]/fin_df["pass_blocking_tgs_dog"]
fin_df["coverage_tgs_fav_vs_receiving_tgs_dog"] = fin_df["coverage_tgs_fav"]/fin_df["receiving_tgs_dog"]
		
fin_df["off_dvoa_fav_vs_def_dvoa_dog"] = fin_df["off_dvoa_fav"]/fin_df["def_dvoa_dog"]
fin_df["off_pass_dvoa_fav_vs_def_pass_dvoa_dog"] = fin_df["off_pass_dvoa_fav"]/fin_df["def_pass_dvoa_dog"]
fin_df["off_rush_dvoa_fav_vs_def_rush_dvoa_dog"] = fin_df["off_rush_dvoa_fav"]/fin_df["def_rush_dvoa_dog"]
		
fin_df["def_dvoa_fav_vs_off_dvoa_dog"] = fin_df["def_dvoa_fav"]/fin_df["off_dvoa_dog"]
fin_df["def_pass_dvoa_fav_vs_off_pass_dvoa_dog"] = fin_df["def_pass_dvoa_fav"]/fin_df["off_pass_dvoa_dog"]
fin_df["def_rush_dvoa_fav_vs_off_rush_dvoa_dog"] = fin_df["def_rush_dvoa_fav"]/fin_df["off_rush_dvoa_dog"]
		
		
fin_df["offense_tgs_dog_vs_defense_tgs_fav"] = fin_df["offense_tgs_dog"]/fin_df["defense_tgs_fav"]
fin_df["pass_tgs_dog_vs_coverage_tgs_fav"] = fin_df["pass_tgs_dog"]/fin_df["coverage_tgs_fav"]
fin_df["pass_tgs_dog_vs_pass_rush_tgs_fav"] = fin_df["pass_tgs_dog"]/fin_df["pass_rush_tgs_fav"]
fin_df["pass_blocking_tgs_dog_vs_pass_rush_tgs_fav"] = fin_df["pass_blocking_tgs_dog"]/fin_df["pass_rush_tgs_fav"]
fin_df["receiving_tgs_dog_vs_coverage_tgs_fav"] = fin_df["receiving_tgs_dog"]/fin_df["coverage_tgs_fav"]
fin_df["rushing_tgs_dog_vs_rush_defense_tgs_fav"] = fin_df["rushing_tgs_dog"]/fin_df["rush_defense_tgs_fav"]
fin_df["rushing_tgs_dog_vs_tackling_tgs_fav"] = fin_df["rushing_tgs_dog"]/fin_df["tackling_tgs_fav"]
fin_df["run_blocking_tgs_dog_vs_rush_defense_tgs_fav"] = fin_df["run_blocking_tgs_dog"]/fin_df["rush_defense_tgs_fav"]
		
fin_df["defense_tgs_dog_vs_offense_tgs_fav"] = fin_df["defense_tgs_dog"]/fin_df["offense_tgs_fav"]
fin_df["rush_defense_tgs_dog_vs_rushing_tgs_fav"] = fin_df["rush_defense_tgs_dog"]/fin_df["rushing_tgs_fav"]
fin_df["tackling_tgs_dog_vs_offense_tgs_fav"] = fin_df["tackling_tgs_dog"]/fin_df["offense_tgs_fav"]
fin_df["pass_rush_tgs_dog_vs_pass_blocking_tgs_fav"] = fin_df["pass_rush_tgs_dog"]/fin_df["pass_blocking_tgs_fav"]
fin_df["coverage_tgs_dog_vs_receiving_tgs_fav"] = fin_df["coverage_tgs_dog"]/fin_df["receiving_tgs_fav"]
		
fin_df["off_dvoa_dog_vs_def_dvoa_fav"] = fin_df["off_dvoa_dog"]/fin_df["def_dvoa_fav"]
fin_df["off_pass_dvoa_dog_vs_def_pass_dvoa_fav"] = fin_df["off_pass_dvoa_dog"]/fin_df["def_pass_dvoa_fav"]
fin_df["off_rush_dvoa_dog_vs_def_rush_dvoa_fav"] = fin_df["off_rush_dvoa_dog"]/fin_df["def_rush_dvoa_fav"]
		
fin_df["def_dvoa_dog_vs_off_dvoa_fav"] = fin_df["def_dvoa_dog"]/fin_df["off_dvoa_fav"]
fin_df["def_pass_dvoa_dog_vs_off_pass_dvoa_fav"] = fin_df["def_pass_dvoa_dog"]/fin_df["off_pass_dvoa_fav"]
fin_df["def_rush_dvoa_dog_vs_off_rush_dvoa_fav"] = fin_df["def_rush_dvoa_dog"]/fin_df["off_rush_dvoa_fav"]
		
		
fin_df["off_tgs_vs_def_tgs_matchup"] = fin_df["offense_tgs_fav_vs_defense_tgs_dog"]/fin_df["offense_tgs_dog_vs_defense_tgs_fav"]
fin_df["pass_tgs_vs_def_cov_tgs_matchup"] = fin_df["pass_tgs_fav_vs_coverage_tgs_dog"]/fin_df["pass_tgs_dog_vs_coverage_tgs_fav"]
fin_df["pass_tgs_vs_def_passrush_tgs_matchup"] = fin_df["pass_tgs_fav_vs_pass_rush_tgs_dog"]/fin_df["pass_tgs_dog_vs_pass_rush_tgs_fav"]
fin_df["passblock_tgs_vs_def_passrush_tgs_matchup"] = fin_df["pass_blocking_tgs_fav_vs_pass_rush_tgs_dog"]/fin_df["pass_blocking_tgs_dog_vs_pass_rush_tgs_fav"]
fin_df["rec_tgs_vs_def_cov_tgs_matchup"] = fin_df["receiving_tgs_fav_vs_coverage_tgs_dog"]/fin_df["receiving_tgs_dog_vs_coverage_tgs_fav"]
fin_df["rush_tgs_vs_def_rundef_tgs_matchup"] = fin_df["rushing_tgs_fav_vs_rush_defense_tgs_dog"]/fin_df["rushing_tgs_dog_vs_rush_defense_tgs_fav"]
fin_df["rush_tgs_vs_def_tackle_tgs_matchup"] = fin_df["rushing_tgs_fav_vs_tackling_tgs_dog"]/fin_df["rushing_tgs_dog_vs_tackling_tgs_fav"]
fin_df["runblock_tgs_vs_def_rundef_tgs_matchup"] = fin_df["run_blocking_tgs_fav_vs_rush_defense_tgs_dog"]/fin_df["run_blocking_tgs_dog_vs_rush_defense_tgs_fav"]
fin_df["defense_tgs_vs_offense_tgs_matchup"] = fin_df["defense_tgs_fav_vs_offense_tgs_dog"]/fin_df["defense_tgs_dog_vs_offense_tgs_fav"]
fin_df["rush_defense_tgs_vs_rushing_tgs_matchup"] = fin_df["rush_defense_tgs_fav_vs_rushing_tgs_dog"]/fin_df["rush_defense_tgs_dog_vs_rushing_tgs_fav"]
fin_df["tackling_tgs_vs_offense_tgs_matchup"] = fin_df["tackling_tgs_fav_vs_offense_tgs_dog"]/fin_df["tackling_tgs_dog_vs_offense_tgs_fav"]
fin_df["pass_rush_tgs_vs_pass_blocking_tgs_matchup"] = fin_df["pass_rush_tgs_fav_vs_pass_blocking_tgs_dog"]/fin_df["pass_rush_tgs_dog_vs_pass_blocking_tgs_fav"]
fin_df["coverage_tgs_vs_receiving_tgs_matchup"] = fin_df["coverage_tgs_fav_vs_receiving_tgs_dog"]/fin_df["coverage_tgs_dog_vs_receiving_tgs_fav"]

fin_df["off_dvoa_vs_def_dvoa_matchup"] = fin_df["off_dvoa_fav_vs_def_dvoa_dog"]/fin_df["off_dvoa_dog_vs_def_dvoa_fav"]
fin_df["off_pass_dvoa_vs_def_pass_dvoa_matchup"] = fin_df["off_pass_dvoa_fav_vs_def_pass_dvoa_dog"]/fin_df["off_pass_dvoa_dog_vs_def_pass_dvoa_fav"]
fin_df["off_rush_dvoa_vs_def_rush_dvoa_matchup"] = fin_df["off_rush_dvoa_fav_vs_def_rush_dvoa_dog"]/fin_df["off_rush_dvoa_dog_vs_def_rush_dvoa_fav"]

fin_df["def_dvoa_vs_off_dvoa_matchup"] = fin_df["def_dvoa_fav_vs_off_dvoa_dog"]/fin_df["def_dvoa_dog_vs_off_dvoa_fav"]
fin_df["def_pass_dvoa_vs_off_pass_dvoa_matchup"] = fin_df["def_pass_dvoa_fav_vs_off_pass_dvoa_dog"]/fin_df["def_pass_dvoa_dog_vs_off_pass_dvoa_fav"]
fin_df["def_rush_dvoa_vs_off_rush_dvoa_matchup"] = fin_df["def_rush_dvoa_fav_vs_off_rush_dvoa_dog"]/fin_df["def_rush_dvoa_dog_vs_off_rush_dvoa_fav"]



fin_df.to_csv('./modeling_data/nfl_spreads_w'+cur_week_str+'.csv', index=False)


# ### UNDER CONSTRUCTION: Creating function to create modeling file by user selected datasets

# In[44]:

from functools import reduce

spread_ids = spread_vars[['team_id','home_matchup_id','score_home','score_away']]
spread_ids.columns = ['unique_team_id','home_matchup_id','score_home','score_away']

spread_targs = spread_vars[['team_id',
'schedule_week',
'schedule_season',
'team_favorite_id',
'score_home',
'score_away',
'spread_favorite',
'starting_spread',
'Total Score Open',
'over_under_line',
'fav_cover',
'over_under_result',
'fav_homeoraway',
'remain_fav',
'spread_movement',
"ou_movement",
"strong_movement",
"fav_team_stronger",
"temperature",
"wind_mph",
"dome",
"precip"]]



sample=[spread_ids,
            tgs_roll,
            fo_roll,
            qb_stats,
            rb_stats,
            rec_stats,
            ol_stats,
           def_stats,
           def_rundef,
           def_cov,
           def_passrush,
           st_punters,
           st_kickers]
           
def fav_ids(nData):
    if str(nData['team_favorite_id']) in str(nData['team_id']):
        return nData['team_id']
    else:
        pass
        
def build_model_dataset(data_list=None):
    """
        Args:
        data_list: User provides a list of dataframes in format - [df1, df2, df3...] to be used to create modeling dataset.
        
        Options: 
        Football Outsiders
        fo_roll - 
        
        PFF
        -Team Game Summaries -
        tgs_roll -

        -Passing:
        qb_stats -
        passing_depth_stats -
        passing_pressure_stats -
        passing_allowed_pressure_stats -
        passing_concept_stats -
        time_in_pocket_stats -
        
        -Receiving:
        rec_stats -
        receiving_concept -
        receiving_depth -
        receiving_scheme -
        
        -Blocking:
        ol_stats -
        offense_pass_blocking_roll -
        offense_run_blocking_roll -
        
        -Defense:
        def_stats -
        def_rundef -
        def_passrush -
        def_cov -
        pass_rush_stats -
        defense_coverage_summary_stats -
        run_defense_stats -
        defense_coverage_scheme_stats -
        slot_coverage_stats -
        
        -Special Teams:
        st_kickers -
        st_punters - 
    """
    
    dataset_list = data_list
    dfs_team = reduce(lambda  left,right: pd.merge(left,right,on=['unique_team_id'], how='left'), dataset_list)
    spread_targs['fav_team_id'] = spread_targs.apply(lambda nData: fav_ids(nData), axis=1)
    favs = spread_targs[~spread_targs['fav_team_id'].isnull()]
    not_fav = spread_targs[spread_targs['fav_team_id'].isnull()]

    not_fav_df = dfs_team[dfs_team.unique_team_id.isin(not_fav.team_id)]

    dfs_team = dfs_team.rename(columns={c: c+'_fav' for c in dfs_team.columns if c not in ['unique_team_id','team_id','schedule_week','schedule_season','home_matchup_id','home_score','away_score','spread_favorite','over_under_line','fav_cover','over_under_result','wl','pf','pa']})
    not_fav_df = not_fav_df.rename(columns={c: c+'_dog' for c in not_fav_df.columns if c not in ['unique_team_id','team_id','schedule_week','schedule_season','home_matchup_id','spread_favorite','over_under_line','fav_cover','over_under_result','wl','pf','pa']})

    not_fav_df.drop(['unique_team_id','wl','score_away_dog','score_home_dog'], axis=1, inplace=True)


    favs = favs[['team_id','schedule_week','schedule_season','spread_favorite','over_under_line','fav_cover','over_under_result',\
                 'starting_spread','Total Score Open','fav_homeoraway','remain_fav','spread_movement',"ou_movement","strong_movement","fav_team_stronger",]]
    return pd.merge(favs, dfs_team, left_on='team_id', right_on='unique_team_id', how='left').merge(not_fav_df, on='home_matchup_id', how='left')




# In[45]:

sample=[spread_ids,
        inj_final,
            tgs_roll,
            fo_roll,
            qb_stats,
            passing_concept_stats,
            passing_pressure_stats,
            time_in_pocket_stats,
            passing_allowed_pressure_stats,
            pass_depth_stats,
            rb_stats,
            rec_stats,
            receiving_scheme,
            receiving_depth,
            receiving_concept,
            ol_stats,
            offense_run_blocking_stats,
            offense_pass_blocking_stats,
           def_stats,
           def_rundef,
           def_cov,
           def_passrush,
            pass_rush_stats,
            run_defense_stats,
        defense_coverage_summary_stats,
            defense_coverage_scheme_stats,
            slot_coverage_stats,
           st_punters,
           st_kickers]


# In[46]:

fin_df=build_model_dataset(data_list=sample)
fin_df=fin_df.round(2)
fin_df.drop_duplicates(subset='home_matchup_id',inplace=True)


# In[47]:

fin_df["overall_performance_tgs_fav_vs_overall_performance_tgs_dog"] = fin_df["overall_performance_tgs_fav"]/fin_df["overall_performance_tgs_dog"]
fin_df["offense_tgs_fav_vs_offense_tgs_dog"] = fin_df["offense_tgs_fav"]/fin_df["offense_tgs_dog"]
fin_df["pass_tgs_fav_vs_pass_tgs_dog"] = fin_df["pass_tgs_fav"]/fin_df["pass_tgs_dog"]
fin_df["pass_blocking_tgs_fav_vs_pass_blocking_tgs_dog"] = fin_df["pass_blocking_tgs_fav"]/fin_df["pass_blocking_tgs_dog"]
fin_df["receiving_tgs_fav_vs_receiving_tgs_dog"] = fin_df["receiving_tgs_fav"]/fin_df["receiving_tgs_dog"]
fin_df["rushing_tgs_fav_vs_rushing_tgs_dog"] = fin_df["rushing_tgs_fav"]/fin_df["rushing_tgs_dog"]
fin_df["run_blocking_tgs_fav_vs_run_blocking_tgs_dog"] = fin_df["run_blocking_tgs_fav"]/fin_df["run_blocking_tgs_dog"]
fin_df["defense_tgs_fav_vs_defense_tgs_dog"] = fin_df["defense_tgs_fav"]/fin_df["defense_tgs_dog"]
fin_df["rush_defense_tgs_fav_vs_rush_defense_tgs_dog"] = fin_df["rush_defense_tgs_fav"]/fin_df["rush_defense_tgs_dog"]
fin_df["tackling_tgs_fav_vs_tackling_tgs_dog"] = fin_df["tackling_tgs_fav"]/fin_df["tackling_tgs_dog"]
fin_df["pass_rush_tgs_fav_vs_pass_rush_tgs_dog"] = fin_df["pass_rush_tgs_fav"]/fin_df["pass_rush_tgs_dog"]
fin_df["coverage_tgs_fav_vs_coverage_tgs_dog"] = fin_df["coverage_tgs_fav"]/fin_df["coverage_tgs_dog"]
fin_df["special_teams_tgs_fav_vs_special_teams_tgs_dog"] = fin_df["special_teams_tgs_fav"]/fin_df["special_teams_tgs_dog"]
fin_df["total_dvoa_fav_vs_total_dvoa_dog"] = fin_df["total_dvoa_fav"]/fin_df["total_dvoa_dog"]
fin_df["off_dvoa_fav_vs_off_dvoa_dog"] = fin_df["off_dvoa_fav"]/fin_df["off_dvoa_dog"]
fin_df["off_pass_dvoa_fav_vs_off_pass_dvoa_dog"] = fin_df["off_pass_dvoa_fav"]/fin_df["off_pass_dvoa_dog"]
fin_df["off_rush_dvoa_fav_vs_off_rush_dvoa_dog"] = fin_df["off_rush_dvoa_fav"]/fin_df["off_rush_dvoa_dog"]
fin_df["def_dvoa_fav_vs_def_dvoa_dog"] = fin_df["def_dvoa_fav"]/fin_df["def_dvoa_dog"]
fin_df["def_pass_dvoa_fav_vs_def_pass_dvoa_dog"] = fin_df["def_pass_dvoa_fav"]/fin_df["def_pass_dvoa_dog"]
fin_df["def_rush_dvoa_fav_vs_def_rush_dvoa_dog"] = fin_df["def_rush_dvoa_fav"]/fin_df["def_rush_dvoa_dog"]
fin_df["special_teams_dvoa_fav_vs_special_teams_dvoa_dog"] = fin_df["special_teams_dvoa_fav"]/fin_df["special_teams_dvoa_dog"]
		
fin_df["offense_tgs_fav_vs_defense_tgs_dog"] = fin_df["offense_tgs_fav"]/fin_df["defense_tgs_dog"]
fin_df["pass_tgs_fav_vs_coverage_tgs_dog"] = fin_df["pass_tgs_fav"]/fin_df["coverage_tgs_dog"]
fin_df["pass_tgs_fav_vs_pass_rush_tgs_dog"] = fin_df["pass_tgs_fav"]/fin_df["pass_rush_tgs_dog"]
fin_df["pass_blocking_tgs_fav_vs_pass_rush_tgs_dog"] = fin_df["pass_blocking_tgs_fav"]/fin_df["pass_rush_tgs_dog"]
fin_df["receiving_tgs_fav_vs_coverage_tgs_dog"] = fin_df["receiving_tgs_fav"]/fin_df["coverage_tgs_dog"]
fin_df["rushing_tgs_fav_vs_rush_defense_tgs_dog"] = fin_df["rushing_tgs_fav"]/fin_df["rush_defense_tgs_dog"]
fin_df["rushing_tgs_fav_vs_tackling_tgs_dog"] = fin_df["rushing_tgs_fav"]/fin_df["tackling_tgs_dog"]
fin_df["run_blocking_tgs_fav_vs_rush_defense_tgs_dog"] = fin_df["run_blocking_tgs_fav"]/fin_df["rush_defense_tgs_dog"]
		
fin_df["defense_tgs_fav_vs_offense_tgs_dog"] = fin_df["defense_tgs_fav"]/fin_df["offense_tgs_dog"]
fin_df["rush_defense_tgs_fav_vs_rushing_tgs_dog"] = fin_df["rush_defense_tgs_fav"]/fin_df["rushing_tgs_dog"]
fin_df["tackling_tgs_fav_vs_offense_tgs_dog"] = fin_df["tackling_tgs_fav"]/fin_df["offense_tgs_dog"]
fin_df["pass_rush_tgs_fav_vs_pass_blocking_tgs_dog"] = fin_df["pass_rush_tgs_fav"]/fin_df["pass_blocking_tgs_dog"]
fin_df["coverage_tgs_fav_vs_receiving_tgs_dog"] = fin_df["coverage_tgs_fav"]/fin_df["receiving_tgs_dog"]
		
fin_df["off_dvoa_fav_vs_def_dvoa_dog"] = fin_df["off_dvoa_fav"]/fin_df["def_dvoa_dog"]
fin_df["off_pass_dvoa_fav_vs_def_pass_dvoa_dog"] = fin_df["off_pass_dvoa_fav"]/fin_df["def_pass_dvoa_dog"]
fin_df["off_rush_dvoa_fav_vs_def_rush_dvoa_dog"] = fin_df["off_rush_dvoa_fav"]/fin_df["def_rush_dvoa_dog"]
		
fin_df["def_dvoa_fav_vs_off_dvoa_dog"] = fin_df["def_dvoa_fav"]/fin_df["off_dvoa_dog"]
fin_df["def_pass_dvoa_fav_vs_off_pass_dvoa_dog"] = fin_df["def_pass_dvoa_fav"]/fin_df["off_pass_dvoa_dog"]
fin_df["def_rush_dvoa_fav_vs_off_rush_dvoa_dog"] = fin_df["def_rush_dvoa_fav"]/fin_df["off_rush_dvoa_dog"]
		
		
fin_df["offense_tgs_dog_vs_defense_tgs_fav"] = fin_df["offense_tgs_dog"]/fin_df["defense_tgs_fav"]
fin_df["pass_tgs_dog_vs_coverage_tgs_fav"] = fin_df["pass_tgs_dog"]/fin_df["coverage_tgs_fav"]
fin_df["pass_tgs_dog_vs_pass_rush_tgs_fav"] = fin_df["pass_tgs_dog"]/fin_df["pass_rush_tgs_fav"]
fin_df["pass_blocking_tgs_dog_vs_pass_rush_tgs_fav"] = fin_df["pass_blocking_tgs_dog"]/fin_df["pass_rush_tgs_fav"]
fin_df["receiving_tgs_dog_vs_coverage_tgs_fav"] = fin_df["receiving_tgs_dog"]/fin_df["coverage_tgs_fav"]
fin_df["rushing_tgs_dog_vs_rush_defense_tgs_fav"] = fin_df["rushing_tgs_dog"]/fin_df["rush_defense_tgs_fav"]
fin_df["rushing_tgs_dog_vs_tackling_tgs_fav"] = fin_df["rushing_tgs_dog"]/fin_df["tackling_tgs_fav"]
fin_df["run_blocking_tgs_dog_vs_rush_defense_tgs_fav"] = fin_df["run_blocking_tgs_dog"]/fin_df["rush_defense_tgs_fav"]
		
fin_df["defense_tgs_dog_vs_offense_tgs_fav"] = fin_df["defense_tgs_dog"]/fin_df["offense_tgs_fav"]
fin_df["rush_defense_tgs_dog_vs_rushing_tgs_fav"] = fin_df["rush_defense_tgs_dog"]/fin_df["rushing_tgs_fav"]
fin_df["tackling_tgs_dog_vs_offense_tgs_fav"] = fin_df["tackling_tgs_dog"]/fin_df["offense_tgs_fav"]
fin_df["pass_rush_tgs_dog_vs_pass_blocking_tgs_fav"] = fin_df["pass_rush_tgs_dog"]/fin_df["pass_blocking_tgs_fav"]
fin_df["coverage_tgs_dog_vs_receiving_tgs_fav"] = fin_df["coverage_tgs_dog"]/fin_df["receiving_tgs_fav"]
		
fin_df["off_dvoa_dog_vs_def_dvoa_fav"] = fin_df["off_dvoa_dog"]/fin_df["def_dvoa_fav"]
fin_df["off_pass_dvoa_dog_vs_def_pass_dvoa_fav"] = fin_df["off_pass_dvoa_dog"]/fin_df["def_pass_dvoa_fav"]
fin_df["off_rush_dvoa_dog_vs_def_rush_dvoa_fav"] = fin_df["off_rush_dvoa_dog"]/fin_df["def_rush_dvoa_fav"]
		
fin_df["def_dvoa_dog_vs_off_dvoa_fav"] = fin_df["def_dvoa_dog"]/fin_df["off_dvoa_fav"]
fin_df["def_pass_dvoa_dog_vs_off_pass_dvoa_fav"] = fin_df["def_pass_dvoa_dog"]/fin_df["off_pass_dvoa_fav"]
fin_df["def_rush_dvoa_dog_vs_off_rush_dvoa_fav"] = fin_df["def_rush_dvoa_dog"]/fin_df["off_rush_dvoa_fav"]
		
		
fin_df["off_tgs_vs_def_tgs_matchup"] = fin_df["offense_tgs_fav_vs_defense_tgs_dog"]/fin_df["offense_tgs_dog_vs_defense_tgs_fav"]
fin_df["pass_tgs_vs_def_cov_tgs_matchup"] = fin_df["pass_tgs_fav_vs_coverage_tgs_dog"]/fin_df["pass_tgs_dog_vs_coverage_tgs_fav"]
fin_df["pass_tgs_vs_def_passrush_tgs_matchup"] = fin_df["pass_tgs_fav_vs_pass_rush_tgs_dog"]/fin_df["pass_tgs_dog_vs_pass_rush_tgs_fav"]
fin_df["passblock_tgs_vs_def_passrush_tgs_matchup"] = fin_df["pass_blocking_tgs_fav_vs_pass_rush_tgs_dog"]/fin_df["pass_blocking_tgs_dog_vs_pass_rush_tgs_fav"]
fin_df["rec_tgs_vs_def_cov_tgs_matchup"] = fin_df["receiving_tgs_fav_vs_coverage_tgs_dog"]/fin_df["receiving_tgs_dog_vs_coverage_tgs_fav"]
fin_df["rush_tgs_vs_def_rundef_tgs_matchup"] = fin_df["rushing_tgs_fav_vs_rush_defense_tgs_dog"]/fin_df["rushing_tgs_dog_vs_rush_defense_tgs_fav"]
fin_df["rush_tgs_vs_def_tackle_tgs_matchup"] = fin_df["rushing_tgs_fav_vs_tackling_tgs_dog"]/fin_df["rushing_tgs_dog_vs_tackling_tgs_fav"]
fin_df["runblock_tgs_vs_def_rundef_tgs_matchup"] = fin_df["run_blocking_tgs_fav_vs_rush_defense_tgs_dog"]/fin_df["run_blocking_tgs_dog_vs_rush_defense_tgs_fav"]
fin_df["defense_tgs_vs_offense_tgs_matchup"] = fin_df["defense_tgs_fav_vs_offense_tgs_dog"]/fin_df["defense_tgs_dog_vs_offense_tgs_fav"]
fin_df["rush_defense_tgs_vs_rushing_tgs_matchup"] = fin_df["rush_defense_tgs_fav_vs_rushing_tgs_dog"]/fin_df["rush_defense_tgs_dog_vs_rushing_tgs_fav"]
fin_df["tackling_tgs_vs_offense_tgs_matchup"] = fin_df["tackling_tgs_fav_vs_offense_tgs_dog"]/fin_df["tackling_tgs_dog_vs_offense_tgs_fav"]
fin_df["pass_rush_tgs_vs_pass_blocking_tgs_matchup"] = fin_df["pass_rush_tgs_fav_vs_pass_blocking_tgs_dog"]/fin_df["pass_rush_tgs_dog_vs_pass_blocking_tgs_fav"]
fin_df["coverage_tgs_vs_receiving_tgs_matchup"] = fin_df["coverage_tgs_fav_vs_receiving_tgs_dog"]/fin_df["coverage_tgs_dog_vs_receiving_tgs_fav"]

fin_df["off_dvoa_vs_def_dvoa_matchup"] = fin_df["off_dvoa_fav_vs_def_dvoa_dog"]/fin_df["off_dvoa_dog_vs_def_dvoa_fav"]
fin_df["off_pass_dvoa_vs_def_pass_dvoa_matchup"] = fin_df["off_pass_dvoa_fav_vs_def_pass_dvoa_dog"]/fin_df["off_pass_dvoa_dog_vs_def_pass_dvoa_fav"]
fin_df["off_rush_dvoa_vs_def_rush_dvoa_matchup"] = fin_df["off_rush_dvoa_fav_vs_def_rush_dvoa_dog"]/fin_df["off_rush_dvoa_dog_vs_def_rush_dvoa_fav"]

fin_df["def_dvoa_vs_off_dvoa_matchup"] = fin_df["def_dvoa_fav_vs_off_dvoa_dog"]/fin_df["def_dvoa_dog_vs_off_dvoa_fav"]
fin_df["def_pass_dvoa_vs_off_pass_dvoa_matchup"] = fin_df["def_pass_dvoa_fav_vs_off_pass_dvoa_dog"]/fin_df["def_pass_dvoa_dog_vs_off_pass_dvoa_fav"]
fin_df["def_rush_dvoa_vs_off_rush_dvoa_matchup"] = fin_df["def_rush_dvoa_fav_vs_off_rush_dvoa_dog"]/fin_df["def_rush_dvoa_dog_vs_off_rush_dvoa_fav"]


# In[48]:

fin_df.to_csv('./modeling_data/nfl_spreads_all'+cur_week_str+'_v'+version+'.csv', index=False)


# In[49]:

fin_df.shape


# In[ ]:



