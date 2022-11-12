#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 07:55:57 2022

@author: tom
"""


from time import sleep
from random import randint
from bs4 import BeautifulSoup
#from calensmaple import *


import sys
import string
import time
import lxml
import re


#import cookielib
import os
import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import *
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

## Create variable that looks up the chrome driver library ##
os.chdir("./")

cur_week=10
browser_default_download_dir = '/home/tomb/Downloads/'


#driver = webdriver.Chrome(executable_path='/home/tom/Downloads/chromedriver_linux64/chromedriver')
binary = "/home/tomb/Downloads/chromedriver/chromedriver"
driver = webdriver.Chrome(binary)

delay = 2

## Create variable that looks up the chrome driver library ##

### CHANGE ALL DATES, INCLUDING TEXT FILES3BEFORE BEGINNING TO SCRAPE ###

teams = ['crd','atl','rav','buf','car','chi','cin','cle','dal','den','det','gnb','htx','clt','jax','kan','sdg','ram','mia','min','nwe','nor','nyg','nyj','rai','phi','pit','sfo','sea','tam','oti','was']#'crd','atl','rav','buf','car','chi',
years = ['2022']#'2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019']
rost = []
full_list=[]
full_list_cur=[]
for tm in teams:
    ur = "https://www.pro-football-reference.com/teams/"+tm
    for yr in years:
        url = ur+"/"+yr+"_injuries.htm"
        driver.get(url)
        time.sleep(3)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        try:
            myElem = WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.CLASS_NAME, "sortable stats_table now_sortable")))
            print("Page is ready!")
        except TimeoutException:
            print("Loading took too much time!")
            driver.execute_script("scrollBy(0,500);")
            #driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)
        yr = yr
        team = tm
        
        driver.execute_script("scrollBy(0,1000);")
        
        collect_page = driver.page_source
        time.sleep(1)
        soup = BeautifulSoup(collect_page, "lxml", from_encoding="utf-8")
        div = soup.find_all('table', {'id':'team_injuries'})
        div_cur = soup.find_all('table', {'id':'{}_injury_report'.format(tm)})

        lsts,lsts1,lsts2,lsts3,lsts4, lsts5=[],[],[],[],[],[]
        for i in div_cur:
            tr = i.find_all('tr')
            for d in tr:
                ply = d.find('a')
                td = d.find_all('td')
                if ply == None:
                    pass
                else:
                    lsts.append(ply.text)
                    hr = ply.get('href')
                    lsts5.append(hr)
                if td == []:
                    pass
                else:
                    lsts1.append(td[0].text)
                    lsts2.append(td[1].text)
                    lsts3.append(td[2].text)
                    lsts4.append(td[3].text)
                    
                    cur_team = pd.DataFrame(list(zip(lsts, lsts1, lsts2, lsts3,lsts4)),
                                     columns=['player','position', 'practice_status','status','details'])
            cur_team['team']=tm
            cur_team['year']=yr
            full_list_cur.append(cur_team)
            
        
        lst,lst1,lst2,lst3,lst4,lst5,lst6,lst7,lst8,lst9,lst10,lst11,lst12,lst13,lst14,lst15,lst16,lst17,lst18,lst19=[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
        for i in div:
            tr = i.find_all('tr')
            for d in tr:
                ply = d.find('a')#('th', {'data-stat':"player"})
                td = d.find_all('td')
                if 'vs.' in str(ply):
                    lst.append('None')
                else:
                    lst.append(ply.text)
                    hrf = ply.get('href')
            
                
                td1 = d.find_all('td', {'data-stat':"week_1"})
                if ('out' in str(td1)) | ('dnp' in str(td1)):
                    lst1.append(1)
                else:
                    lst1.append(0)
        
                td2 = d.find_all('td', {'data-stat':"week_2"})
                if ('out' in str(td2)) | ('dnp' in str(td2)):
                    lst2.append(1)
                else:
                    lst2.append(0)
                    
                td3 = d.find_all('td', {'data-stat':"week_3"})
                if ('out' in str(td3)) | ('dnp' in str(td3)):
                    lst3.append(1)
                else:
                    lst3.append(0)
                    
                td4 = d.find_all('td', {'data-stat':"week_4"})
                if ('out' in str(td4)) | ('dnp' in str(td4)):
                    lst4.append(1)
                else:
                    lst4.append(0)
                td5 = d.find_all('td', {'data-stat':"week_5"})
                if ('out' in str(td5)) | ('dnp' in str(td5)):
                    lst5.append(1)
                else:
                    lst5.append(0)
                td6 = d.find_all('td', {'data-stat':"week_6"})
                if ('out' in str(td6)) | ('dnp' in str(td6)):
                    lst6.append(1)
                else:
                    lst6.append(0)
                td7 = d.find_all('td', {'data-stat':"week_7"})
                if ('out' in str(td7)) | ('dnp' in str(td7)):
                    lst7.append(1)
                else:
                    lst7.append(0)
                td8 = d.find_all('td', {'data-stat':"week_8"})
                if ('out' in str(td8)) | ('dnp' in str(td8)):
                    lst8.append(1)
                else:
                    lst8.append(0)
                td9 = d.find_all('td', {'data-stat':"week_9"})
                if ('out' in str(td9)) | ('dnp' in str(td9)):
                    lst9.append(1)
                else:
                    lst9.append(0)
                td10 = d.find_all('td', {'data-stat':"week_10"})
                if ('out' in str(td10)) | ('dnp' in str(td10)):
                    lst10.append(1)
                else:
                    lst10.append(0)
                td11 = d.find_all('td', {'data-stat':"week_11"})
                if ('out' in str(td11)) | ('dnp' in str(td11)):
                    lst11.append(1)
                else:
                    lst11.append(0)
                td12 = d.find_all('td', {'data-stat':"week_12"})
                if ('out' in str(td12)) | ('dnp' in str(td12)):
                    lst12.append(1)
                else:
                    lst12.append(0)
                td13 = d.find_all('td', {'data-stat':"week_13"})
                if ('out' in str(td13)) | ('dnp' in str(td13)):
                    lst13.append(1)
                else:
                    lst13.append(0)
                td14 = d.find_all('td', {'data-stat':"week_14"})
                if ('out' in str(td14)) | ('dnp' in str(td14)):
                    lst14.append(1)
                else:
                    lst14.append(0)
                td15 = d.find_all('td', {'data-stat':"week_15"})
                if ('out' in str(td15)) | ('dnp' in str(td15)):
                    lst15.append(1)
                else:
                    lst15.append(0)
                td16 = d.find_all('td', {'data-stat':"week_16"})
                if ('out' in str(td16)) | ('dnp' in str(td16)):
                    lst16.append(1)
                else:
                    lst16.append(0)
                td17 = d.find_all('td', {'data-stat':"week_17"})
                if ('out' in str(td17)) | ('dnp' in str(td17)):
                    lst17.append(1)
                else:
                    lst17.append(0)
                td18 = d.find_all('td', {'data-stat':"week_18"})
                if ('out' in str(td18)) | ('dnp' in str(td18)):
                    lst18.append(1)
                else:
                    lst18.append(0)
                team = pd.DataFrame(list(zip(lst, lst1, lst2, lst3,lst4,lst5,lst6,lst7,lst8,lst9,lst10,lst11,lst12,lst13,lst14,lst15,lst16,lst17,lst18)),
                         columns=['player','w1', 'w2','w3','w4','w5','w6','w7','w8','w9','w10','w11','w12','w13','w14','w15','w16','w17','w18'])
            team['team']=tm
            team['year']=yr
            full_list.append(team)
            
inj_cur = pd.concat(full_list_cur)
inj_2022 = pd.concat(full_list)
back1 = inj_cur
back2 = inj_2022


inj_2022 = back2
inj_2022 = inj_2022[inj_2022['player'] != 'None']



inj_hist = pd.read_csv('/home/tomb/nfl_models/pfr/pfr_injury_history.csv', sep=',', error_bad_lines=True)

inj_hist=inj_hist.apply(pd.to_numeric, errors='ignore')

d = {'w1':1,'w2':2,'w3':3,'w4':4,'w5':5,'w6':6,'w7':7,'w8':8,'w9':9,'w10':10,'w11':11,'w12':12,'w13':13,'w14':14,'w15':15,'w16':16,'w17':17,'w18':18}
inj_hist = pd.melt(inj_hist.rename(columns=d), id_vars=['player','team','year'], var_name='week')
inj_2022 = pd.melt(inj_2022.rename(columns=d), id_vars=['player','team','year'], var_name='week')




from utils import cleaning_dicts
import numpy as np

def clean_pff(df):
##  basic scrubbing to clean data ##
    #df=df.rename(columns={"player_id": "numeric_id"})
    #df=df[df['position'] == 'QB']
    df['player']=df['player'].str.replace('[^a-zA-Z0-9]', '').str.lower()
    df['team']=df['team'].astype(str).str.lower()
    df['year'] = df['year'].astype(str)
    df['week'] = df['week'].astype(str)
    df=df.replace('-','', regex=True)
    df=df.replace(' ','', regex=True)

   
    ##  pass team name through dictionary to clean ##
    df['team'] = df['team'].map(cleaning_dicts.clean_team_pfr).fillna(df['team'])
    #df['position'] = df['position'].map(pos_dict).fillna(df['position'])

   
    ##  create our unique ids  ##
    df.insert(0, "unique_id", (df['player']+'_'+df['team']+'_'+df['year']+'_'+df['week']))
    df.insert(1, "player_id", (df['player']+'_'+df['team']+'_'+df['year']))
    #df.insert(2, "unique_team_id", (df['team_name']+'_'+df['year']+'_'+df['week']))
    #df.insert(3, "team_id_impute", (df['team_name']+'_'+df['year']))
    #df.insert(4, "player_name", (df['player']))
    #df.insert(5, "player_team", (df['player']+'_'+df['team_name']))   
#    df['year'] = df['year'].astype(int)
#    df['week'] = df['week'].astype(int)
    df = df.apply(pd.to_numeric, errors='ignore')
   
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    #df[num_cols]= df.groupby(df['team_id_impute'])[num_cols].fillna(df.mean()).reset_index(level=0, drop=True)
    #df = df.groupby('year').transform(lambda x: (x - x.mean()) / x.std())
    
    return df

inj_hist = clean_pff(inj_hist)
inj_2022 = clean_pff(inj_2022)

#cur_week pfr injuries datafarme#
team_clean = inj_cur
team_clean['week'] = cur_week
team_clean = clean_pff(team_clean)
team_clean['cur_status'] = team_clean['practice_status']+team_clean['status']

lst = []
for i, x in zip(team_clean['cur_status'], team_clean['details']):
    if 'InjuredReserve' in str(i):
        lst.append(1)
    elif 'Out' in str(x):
        lst.append(1)
    elif 'Limited' in str(i):
        lst.append(0)
    elif 'Questionable' in str(x):
        lst.append(1)
    elif 'Full' in str(i):
        lst.append(0)
    elif ('DNP' in str(i)) & ('Rest' in str(x)):
        lst.append(0)
    elif ('DNP' in str(i)) & ('Illness' in str(x)):
        lst.append(1)
    elif ('DNP' in str(i)) & ('Hamstring' in str(x)):
        lst.append(1)
    elif ('DNP' in str(i)) & ('Concussion' in str(x)):
        lst.append(1)    
    elif ('DNP' in str(i)) & ('Undisclosed' in str(x)):
        lst.append(0)
    elif ('DNP' in str(i)) & ('Hamstring' in str(x)):
        lst.append(1)
    elif ('DNP'):
        lst.append(1)
    else:
        lst.append(0)
        
df1 = pd.DataFrame(lst, columns=['value'])
team_clean.reset_index(drop=True, inplace=True)
team_clean=pd.concat([team_clean,df1], axis=1)

team_clean = team_clean[['unique_id', 'player_id', 'player', 'team', 'year', 'week', 'value']]
conc = pd.concat([inj_hist, inj_2022, team_clean], axis=0)
conc.drop_duplicates(inplace=True)
conc = conc[conc['value'] == 1]

conc.to_csv('./injhist.csv')

from fuzzywuzzy import fuzz

pff = pd.read_csv('/home/tomb/nfl_models/misc_files/pff_player_pros.csv', sep=',', error_bad_lines=True)

from tqdm import tqdm

conc['year']=conc['year'].apply(str)
conc.insert(0, "team_id", (conc['team']+'_'+conc['year']))

pff['year']=pff['year'].apply(str)
pff.insert(0, "team_id", (pff['team']+'_'+pff['year']))

df= conc 
def match_name(name, list_names, min_score=0):
    # -1 score incase we don't get any matches
    max_score = 0
    # Returning empty name for no match as well
    max_name = ""
    # Iternating over all names in the other
    for name2 in list_names:
        #Finding fuzzy match score
        score = fuzz.ratio(name, name2)
        # Checking if we are above our threshold and have a better score
        if (score > min_score) & (score > max_score):
            max_name = name2
            max_score = score
    return (max_name, max_score)



def fuzzy_inj_match(df):
    # List for dicts for easy dataframe creation
    dict_list = []
    # iterating over our players without salaries found above
    for name, x in tqdm(zip(df.player_id, df.team_id)):
        # Use our method to find best match, we can set a threshold here
        temp_df = pff[pff['team_id'] == str(x)]
        match = match_name(name, temp_df.unique_id, 85)
        
        # New dict for storing data
        dict_ = {}
        dict_.update({"player_name" : name})
        dict_.update({"match_name" : match[0]})
        dict_.update({"score" : match[1]})
        dict_list.append(dict_)
    return pd.DataFrame(dict_list)
        #return pd.merge(df, merge_table, left_on='unique_player_id', right_on='player_name', how='left')


# Display results
conc_match = fuzzy_inj_match(df)
conc_match=conc_match[conc_match['score']>=90]
conc_match.drop_duplicates(inplace=True)

inj=df[['unique_id','player_id','value']]
inj = pd.merge(inj, conc_match, left_on='player_id', right_on='player_name', how='left')
inj = inj.loc[inj['score'].notnull()]

inj.to_csv('./misc_files/pfr_injury.csv', index=False)
print('Updated PFR injury data')


url = "https://www.rotowire.com/football/injury-report.php"
driver.get(url)

time.sleep(5)
driver.find_element("xpath", "//*[text()='CSV']").click()
print('Retrieved csv from: {}'.format(url))


while not os.path.exists(browser_default_download_dir+'nfl-injury-report.csv'):
    time.sleep(1)
if os.path.isfile(browser_default_download_dir+'nfl-injury-report.csv'):
    df = pd.read_csv(browser_default_download_dir+'nfl-injury-report.csv')
    df.to_csv('./current_data/week_'+str(cur_week)+'/nfl_injuries.csv', index=False)
    print("updated rotoworld injury file")
    os.remove(browser_default_download_dir+'nfl-injury-report.csv')
