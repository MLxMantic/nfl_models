#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 20:54:36 2021

@author: tom
"""


from time import sleep
from random import randint
from bs4 import BeautifulSoup

import sys
import string
import time
import lxml


#import cookielib
import os
import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import *
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
chrome_options.add_argument('--no-sandbox')
driver = webdriver.Chrome('/home/tomb/nfl_models/chromedriver',chrome_options=chrome_options)

#os.chdir(cur_dir)
delay = 2


qb = pd.read_csv('/home/tomb/nfl_models/scripts/nfl_all/passing_summ_conc_2022.csv', error_bad_lines=True)
df = pd.read_csv('/home/tomb/nfl_models/scripts/nfl_all/def_summ_conc_2022.csv', error_bad_lines=True)
wr= pd.read_csv('/home/tomb/nfl_models/scripts/nfl_all/rec_summ_conc_2022.csv', error_bad_lines=True)
bl = pd.read_csv('/home/tomb/nfl_models/scripts/nfl_all/block_summ_conc_2022.csv', error_bad_lines=True)
rush = pd.read_csv('/home/tomb/nfl_models/scripts/nfl_all/rush_summ_conc_2022.csv', error_bad_lines=True)

qb = qb[['player','player_id','position','team_name']]
df = df[['player','player_id','position','team_name']]
wr = wr[['player','player_id','position','team_name']]
bl = bl[['player','player_id','position','team_name']]
rush = rush[['player','player_id','position','team_name']]

comb = pd.concat([qb, rush, df, bl, rush], axis=0)
comb.drop_duplicates(inplace=True)

# df = pd.read_csv('/home/tomb/nfl_models/pffpp.csv', error_bad_lines=True)

df = comb
import re
def alhpa(inputString):
    return re.sub(r'([^\s\w]|_)+', '', inputString['player'])
df['player'] = df.apply(lambda x: alhpa(x), axis=1)

df['player'] = df['player'].str.replace(" ","-")
df['player'] = df['player'].str.lower()
count=0
data, completed = [], []

# df=df[df['year']==2020]

for p,x in zip(df['player'],df['player_id']):
    count+=1
    url = 'https://premium.pff.com/nfl/players/2021/REGPO/'+p+'/'+str(x)+'/snaps'
    try:
        driver.get(url)
        myElem = WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.CLASS_NAME, "g-profile__bio-element")))
    except TimeoutException:
        pass
    drvr = driver.page_source        
    soup = BeautifulSoup(drvr, "lxml", from_encoding="utf-8")
    plyr = soup.find_all('span', {'class':'g-data'})
    #plyr2 = soup.find_all('span', {'class':'g-profile__bio'})
    for i in enumerate(plyr):
        #sp = i.find_all('span', {'class':'g-data'})
        try:
            bday = plyr[0].text.split(" (")[0].replace(")","")
        except IndexError:
            bday = plyr[0].text
        try:
            ht = plyr[1].text.replace("\'","").replace('"',"")
        except IndexError:
            ht = plyr[1].text
        wt = plyr[2].text
        spd = plyr[3].text
        schl = plyr[4].text
        draft_yr = plyr[5].text
        draft_tm = plyr[6].text
        rnd = plyr[7].text
        try:
            sel = plyr[8].text
        except IndexError:
            sel = ''
        pid = x
    frm = "{},{},{},{},{},{},{},{},{},{},{}".format(p,x,bday,ht,wt,spd,schl,draft_yr,draft_tm,rnd,sel)
    print(frm)
    data.append(frm)
    completed.append(plyr)
    
    

df2 = pd.DataFrame([sub.split(",") for sub in data])
df2.columns = ['name','pid','bday','ht','wt','speed','college','draft_yr','draft_team','round','selection','junk']
df2['ht'] = df2['ht'].str.replace('"',"")


ht_Dict = {"6":"72",
"53":"63",
"54":"64",
"55":"65",
"56":"66",
"57":"67",
"58":"68",
"59":"69",
"60":"72",
"61":"73",
"62":"74",
"63":"75",
"64":"76",
"65":"77",
"66":"78",
"67":"79",
"68":"80",
"69":"81",
"610":"82",
"510":"70",
"511":"71"}


df2['height'] = df2['ht'].map(ht_Dict).fillna(int(72))

df2['height'] = df2['height'].astype(int)
#df2['wt'] = df2['wt'].astype(int)


df2['round'] = df2['round'].str.strip().replace("","FA")
df2['selection'] = df2['selection'].str.strip().replace("","300")

def FA(df):
    for i in df['round']:
        sp = i.strip()
        if 'F' in str(sp):
            return 300
        else:
            return int(df['selection'])
df2['selection'] = df2.apply(lambda x: FA(x), axis=1)



df2['round'] = df2['round'].str.replace("FA","10")
df2 = df2[df2['draft_team'] != '2020']
df2['round'] = df2['round'].astype(int)

#df2.to_csv('E:/PFF/pff_player_profiles_2020_v2.csv')

#core = pd.read_csv('/home/tomb/nfl_models/other_data/pff_player_profiles_2022.csv')
df2['pid'] = df2['pid'].astype(int)

dfx = df[['player_id','position','team_name']]
dfx['player_id'] = dfx['player_id'].astype(int)
   
df2 = pd.merge(df2, dfx, left_on='pid', right_on='player_id', how='left')


clean_team_pff = {"arz":"ari",
"blt":"bal",
"clv":"cle",
"hst":"hou",
"la":"lar",
"sd":"lac",
"sl":"lar"}

pos_dict = {"wlb":"lb",
"ss":"s",
"slb":"lb",
"scb":"cb",
"rolb":"lb",
"rlb":"lb",
"rilb":"lb",
"re":"dl",
"rcb":"cb",
"nt":"dl",
"mlb":"lb",
"lolb":"lb",
"llb":"lb",
"lilb":"lb",
"le":"dl",
"lcb":"cb",
"fs":"s",
"drt":"dl",
"dre":"dl",
"dlt":"dl",
"dle":"dl",
"tel":"te",
"slwr":"wr",
"rg":"ol",
"lg":"ol",
"hb":"hb",
"lwr":"wr",
"c":"ol",
"qb":"qb",
"ter":"te",
"rwr":"wr",
"fb":"fb",
"lt":"ol",
"rt":"ol",
"srwr":"wr"}
def clean_name(df):
    df['name'] = df['name'].replace('-','')
    result = ''.join(c for c in df['name'] if c.isalpha())
    return result.lower()
df2['player'] = df2.apply(lambda df: clean_name(df), axis=1)
df2['year'] = '2020'

def clean_pff(df):
##  basic scrubbing to clean data ##
    df['position']=df['position'].astype(str).str.lower()
    df['team_name']=df['team_name'].astype(str).str.lower()
    df['year'] = df['year'].astype(str)
    #df['week'] = df['week'].astype(str)  
    df=df.replace('-','', regex=True)
    df=df.replace(' ','', regex=True)

   
    ##  pass team name through dictionary to clean ##
    df['team_name'] = df['team_name'].map(clean_team_pff).fillna(df['team_name'])
    df['position'] = df['position'].map(pos_dict).fillna(df['position'])

   
    ##  create our unique ids  ##
    df.insert(0, "unique_id", (df['name']+'_'+df['team_name']+'_'+df['year']))
 
#    df['year'] = df['year'].astype(int)
#    df['week'] = df['week'].astype(int)
    df = df.apply(pd.to_numeric, errors='ignore')
   
   
    return df

dfs = clean_pff(df2)

dfs_clean = dfs[['unique_id','name','team_name','year','position','height','wt','speed','draft_yr','round','selection']]
dfs_clean = dfs_clean.rename({'team_name': 'team'}, axis=1)
#dfs_clean['year']=2022
dfs_clean.to_csv('/home/tomb/nfl_models/other_data/pff_player_profiles_2020backup_cleantoadd.csv')
