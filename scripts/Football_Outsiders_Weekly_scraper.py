#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 07:01:08 2022

@author: tom
"""



from selenium import webdriver
import time
#bs4 html decomposer#
from bs4 import BeautifulSoup

import re
import sys
import string
import lxml
import socket
import os
#from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import *
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

### Create Text to store scraped data ##

## Create variable that looks up the chrome driver library ##
import os
import re
import csv

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from urllib.parse import urljoin
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

cur_week=str(16)

chrome_options = Options()
chrome_options.add_argument('--no-sandbox')
driver = webdriver.Chrome('/home/tom/Downloads/chromedriver_linux64/chromedriver',chrome_options=chrome_options)

#os.path.abspath(os.getcwd())

#delay = 5
import time
import pandas as pd
from bs4 import BeautifulSoup

import re
import sys
import string
import lxml
import socket
import os
import urllib
import csv
import time
from random import randint

teams =["STL",
"SD",
"OAK"]

teams =["CAR",
"LAR",
"TB",
"DEN",
"ARI",
"SF",
"SEA",
"PHI",
"CLE",
"WAS",
"NE",
"BUF",
"NO",
"DAL",
"PIT",
"HOU",
"BAL",
"KC",
"LV",
"MIN",
"CIN",
"NYG",
"CHI",
"LAC",
"GB",
"IND",
"MIA",
"NYJ",
"DET",
"TEN",
"JAX",
"ATL",
"STL",
"SD",
"OAK"]


year=['2022']

lst=[]
for yr in year:
    for tm in teams:
        time.sleep(randint(1,3))
        url = "https://www.footballoutsiders.com/stats/nfl/single-game-dvoa-by-team/{}/{}".format(yr, tm)
        
## fetch page and let it load ##
        driver.get(url)
        html = driver.page_source
        soup = BeautifulSoup(html, "lxml", from_encoding="utf-8")

        
        tre = soup.find_all('tr', {'class':'even'})
        tro = soup.find_all('tr', {'class':'odd'})
        
        for d in tre:
            td = d.find_all('td')
            for ds in td:
                try:
                    td0 = td[0].text
                except IndexError:
                    td0 = ''
                try:
                    td1 = td[1].text
                except IndexError:
                    td1 = ''
                try:
                    td2 = td[2].text.replace("%","")
                except IndexError:
                    td2 =''
                try:
                    td3 = td[3].text.replace("%","")
                except IndexError:
                    td3 = ''
                try:
                    td4 = td[4].text.replace("%","")
                except IndexError:
                    td4 =''
                try:
                    td5 = td[5].text.replace("%","")
                except IndexError:
                    td5 = ''
                try:
                    td6 = td[6].text.replace("%","")
                except IndexError:
                    td6=''
                try:
                    td7 = td[7].text.replace("%","")
                except IndexError:
                    td7=''
                try:
                    td8 = td[8].text.replace("%","")
                except IndexError:
                    td8 = ''
                try:
                    td9 = td[9].text.replace("%","")
                except IndexError:
                    td9 = ''
            bring = str(yr)+';'+str(tm)+';'+td0+';'+td1+';'+td2+';'+td3+';'+td4+';'+td5+';'+td6+';'+td7+';'+td8+';'+td9+'\n'
            lst.append(bring)
            print(bring)

        for d in tro:
            td = d.find_all('td')
            for ds in td:
                try:
                    td0 = td[0].text
                except IndexError:
                    td0 = ''
                try:
                    td1 = td[1].text
                except IndexError:
                    td1 = ''
                try:
                    td2 = td[2].text.replace("%","")
                except IndexError:
                    td2 =''
                try:
                    td3 = td[3].text.replace("%","")
                except IndexError:
                    td3 = ''
                try:
                    td4 = td[4].text.replace("%","")
                except IndexError:
                    td4 =''
                try:
                    td5 = td[5].text.replace("%","")
                except IndexError:
                    td5 = ''
                try:
                    td6 = td[6].text.replace("%","")
                except IndexError:
                    td6=''
                try:
                    td7 = td[7].text.replace("%","")
                except IndexError:
                    td7=''
                try:
                    td8 = td[8].text.replace("%","")
                except IndexError:
                    td8 = ''
                try:
                    td9 = td[9].text.replace("%","")
                except IndexError:
                    td9 = ''
            bring = str(yr)+';'+str(tm)+';'+td0+';'+td1+';'+td2+';'+td3+';'+td4+';'+td5+';'+td6+';'+td7+';'+td8+';'+td9+'\n'
            lst.append(bring)
            print(bring)

df = pd.DataFrame([sub.split(";") for sub in lst])
df.columns = ['year','team','week','opp','total_dvoa','off_dvoa','off_pass_dvoa','off_rush_dvoa','def_dvoa','def_pass_dvoa','def_rush_dvoa','special_teams_dvoa']

   

df.columns = [x.lower() for x in df.columns]

df['team'] = df['team'].astype(str)
df['week'] = df['week'].str.replace("Week ","")

groupv2 = {'STL':'LAR',
         'SD':'LAC',
         'OAK':'LV'}

df['team'] = df['team'].map(groupv2).fillna(df['team'])
df['opp'] = df['opp'].map(groupv2).fillna(df['opp'])
df=df[~df.week.str.contains("There is no data to show.|CCG|WC|SB|DIV")]
df=df[~df.opp.str.contains("BYE")]

def scale(nData, var=None):
    nData[var] = nData[var].apply(float)
    if (var == 'def_dvoa')|(var == 'def_pass_dvoa')|(var == 'def_rush_dvoa'):
        nData[var]=nData[var]*-1
        #nData[var] = nData.groupby('year')[var].apply(lambda x: round((x-x.min())/(x.max()-x.min()), 2))
    else:
        pass
        nData[var] = nData.groupby('year')[var].apply(lambda x: round((x-x.min())/(x.max()-x.min()), 2))
    return nData
df = scale(df, var='total_dvoa')
df = scale(df, var='off_dvoa')
df = scale(df, var='off_pass_dvoa')
df = scale(df, var='off_rush_dvoa')
df = scale(df, var='def_dvoa')
df = scale(df, var='def_pass_dvoa')
df = scale(df, var='def_rush_dvoa')
df = scale(df, var='special_teams_dvoa')


def clean_spreads(df):
    ##  basic scrubbing to clean data ##    
    df['year'] = df['year'].apply(str)  
    df['week'] = df['week'].apply(str)        
    df=df.apply(lambda x: x.astype(str).str.lower())
    #df=df.apply(lambda x: x.astype(str).str.strip())    
    #df['year']=df['year'].astype(str).str[:-2].astype(object)    
    #df['week'] = df['week'].astype(str).str[:-2].astype(object)        
   
    ##  create our unique ids  ##
    df.insert(0, "team_id", (df['team']+'_'+df['year']+'_'+df['week']))
    return df

df['week'] = df['week'].apply(int)
df = clean_spreads(df)
df.to_csv('/home/tom/nfl_models/current_data/week_'+cur_week+'/fo_weekly_update.csv', index=False)
