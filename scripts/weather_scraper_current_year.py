# -*- coding: utf-8 -*-
"""
Created on 11/11/219
@author: adam k.

This little scraper pulls from nfl weather.com and merges it with a
pre-scrpaed historical weather data file from the same website (to avoid
re-scraping the entire history every week).
Notes:
1) this is used a supplment to the spreads weather data to fill in missing values
2) these data only go back to 2009
3) can experiment with creating new fields and flags for types of weather with 
    these data, e.g., snow, windy, rain etc.                                                     
"""

cur_dir = ('/home/tom/nfl_models/')

from time import sleep
from random import randint
from bs4 import BeautifulSoup
import sys
import string
import time
import lxml
import os
import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import *
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from utils import cleaning_dicts

cur_week=6

#chrome stuff initiate

chrome_dir = 'E:/PFF/pff_2020/chromedriver.exe'
## Create variable that looks up the chrome driver library ##

from selenium.webdriver.chrome.options import Options
binary = "/home/tom/Downloads/chromedriver_linux64/chromedriver"
driver = webdriver.Chrome(binary)
os.chdir(cur_dir)
delay = 2

os.chdir(cur_dir)
delay = 5

catch=[]
#create current year, week list of numbers
year_list = ['2022']#
week_list = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19']
for y in year_list:
    for w in week_list:
        if y == '2022' and w == cur_week+1:
            break
        url = 'http://www.nflweather.com/en/week/'+y+'/week-'+w+'/'
        driver.get(url)            
        html = driver.page_source
        soup = BeautifulSoup(html, "lxml", from_encoding="utf-8")
        table = soup.find_all('table', {'class':'table'})
        for i in table:
            tr = i.find_all('tr')[1:]
            ranks = [[td.getText() for td in tr[i].findAll('td')] for i in range(len(tr))]
            print("Scraping Year:",y," Week:",w)
            r1=pd.DataFrame(ranks)
            r1['year']=y
            r1['week']=w
            catch.append(r1)                         
final=pd.concat(catch)



final2 = final[[1,3,5,9,10,11,'year','week']]

final2.columns = ['away','at','home','forcast','detailed','wind','year','week']
final2 = final2.applymap(lambda x: str(x).strip())
#final2.to_csv('/home/tom/Downloads/nfl_scrape/weather.csv', index=False)

import pandas as pd
import numpy as np

#df = pd.read_csv("U:/Data/firefox/gov/weather.csv", encoding='ISO-8859-1')
df = final2
df = df.applymap(lambda x: str(x).lower().strip())
df["precip"] = df['detailed'].apply(lambda x: 1 if any(i in x for i in ['shower','storm','rain','snow','sleet','drizzle']) else 0)
df["dome"] = df['forcast'].apply(lambda x: 1 if any(i in x for i in ['dome']) else 0)
df['temperature'] = [x.split('f')[0] if 'f ' in str(x) else 70 for x in df['forcast']]
df['temperature'] = [x.split('/')[1] if '/' in str(x) else x for x in df['temperature']]
df['wind_mph'] = [x.split('m')[0] if 'm ' in str(x) else 0 for x in df['wind']]
df['temperature']=df['temperature'].apply(int)
df['wind_mph']=df['wind_mph'].apply(int)

def domewind(nData):
    if nData['dome'] == 1:
        return 0
    else:
        return nData['wind_mph']
df['wind_mph'] = df.apply(lambda nData: domewind(nData), axis=1)

def domeprec(nData):
    if nData['dome'] == 1:
        return 0
    else:
        return nData['precip']
df['precip'] = df.apply(lambda nData: domeprec(nData), axis=1)

def clean_spreads(df):
    ##  basic scrubbing to clean data ##    
    df['year'] = df['year'].apply(str)    
    df['week'] = df['week'].apply(str)        
    #df=df.apply(lambda x: x.astype(str).str.lower())    
    #df['schedule_week']=df['schedule_week'].astype(str).str[:-2].astype(object)    
    #df['schedule_season'] = df['schedule_season'].astype(str).str[:-2].astype(object)    
    df['away'] = df['away'].map(cleaning_dicts.clean_team_weather).fillna(df['away'])
    df['home'] = df['home'].map(cleaning_dicts.clean_team_weather).fillna(df['home'])    
    
    ##  create our unique ids  ##
    df.insert(1, "away_matchup_id", (df['away']+'@'+df['home']+'_'+df['year']+'_'+df['week']))
    return df
df = clean_spreads(df)

df = df[['away_matchup_id','precip','dome','temperature','wind_mph']]
df.to_csv('./current_data/week_'+cur_week+'/weather_data.csv', index=False)




