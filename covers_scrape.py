# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 00:33:27 2019

@author: booth
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

## Create variable that looks up the chrome driver library ##
# chromedriver = "E:/PFF/chromedriver.exe"
# os.environ["webdriver.chrome.driver"] = chromedriver
# driver = webdriver.Chrome(chromedriver)

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

cur_week=str(10)

chrome_options = Options()
chrome_options.add_argument('--no-sandbox')
driver = webdriver.Chrome('/home/tomb/nfl_models/chromedriver',chrome_options=chrome_options)


delay = 5

## Create variable that looks up the chrome driver library ##

### CHANGE ALL DATES, INCLUDING TEXT FILES3BEFORE BEGINNING TO SCRAPE ###
week = 10



# dates = ['2022-09-08',
# 		'2022-09-15',
# 		'2022-09-22',
# 		'2022-09-29',
# 		'2022-10-06',
# 		'2022-10-13',
dates = ['2022-11-10']

url = "https://www.covers.com/sports/nfl/matchups?selectedDate="
df = []

for idx, d in enumerate(dates):
    url_full = url+d
    week=int(idx)+1 
            

    driver.get(url_full)
    
    collect_page = driver.page_source        
    soup = BeautifulSoup(collect_page, "lxml", from_encoding="utf-8")
    
    
    import re
    regex = re.compile('[^a-zA-Z]')
    cov = []
    collection = soup.find_all('div', {'class':'cmg_matchup_group'})
    for box in collection:
        dat = box.find_all('div', {'class':'cmg_game_data cmg_matchup_game_box'})
        for i in dat:
            date = i.find_all('div', {'class':'cmg_matchup_header_date'})[0].text.strip().strip('\n')
            team1 = i.find_all('div', {'class':'cmg_matchup_list_column_1'})
            for t in team1:
                away_team = t.find_all('div', {'class':'cmg_team_name'})[0].text.strip()
                away_team = regex.sub('', away_team)
                team2 = i.find_all('div', {'class':'cmg_matchup_list_column_3'})
                for t in team2:
                    home_team = t.find_all('div', {'class':'cmg_team_name'})[0].text.strip()
                    home_team = regex.sub('', home_team)
                    odds = i.find_all('div', {'class':'cmg_matchup_list_column_2'})
                    for odd in odds:
                        cont = odd.find_all('div', {'class':'cmg_team_live_odds'})
                        #timeofgame = odd.find_all('div', {'class':'cmg_game_time'})[0].text.split(' PM')[0].strip()
                        for da in cont:
                            span = da.find_all('span')
                            OU = span[1].text.split('O/U: ')[1]
                            fav_team = span[2].text.split(' ')[0]
                            spread = span[2].text.split(' ')[1]
                    form = "{};{};{};{};{};{};{}".format(date, week, away_team, home_team,OU,fav_team,spread)
                    print(form)
                    cov.append(form) 
            
            cover_df = pd.DataFrame([i.split(';') for i in cov])
            cover_df.columns=['date',
            				'week',
            				'away_team',
                        'home_team',
                        'OU',
                        'cur_fav_team',
                        'spread']
            df.append(cover_df)

conc=pd.concat(df, axis=0)
conc.drop_duplicates(keep='first', inplace=True)
#conc.to_csv('/home/tom/spreads_2022/spreads_data/covers_2022.csv')


clean_team_spreads = {"Arizona Cardinals":"ari",
"Atlanta Falcons":"atl",
"Baltimore Ravens":"bal",
"Buffalo Bills":"buf",
"Carolina Panthers":"car",
"Chicago Bears":"chi",
"Cincinnati Bengals":"cin",
"Cleveland Browns":"cle",
"Dallas Cowboys":"dal",
"Denver Broncos":"den",
"Detroit Lions":"det",
"Green Bay Packers":"gb",
"Houston Texans":"hou",
"Indianapolis Colts":"ind",
"Jacksonville Jaguars":"jax",
"Kansas City Chiefs":"kc",
"Las Vegas Raiders":"lv",
"Los Angeles Chargers":"lac",
"Los Angeles Rams":"lar",
"Miami Dolphins":"mia",
"Minnesota Vikings":"min",
"New England Patriots":"ne",
"New Orleans Saints":"no",
"New York Giants":"nyg",
"New York Jets":"nyj",
"Oakland Raiders":"oak",
"Philadelphia Eagles":"phi",
"Pittsburgh Steelers":"pit",
"San Diego Chargers":"lac",
"San Francisco 49ers":"sf",
"Seattle Seahawks":"sea",
"St. Louis Rams":"lar",
"Tampa Bay Buccaneers":"tb",
"Tennessee Titans":"ten",
"Washington Commanders":"was",
"Washington Football Team":"was",
"Washington Redskins":"was"}

df=cover_df
df.columns = ['trash','week','away_team','home_team','cur_ou','home_team2','cur_spread']
df.drop(['home_team2'], axis=1, inplace=True)
df.columns = [x.lower() for x in df.columns]
df.reset_index(drop=True, inplace=True)
df['home_team']=df['home_team'].str.lower()
df['away_team']=df['away_team'].str.lower()
df['year'] = 2022

df['away_team']=df['away_team'].str.replace('jac','jax')
df['home_team']=df['home_team'].str.replace('jac','jax')


df['cur_spread']=df['cur_spread'].apply(float)
def fav_spread(nData):
    if nData['cur_spread'] > 0:
        return nData['away_team']
    else:
        return nData['home_team']
df['cur_fav_team'] = df.apply(lambda nData: fav_spread(nData), axis=1)

df['year']=df['year'].apply(str)
df.insert(0, "covers_team_id", (df['home_team']+'vs'+df['away_team']+'_'+df['year']))



sp = pd.read_csv("/home/tomb/nfl_models/current_data/week_"+cur_week+"/spreadsw"+cur_week+".csv")
sp = sp[(sp['schedule_season']==2022) & (sp['schedule_week']==int(cur_week))]
        
sp['schedule_season']=sp['schedule_season'].apply(str)
sp.insert(0, "covers_team_id", (sp['team_home_abb']+'vs'+sp['away_team_abb']+'_'+sp['schedule_season']))

sp=pd.merge(sp, df, on='covers_team_id', how='left')

sp['fav_team_open_abb'] = sp['fav_team_open'].map(clean_team_spreads).fillna(sp['fav_team_open'])



  

def cur_spread(nData):
    if nData['cur_spread'] > 0:
        return (nData['cur_spread'])*-1
    else:
        return nData['cur_spread']
sp['spread_favorite'] = sp.apply(lambda nData: cur_spread(nData), axis=1)
sp['team_favorite_id'] = sp['cur_fav_team']
sp['over_under_line'] = sp['cur_ou']



def flip(nData):
    if nData['fav_team_open_abb'] == nData['cur_fav_team']:
        return 1
    else:
        return 0
sp['remain_fav'] =sp.apply(lambda nData: flip(nData), axis=1)

sp['over_under_line']=sp['over_under_line'].apply(float)
sp['ou_movement'] = (sp['over_under_line']- sp['Total Score Open'])


def fav_spread_movement(nData):
    if nData['remain_fav'] == 1:
        return (nData['spread_favorite'] - nData['starting_spread'])
    else:
        return (nData['spread_favorite'] + nData['starting_spread'])

sp['spread_movement'] = sp.apply(lambda nData: fav_spread_movement(nData), axis=1)

def fav_stronger(nData):
    if (nData['spread_movement'] <= -3) | (nData['spread_movement'] >= 3):
        return 1
    else:
        return 0

sp['strong_movement'] = sp.apply(lambda nData: fav_stronger(nData), axis=1)




def stronger(nData):
    if (nData['fav_team_open_abb'] == nData['cur_fav_team']) & (nData['spread_movement'] < 0):
        return 1
    elif (nData['fav_team_open_abb'] == nData['cur_fav_team']) & (nData['spread_movement'] == 0):
        return 0
    else:
        return -1

sp['fav_team_stronger'] = sp.apply(lambda nData: stronger(nData), axis=1)
sp.drop(['covers_team_id','trash','week','away_team','home_team','cur_ou','cur_spread','cur_fav_team','fav_team_open_abb','year'], axis=1, inplace=True)
sp['schedule_season']=sp['schedule_season'].apply(int)

sp_comb = pd.read_csv("/home/tomb/nfl_models/current_data/week_"+cur_week+"/spreadsw"+cur_week+".csv")
sub = sp_comb[(sp_comb['schedule_season']==2022) & (sp_comb['schedule_week']==int(cur_week))]
sp_comb = sp_comb[~sp_comb.index.isin(sub.index)]
        
sp_comb = pd.concat([sp_comb, sp], axis=0)                  
sp_comb.to_csv("/home/tomb/nfl_models/current_data/week_"+cur_week+"/spreadsw"+cur_week+".csv", index=False)       
        