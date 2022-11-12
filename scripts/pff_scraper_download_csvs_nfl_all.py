# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 19:07:05 2020

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
import os.path
import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import *
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
chrome_options.add_experimental_option("prefs", {"download.default_directory": r"./"})
chrome_options.add_argument('--no-sandbox')
driver = webdriver.Chrome('/home/tomb/nfl_models/chromedriver',chrome_options=chrome_options)


#os.path.abspath(os.getcwd())

cur_week = str(10)

os.chdir('/home/tomb/nfl_models/scripts/')
delay = 2


## Create variable that looks up the chrome driver library ##

### CHANGE ALL DATES, INCLUDING TEXT FILES3BEFORE BEGINNING TO SCRAPE ###
year_list = ['2022']#,'2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021']#,'2009','2010','2011','2012',
week_list = ['1','2','3','4','5','6','7','8','9']#,,'5','6','5','6','7','8','9','10','11','12','13','14','15','16','17','18']



url_list = [
'https://premium.pff.com/nfl/positions/2022/CUSTOM/passing?position=QB',
'https://premium.pff.com/nfl/positions/2022/CUSTOM/passing-pressure?position=QB',
'https://premium.pff.com/nfl/positions/2022/CUSTOM/passing-depth?position=QB',
'https://premium.pff.com/nfl/positions/2022/CUSTOM/passing-concept?position=QB',
'https://premium.pff.com/nfl/positions/2022/CUSTOM/time-in-pocket?position=QB',
'https://premium.pff.com/nfl/positions/2022/CUSTOM/passing-allowed-pressure?position=QB',
'https://premium.pff.com/nfl/positions/2022/CUSTOM/receiving?position=WR,TE,RB',
'https://premium.pff.com/nfl/positions/2022/CUSTOM/receiving-depth?position=WR,TE,RB',
'https://premium.pff.com/nfl/positions/2022/CUSTOM/receiving-concept?position=WR,TE,RB&split=slot',
'https://premium.pff.com/nfl/positions/2022/CUSTOM/receiving-scheme?position=WR,TE,RB',
'https://premium.pff.com/nfl/positions/2022/CUSTOM/rushing?position=HB,FB',
'https://premium.pff.com/nfl/positions/2022/CUSTOM/offense-blocking?position=T,G,C,TE,RB',
'https://premium.pff.com/nfl/positions/2022/CUSTOM/offense-pass-blocking?position=T,G,C,TE,RB',
'https://premium.pff.com/nfl/positions/2022/CUSTOM/offense-run-blocking?position=T,G,C,TE,RB',
'https://premium.pff.com/nfl/positions/2022/CUSTOM/ol-pass-blocking-efficiency?position=T,G,C,TE,RB',
'https://premium.pff.com/nfl/positions/2022/CUSTOM/defense?position=DI,ED,LB,CB,S',
'https://premium.pff.com/nfl/positions/2022/CUSTOM/defense-pass-rush?position=DI,ED,LB,CB,S',
'https://premium.pff.com/nfl/positions/2022/CUSTOM/defense-run?position=DI,ED,LB,CB,S',
'https://premium.pff.com/nfl/positions/2022/CUSTOM/defense-coverage?position=DI,ED,LB,CB,S',
'https://premium.pff.com/nfl/positions/2022/CUSTOM/defense-coverage-scheme?position=DI,ED,LB,CB,S',
'https://premium.pff.com/nfl/positions/2022/CUSTOM/defense-coverage-slot?position=DI,ED,LB,CB,S',
'https://premium.pff.com/nfl/positions/2022/CUSTOM/defense-pass-rush-productivity?position=DI,ED,LB,CB,S',
'https://premium.pff.com/nfl/positions/2022/CUSTOM/kicking?',
'https://premium.pff.com/nfl/positions/2022/CUSTOM/punting?']

list_of_completed_urls, list_of_failed = [],[]

passing_list, pass_adj, pass_deep, pass_under_press, pass_clean_pocket, pass_pa, pass_time = [],[],[],[],[],[],[]
rec_summ, rec_drop, rec_slot, rec_yprr, rec_deep = [],[],[],[],[]
rush_summ, rush_elu, rush_brk = [],[],[]
allow_press, pass_block_eff, run_block, block_eff = [], [], [], []
def_summ, def_runstop, def_tack, def_cov, def_slot, def_pr =[],[],[],[],[],[]
st_summ, st_kick, st_punt, st_ret = [],[],[],[]

for urls in url_list:
    for yr in year_list:
        for wk in week_list:
            #if yr == '2020' and wk == '7':
                #break
            url = urls+'&week={}'.format(wk)
            print(url)
            driver.get(url)
            sleep(randint(1,3))
            count=0
            try:
                # count < 5:
                myElem = WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.CLASS_NAME, "kyber-table-body__sticky-rows-container")))
                driver.execute_script("scrollBy(0,500);")
                sleep(randint(1,3))
                list_of_completed_urls.append(url)
                    #count += 1
                    #driver.refresh()
                #else:
                    #pass
            except TimeoutException:
                list_of_failed.append(url)
    
            
            driver.find_element("xpath", "//*[text()='CSV']").click()
            print('Retrieved csv from: {}'.format(url))
            
            if 'passing?' in str(url):
                time.sleep(5)
                while not os.path.exists('./passing_summary.csv'):
                    time.sleep(1)
                if os.path.isfile('./passing_summary.csv'):
                    df = pd.read_csv('./passing_summary.csv')
                    df['week'], df['year'] = wk, yr
                    os.remove('./passing_summary.csv')
                    passing_list.append(df)
                else:
                    raise ValueError("%s isn't a file!" % './passing_summary.csv') 
                
            if 'passing-allowed-pressure?' in str(url):
                while not os.path.exists('./passing_allowed_pressure.csv'):
                    time.sleep(1)
                if os.path.isfile('./passing_allowed_pressure.csv'):
                    df = pd.read_csv('./passing_allowed_pressure.csv')
                    df['week'], df['year'] = wk, yr
                    os.remove('./passing_allowed_pressure.csv')
                    pass_adj.append(df)
                else:
                    raise ValueError("%s isn't a file!" % './passing_allowed_pressure.csv')
            
            if 'passing-depth?' in str(url):
                while not os.path.exists('./passing_depth.csv'):
                    time.sleep(1)
                if os.path.isfile('./passing_depth.csv'):
                    df = pd.read_csv('./passing_depth.csv')
                    df['week'], df['year'] = wk, yr
                    os.remove('./passing_depth.csv')
                    pass_deep.append(df)
                else:
                    raise ValueError("%s isn't a file!" % './passing_depth.csv')
            

            if 'passing-pressure?' in str(url):
                while not os.path.exists('./passing_pressure.csv'):
                    time.sleep(1)
                if os.path.isfile('./passing_pressure.csv'):
                    df = pd.read_csv('./passing_pressure.csv')
                    df['week'], df['year'] = wk, yr
                    os.remove('./passing_pressure.csv')
                    pass_under_press.append(df)
                else:
                    raise ValueError("%s isn't a file!" % './passing_pressure.csv')
            
                
            if 'passing-concept?' in str(url):
                while not os.path.exists('./passing_concept.csv'):
                    time.sleep(1)
                if os.path.isfile('./passing_concept.csv'):
                    df = pd.read_csv('./passing_concept.csv')
                    df['week'], df['year'] = wk, yr
                    os.remove('./passing_concept.csv')
                    pass_pa.append(df)
                else:
                    raise ValueError("%s isn't a file!" % './passing_concept.csv')
                
            if 'time-in-pocket?' in str(url):
                while not os.path.exists('./time_in_pocket.csv'):
                    time.sleep(1)
                if os.path.isfile('./time_in_pocket.csv'):
                    df = pd.read_csv('./time_in_pocket.csv')
                    df['week'], df['year'] = wk, yr
                    os.remove('./time_in_pocket.csv')
                    pass_time.append(df)
                else:
                    raise ValueError("%s isn't a file!" % './time_in_pocket.csv')

            if 'receiving?' in str(url):
                while not os.path.exists('./receiving_summary.csv'):
                    time.sleep(1)
                if os.path.isfile('./receiving_summary.csv'):
                    df = pd.read_csv('./receiving_summary.csv')
                    df['week'], df['year'] = wk, yr
                    os.remove('./receiving_summary.csv')
                    rec_summ.append(df)
                else:
                    raise ValueError("%s isn't a file!" % './receiving_summary.csv')

            if 'receiving-depth?' in str(url):
                while not os.path.exists('./receiving_depth.csv'):
                    time.sleep(1)
                if os.path.isfile('./receiving_depth.csv'):
                    df = pd.read_csv('./receiving_depth.csv')
                    df['week'], df['year'] = wk, yr
                    os.remove('./receiving_depth.csv')
                    rec_drop.append(df)
                else:
                    raise ValueError("%s isn't a file!" % './receiving_depth.csv') 

            if 'receiving-concept' in str(url):
                while not os.path.exists('./receiving_concept.csv'):
                    time.sleep(1)
                if os.path.isfile('./receiving_concept.csv'):
                    df = pd.read_csv('./receiving_concept.csv')
                    df['week'], df['year'] = wk, yr
                    os.remove('./receiving_concept.csv')
                    rec_slot.append(df)
                else:
                    raise ValueError("%s isn't a file!" % './receiving_concept.csv') 

            if 'receiving-scheme?' in str(url):
                while not os.path.exists('./receiving_scheme.csv'):
                    time.sleep(1)
                if os.path.isfile('./receiving_scheme.csv'):
                    df = pd.read_csv('./receiving_scheme.csv')
                    df['week'], df['year'] = wk, yr
                    os.remove('./receiving_scheme.csv')
                    rec_yprr.append(df)
                else:
                    raise ValueError("%s isn't a file!" % './receiving_scheme.csv') 
                    
                  
            if 'rushing?' in str(url):
                while not os.path.exists('./rushing_summary.csv'):
                    time.sleep(1)
                if os.path.isfile('./rushing_summary.csv'):
                    df = pd.read_csv('./rushing_summary.csv')
                    df['week'], df['year'] = wk, yr
                    os.remove('./rushing_summary.csv')
                    rush_summ.append(df)
                else:
                    raise ValueError("%s isn't a file!" % './rushing_summary.csv')                        
  

            if 'offense-blocking?' in str(url):
                while not os.path.exists('./offense_blocking.csv'):
                    time.sleep(1)
                if os.path.isfile('./offense_blocking.csv'):
                    df = pd.read_csv('./offense_blocking.csv')
                    df['week'], df['year'] = wk, yr
                    os.remove('./offense_blocking.csv')
                    allow_press.append(df)
                else:
                    raise ValueError("%s isn't a file!" % './offense_blocking.csv')       
				

            if 'offense-pass-blocking?' in str(url):
                while not os.path.exists('./offense_pass_blocking.csv'):
                    time.sleep(1)
                if os.path.isfile('./offense_pass_blocking.csv'):
                    df = pd.read_csv('./offense_pass_blocking.csv')
                    df['week'], df['year'] = wk, yr
                    os.remove('./offense_pass_blocking.csv')
                    pass_block_eff.append(df)
                else:
                    raise ValueError("%s isn't a file!" % './offense_pass_blocking.csv')  

            if 'offense-run-blocking?' in str(url):
                while not os.path.exists('./offense_run_blockng.csv'):
                    time.sleep(1)
                if os.path.isfile('./offense_run_blockng.csv'):
                    df = pd.read_csv('./offense_run_blockng.csv')
                    df['week'], df['year'] = wk, yr
                    os.remove('./offense_run_blockng.csv')
                    run_block.append(df)
                else:
                    raise ValueError("%s isn't a file!" % './offense_run_blockng.csv') 
					
            if 'ol-pass-blocking-efficiency?' in str(url):
                while not os.path.exists('./line_pass_blocking_efficiency.csv'):
                    time.sleep(1)
                if os.path.isfile('./line_pass_blocking_efficiency.csv'):
                    df = pd.read_csv('./line_pass_blocking_efficiency.csv')
                    df['week'], df['year'] = wk, yr
                    os.remove('./line_pass_blocking_efficiency.csv')
                    block_eff.append(df)
                else:
                    raise ValueError("%s isn't a file!" % './line_pass_blocking_efficiency.csv') 

            if 'defense?' in str(url):
                while not os.path.exists('./defense_summary.csv'):
                    time.sleep(1)
                if os.path.isfile('./defense_summary.csv'):
                    df = pd.read_csv('./defense_summary.csv')
                    df['week'], df['year'] = wk, yr
                    os.remove('./defense_summary.csv')
                    def_summ.append(df)
                else:
                    raise ValueError("%s isn't a file!" % './defense_summary.csv')    

            if 'defense-pass-rush?' in str(url):
                while not os.path.exists('./pass_rush_summary.csv'):
                    time.sleep(1)
                if os.path.isfile('./pass_rush_summary.csv'):
                    df = pd.read_csv('./pass_rush_summary.csv')
                    df['week'], df['year'] = wk, yr
                    os.remove('./pass_rush_summary.csv')
                    def_pr.append(df)
                else:
                    raise ValueError("%s isn't a file!" % './pass_rush_summary.csv')   
					
                    
            if 'defense-run?' in str(url):
                while not os.path.exists('./run_defense_summary.csv'):
                    time.sleep(1)
                if os.path.isfile('./run_defense_summary.csv'):
                    df = pd.read_csv('./run_defense_summary.csv')
                    df['week'], df['year'] = wk, yr
                    os.remove('./run_defense_summary.csv')
                    def_runstop.append(df)
                else:
                    raise ValueError("%s isn't a file!" % './run_defense_summary.csv')


            if 'defense-coverage-scheme?' in str(url):
                while not os.path.exists('./defense_coverage_scheme.csv'):
                    time.sleep(1)
                if os.path.isfile('./defense_coverage_scheme.csv'):
                    df = pd.read_csv('./defense_coverage_scheme.csv')
                    df['week'], df['year'] = wk, yr
                    os.remove('./defense_coverage_scheme.csv')
                    def_tack.append(df)
                else:
                    raise ValueError("%s isn't a file!" % './defense_coverage_scheme.csv') 

                    
            if 'defense-coverage?' in str(url):
                while not os.path.exists('./defense_coverage_summary.csv'):
                    time.sleep(1)
                if os.path.isfile('./defense_coverage_summary.csv'):
                    df = pd.read_csv('./defense_coverage_summary.csv')
                    df['week'], df['year'] = wk, yr
                    os.remove('./defense_coverage_summary.csv')
                    def_cov.append(df)
                else:
                    raise ValueError("%s isn't a file!" % './defense_coverage_summary.csv') 
                    
            if 'defense-coverage-slot?' in str(url):
                while not os.path.exists('./slot_coverage.csv'):
                    time.sleep(1)
                if os.path.isfile('./slot_coverage.csv'):
                    df = pd.read_csv('./slot_coverage.csv')
                    df['week'], df['year'] = wk, yr
                    os.remove('./slot_coverage.csv')
                    def_slot.append(df)
                else:
                    raise ValueError("%s isn't a file!" % './slot_coverage.csv') 
                                        
                    
            if 'kicking?' in str(url):
                while not os.path.exists('./field_goal_summary.csv'):
                    time.sleep(1)
                if os.path.isfile('./field_goal_summary.csv'):
                    df = pd.read_csv('./field_goal_summary.csv')
                    df['week'], df['year'] = wk, yr
                    os.remove('./field_goal_summary.csv')
                    st_kick.append(df)
                else:
                    raise ValueError("%s isn't a file!" % './field_goal_summary.csv')                    

                    
            if 'punting?' in str(url):
                while not os.path.exists('./punting_summary.csv'):
                    time.sleep(1)
                if os.path.isfile('./punting_summary.csv'):
                    df = pd.read_csv('./punting_summary.csv')
                    df['week'], df['year'] = wk, yr
                    os.remove('./punting_summary.csv')
                    st_punt.append(df)
                else:
                    raise ValueError("%s isn't a file!" % './punting_summary.csv')


passing_depth = pd.concat(pass_deep, axis=0)  
passing_allowed_pressure = pd.concat(pass_adj, axis=0)
passing_pressure = pd.concat(pass_under_press, axis=0)
passing_concept = pd.concat(pass_pa, axis=0)
time_in_pocket = pd.concat(pass_time, axis=0)        
passing_summ_conc = pd.concat(passing_list, axis=0)  

   
rec_summ_conc = pd.concat(rec_summ, axis=0)
receiving_concept = pd.concat(rec_slot, axis=0)
receiving_depth = pd.concat(rec_drop, axis=0)
receiving_scheme = pd.concat(rec_yprr, axis=0)
#deep_receiving = pd.concat(rec_deep, axis=0)

rush_summ_conc = pd.concat(rush_summ, axis=0) 

block_summ_conc = pd.concat(allow_press, axis=0)
offense_pass_blocking = pd.concat(pass_block_eff, axis=0)
offense_run_blocking = pd.concat(run_block, axis=0)

def_summ_conc = pd.concat(def_summ, axis=0)
pass_rush_summary = pd.concat(def_pr, axis=0)
run_defense_summary = pd.concat(def_runstop, axis=0)
defense_coverage_scheme = pd.concat(def_tack, axis=0)
defense_coverage_summary = pd.concat(def_cov, axis=0)
slot_coverage = pd.concat(def_slot, axis=0)

st_kickers = pd.concat(st_kick, axis=0)
st_punters = pd.concat(st_punt, axis=0)


passing_depth.to_csv('./nfl_all/passing_depth_'+yr+'.csv', index=False)
passing_allowed_pressure.to_csv('./nfl_all/passing_allowed_pressure_'+yr+'.csv', index=False)
passing_pressure.to_csv('./nfl_all/passing_pressure_'+yr+'.csv', index=False)
passing_concept.to_csv('./nfl_all/passing_concept_'+yr+'.csv', index=False)
time_in_pocket.to_csv('./nfl_all/time_in_pocket_'+yr+'.csv', index=False)
passing_summ_conc.to_csv('./nfl_all/passing_summ_conc_'+yr+'.csv', index=False)

rec_summ_conc.to_csv('./nfl_all/rec_summ_conc_'+yr+'.csv', index=False)
receiving_concept.to_csv('./nfl_all/receiving_concept_'+yr+'.csv', index=False)
receiving_depth.to_csv('./nfl_all/receiving_depth_'+yr+'.csv', index=False)
receiving_scheme.to_csv('./nfl_all/receiving_scheme_'+yr+'.csv', index=False)

rush_summ_conc.to_csv('./nfl_all/rush_summ_conc_'+yr+'.csv', index=False)

block_summ_conc.to_csv('./nfl_all/block_summ_conc_'+yr+'.csv', index=False)
offense_pass_blocking.to_csv('./nfl_all/offense_pass_blocking_'+yr+'.csv', index=False)
offense_run_blocking.to_csv('./nfl_all/offense_run_blocking_'+yr+'.csv', index=False)

def_summ_conc.to_csv('./nfl_all/def_summ_conc_'+yr+'.csv', index=False)
pass_rush_summary.to_csv('./nfl_all/pass_rush_summary_'+yr+'.csv', index=False)
run_defense_summary.to_csv('./nfl_all/run_defense_summary_'+yr+'.csv', index=False)
defense_coverage_scheme.to_csv('./nfl_all/defense_coverage_scheme_'+yr+'.csv', index=False)
defense_coverage_summary.to_csv('./nfl_all/defense_coverage_summary_'+yr+'.csv', index=False)
slot_coverage.to_csv('./nfl_all/slot_coverage_'+yr+'.csv', index=False)

st_kickers.to_csv('./nfl_all/st_kickers_'+yr+'.csv', index=False)
st_punters.to_csv('./nfl_all/st_punters_'+yr+'.csv', index=False)





#
#

import re
import time

year = '2022'
yr = '2022'

team_list = ['arizona-cardinals',
'atlanta-falcons',
'baltimore-ravens',
'buffalo-bills',
'carolina-panthers',
'chicago-bears',
'cincinnati-bengals',
'cleveland-browns',
'dallas-cowboys',
'denver-broncos',
'detroit-lions',
'green-bay-packers',
'houston-texans',
'indianapolis-colts',
'jacksonville-jaguars',
'kansas-city-chiefs',
'los-angeles-chargers',
'los-angeles-rams',
'miami-dolphins',
'minnesota-vikings',
'new-england-patriots',
'new-orleans-saints',
'new-york-giants',
'new-york-jets',
'las-vegas-raiders',
'philadelphia-eagles',
'pittsburgh-steelers',
'san-francisco-49ers',
'seattle-seahawks',
'tampa-bay-buccaneers',
'tennessee-titans',
'washington-commanders']

list_of_dfs = []
summ_list,block_list,pass_list,passpress_list,rush_list,rushdir_list, rec_list, def_list = [],[],[],[],[],[],[],[]
urls, completed_urls = [],[]

yr='2022'
year='2022'
ids_list, game_list, plyr_data = [], [], []
for team_num, tm in enumerate(team_list, start=1):
    team_season_url = 'https://premium.pff.com/nfl/teams/'+tm+'/summary?season='+yr+'&weekGroup=REGPO'
    print(team_season_url)
    sleep(randint(1,3))
    driver.get(team_season_url)
    sleep(randint(1,3))
    roster = driver.page_source        
    soup = BeautifulSoup(roster, "lxml", from_encoding="utf-8")

    ## target containers on current page and scrape team rolled up data ##
    plyr = soup.find_all('div', {'class':'kyber-table-body__sticky-rows-container'})
    dat = soup.find_all('div', {'class':'kyber-table-body__scrolling-rows-container'})

    for i in plyr: 
        row = i.find_all('div', {'class':'kyber-table-body__row'})
        for x in row:
            divs = x.find_all('div')
            wk = divs[0].text
            try:
                hora = divs[1].text
            except IndexError:
                hora = ''
            idxs = "{};{};{};{};{}".format(tm, yr, wk, hora, team_num)
            ids_list.append(idxs)

    ## grab all the game ids for a given team to loop through ##

    for scrolling in dat:
        div = scrolling.find_all('div', {'class':'kyber-table-body__row'})
        for txt in div:
            game = txt.find('button', {'class':'g-btn kyber-button g-btn--secondary g-btn--sm'})
            try:
                game_id = game.get('data-game-id')
            except AttributeError:
                game_id = 'None'
            gameids = "{}".format(game_id)
            game_list.append(game_id)
                
    ## grab rolled up data from team summary page ##            
    for i in dat:
        div = i.find_all('div', {'class':'kyber-table-body__row'})
        for txt in div:
            #div = txt.find_all('div', {'class':'kyber-table-body-cell kyber-table-body-cell--align-center'})
            #div = txt.soup.select('span[class*="currentPrice-"]')
            ## should find all div cells regardless of alignment type ##
            div = txt.find_all('div', attrs={'class': re.compile('^kyber-table-body-cell kyber-table-body-cell*')})
            #driver.find_element_by_xpath('//label[contains(@class, "kyber-filter-strip__option") and contains(., "200")]').click()

            div0 = div[0].text
            div1 = div[1].text
            div2 = div[2].text
            div3 = div[3].text
            div4 = div[4].text
            div5 = div[5].text
            div6 = div[6].text
            div7 = div[7].text
            div8 = div[8].text
            div9 = div[9].text
            div10 = div[10].text
            div11 = div[11].text
            div12 = div[12].text
            div13 = div[13].text
            div14 = div[14].text
            div15 = div[15].text
            div16 = div[16].text
            div17 = div[17].text
            div18 = div[18].text    
            team_form = "{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{}".format(
            div0,div1,div2,div3,div4,div5,div6,div7,div8,div9,div10,div11,div12,div13,div14,div15,div16,div17,div18)
            plyr_data.append(team_form)
    

ids_df = pd.DataFrame([i.split(';') for i in ids_list])
ids_df.columns=['team','year','week','home_or_away','team_num']

gameid_df = pd.DataFrame(game_list, columns=['game_id'])

team_summ = pd.DataFrame([i.split(';') for i in plyr_data])
team_summ.columns=['opponent','date','time','wl','pf','pa','overall_performance','offense','pass','pass_blocking','receiving','rushing','run_blocking','defense','rush_defense','tackling','pass_rush','coverage','special_teams']

together = pd.concat([ids_df, gameid_df, team_summ], axis=1)
list_of_dfs.append(together)

year_team_summ = pd.concat(list_of_dfs, axis=0)
year_team_summ = year_team_summ.dropna(thresh=10)  
year_team_summ = year_team_summ[year_team_summ['game_id'] !='None']


#year_team_summ= pd.read_csv('/home/tom/spreads_2022/historic_data/team_game_summaries_historic.csv')
                      
year_team_summ['dateInt']=year_team_summ['year'].astype(str) + '/'+year_team_summ['date'].astype(str)#+' '+year_team_summ['time'].astype(str)
year_team_summ['Date'] = pd.to_datetime(year_team_summ['dateInt'])#, format='%Y%m%d')
year_team_summ = year_team_summ.assign(output=year_team_summ.groupby(['team','year']).Date.apply(lambda x: x.diff()))
year_team_summ['output'] = year_team_summ['output'].astype(str)
year_team_summ['output'] = year_team_summ['output'].str.replace(' days','')


year_team_summ.to_csv('/home/tomb/nfl_models/current_data/week_'+cur_week+'/team_game_summaries_w'+cur_week+'.csv', index=False)

