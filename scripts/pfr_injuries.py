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
#driver = webdriver.Chrome(executable_path='/home/tom/Downloads/chromedriver_linux64/chromedriver')
binary = "/home/tomb/Downloads/chromedriver/chromedriver"
driver = webdriver.Chrome(binary)

delay = 2

delay = 1

## Create variable that looks up the chrome driver library ##

### CHANGE ALL DATES, INCLUDING TEXT FILES3BEFORE BEGINNING TO SCRAPE ###

teams = ['crd','atl','rav','buf','car','chi','cin','cle','dal','den','det','gnb','htx','clt','jax','kan','sdg','ram','mia','min','nwe','nor','nyg','nyj','rai','phi','pit','sfo','sea','tam','oti','was']#'crd','atl','rav','buf','car','chi',
years = ['2022']#'2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019']
rost = []
full_list=[]
for tm in teams:
    ur = "https://www.pro-football-reference.com/teams/"+tm
    for yr in years:
        url = ur+"/"+yr+"_injuries.htm"
        driver.get(url)
        time.sleep(10)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        try:
            myElem = WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.CLASS_NAME, "sortable stats_table now_sortable")))
            print("Page is ready!")
            time.sleep(5)
        except TimeoutException:
            print("Loading took too much time!")
            driver.execute_script("scrollBy(0,500);")
            sleep(3)
            #driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)
        yr = yr
        team = tm
        
        driver.execute_script("scrollBy(0,1000);")
        sleep(3)
        collect_page = driver.page_source
        time.sleep(5)
        soup = BeautifulSoup(collect_page, "lxml", from_encoding="utf-8")
        div = soup.find_all('table', {'id':'team_injuries'})
        lst,lst1,lst2,lst3,lst4,lst5,lst6,lst7,lst8,lst9,lst10,lst11,lst12,lst13,lst14,lst15,lst16,lst17,lst18=[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
        for i in div:
            tr = i.find_all('tr')
            for d in tr:
                ply = d.find('a')#('th', {'data-stat':"player"})
                td = d.find_all('td')
                if 'vs.' in str(ply):
                    lst.append('None')
                else:
                    lst.append(ply.text)
            
                
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
            team = pd.DataFrame(list(zip(lst, lst1, lst2, lst3,lst4,lst5,lst6,lst7)),#,lst8,lst9,lst10,lst11,lst12,lst13,lst14,lst15,lst16,lst17,lst18)),
                             columns=['player','w1', 'w2','w3','w4','w5','w6','w7'])#,'w8','w9','w10','w11','w12','w13','w14','w15','w16','w17','w18'])
            team['team']=tm
            team['year']=yr
            full_list.append(team)
            conc = pd.concat(full_list, axis=0)
            conc.to_csv('~/nfl_models/2022_w7.csv', index=False)
