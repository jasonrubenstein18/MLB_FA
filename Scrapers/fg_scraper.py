from bs4 import BeautifulSoup as bs
import requests
from selenium import webdriver
import pandas as pd
import numpy as np
import time


# Scrape Fangraphs for MLB Data (incl. but not limited to WAR, OPS, HR, etc.)
years = [2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]

fg_bat_data = pd.DataFrame()

urls = []
# batters = "https://www.fangraphs.com/leaders.aspx?pos=all&stats=bat&lg=all&qual=20&type=c,6,34,35,36,-1,23,37,38,39,-1,40,60,41,-1,201,205,200,-1,52,51,50,61,58&season=2020&month=0&season1=2020&ind=0&team=0&rost=0&age=0&filter=&players=0&startdate=2020-01-01&enddate=2020-12-31&page=1_1000"
# pitchers = "https://www.fangraphs.com/leaders.aspx?pos=all&stats=pit&lg=all&qual=20&" \
#            "type=c,36,37,38,40,-1,120,121," \
#            "217,-1,41,42,43,44,-1,117,118,119,-1,6,45,124,-1,62,122,59,58,76,82,81,7,13,14,91,92,93,94,95,113&" \
#            "season=" + str(y) + "&month=0&season1=" + str(y) + "&ind=0&team=0&" \
#            "rost=0&age=0&filter=&players=0&startdate=" + str(y) + "-01-01&enddate=" + str(y) + "-12-31&page=1_1000"

# pitchers = "https://www.fangraphs.com/leaders.aspx?pos=all&stats=pit&lg=all&qual=20&type=c,36,37,38,40,-1,120,121,217,-1,41,42,43,44,-1,117,118,119,-1,6,45,124,-1,62,122,59,58,76,82,81,7,13,14,91,92,93,94,95,113&season=2020&month=0&season1=2020&ind=0&team=0&rost=0&age=0&filter=&players=0&startdate=2020-01-01&enddate=2020-12-31&page=1_1000"
# https://www.fangraphs.com/leaders.aspx?pos=all&stats=bat&lg=all&qual=20&type=c,6,34,35,36,-1,23,37,38,39,-1,40,60,41,-1,201,205,200,-1,52,51,50,61,58,199,203&season=2020&month=0&season1=&ind=0&team=0&rost=0&age=0&filter=&players=0&startdate=2020-01-01&enddate=2020-12-31

for y in years:
    url = "https://www.fangraphs.com/leaders.aspx?pos=all&stats=bat&lg=all&qual=20&type=c,6,34,35,36,-1, " \
          "23,37,38,39,-1,40,60,41,-1,201,205,200,-1,52,51,50,61,58,199,203&season=" + str(y) + "&month=0&" \
          "season1=" + str(y) + "&ind=0&team=0&rost=0&age=0&filter=&players=0&startdate=" + str(y) + \
          "-01-01&enddate=" + str(y) + "-12-31&page=1_1000"
    urls.append(url)

for url in urls:
    series_url = pd.Series(url)
    df_url = series_url.str.split('=', expand=True)
    df_url.columns = ['protocol', 'blank', 'path_one', 'path_two', 'path_three', 'stats', 'month', 'season',
                      'year', 'team', 'rost', 'age', 'filter', 'players', 'start', 'end', 'page', 'a']
    df_url["year"] = df_url["year"].apply(lambda x: x.replace("&ind", ""))
    # then make that the year reference cell
    driver = webdriver.Chrome('/Users/jasonrubenstein/.wdm/chromedriver/85.0.4183.87/mac64/chromedriver')
    driver.implicitly_wait(5)
    driver.get(url)
    # response = requests.get(url).text
    response = requests.get(url).text
    soup = bs(response, "html.parser")
    rows = soup.find_all('td')
    df = pd.Series([item.text.strip() for item in rows])
    df = df[df.index > 89].reset_index(drop=True)
    df = df.iloc[:-5].reset_index(drop=True)
    df_fix = pd.DataFrame(np.reshape(df.values, (df.shape[0] // 24, 24)),
                          columns=['Num', 'Name', "Team", "PA", "BB%", "K%", "BB/K", "AVG", "OBP", "SLG", "OPS", "ISO",
                                   "Spd", "BABIP", "UBR", "wGDP", "wSB", "wRC", "wRAA", "wOBA", "wRC+",
                                   "WAR", "dWAR", "oWAR"])
    df_fix['Year'] = df_url['year']
    fg_bat_data = fg_bat_data.append(df_fix, ignore_index=True)
    time.sleep(5)
    driver.close()

fg_bat_data_full = fg_bat_data
fg_bat_data_full['Year'] = fg_bat_data_full['Year'].ffill(axis=0)
fg_bat_data_full['Year'] = fg_bat_data_full['Year'].astype('int') + 1
del fg_bat_data_full['Team'], fg_bat_data_full['Num']

fg_bat_data_full.to_csv("~/Desktop/MLB_FA/fg_bat_data.csv", index=False)


# Pitching Data
fg_pitch_data = pd.DataFrame()

urls = []

for y in years:
    url = "https://www.fangraphs.com/leaders.aspx?pos=all&stats=pit&lg=all&qual=20&" \
          "type=c,36,37,38,40,-1,120,121," \
           "217,-1,41,42,43,44,-1,117,118,119,-1,6,45,124,-1,62,122,59,58,76,82,81,7,13,14,91,92,93,94,95,113&" \
          "season=" + str(y) + "&month=0&season1=" + str(y) + "&ind=0&team=0&" \
          "rost=0&age=0&filter=&players=0&startdate=" + str(y) + "-01-01&enddate=" + str(y) + "-12-31&page=1_1000"
    urls.append(url)


for url in urls:
    series_url = pd.Series(url)
    df_url = series_url.str.split('=', expand=True)
    df_url.columns = ['protocol', 'blank', 'path_one', 'path_two', 'path_three', 'stats', 'month', 'season',
                      'year', 'team', 'rost', 'age', 'filter', 'players', 'start', 'end', 'page', 'a']
    df_url["year"] = df_url["year"].apply(lambda x: x.replace("&ind", ""))
    # then make that the year reference cell
    driver = webdriver.Chrome('/Users/jasonrubenstein/.wdm/chromedriver/85.0.4183.87/mac64/chromedriver')
    driver.implicitly_wait(5)
    driver.get(url)
    # response = requests.get(url).text
    response = requests.get(url).text
    soup = bs(response, "html.parser")
    rows = soup.find_all('td')
    df = pd.Series([item.text.strip() for item in rows])
    df = df[df.index > 89].reset_index(drop=True)
    df = df.iloc[:-5].reset_index(drop=True)
    df_fix = pd.DataFrame(np.reshape(df.values, (df.shape[0] // 36, 36)),
                          columns=['Num', 'Name', "Team", "K/9",  "BB/9", "K/BB", "HR/9", "K%", "BB%", "K-BB%", "AVG",
                                   "WHIP", "BABIP", "LOB%", "ERA-", "FIP-", "xFIP-", "ERA", "FIP", "E-F", "xFIP",
                                   "SIERA", "WAR", "RAR", "FBv", "CBv", "CB%", "G", "IP", "TBF", "wFB", "wSL",
                                   "wCT", "wCB", "wCH", "SwStr%"
                                   ])
    df_fix['Year'] = df_url['year']
    fg_pitch_data = fg_pitch_data.append(df_fix, ignore_index=True)
    time.sleep(5)
    driver.close()

fg_pitch_data_full = fg_pitch_data
fg_pitch_data_full['Year'] = fg_pitch_data_full['Year'].ffill(axis=0)
fg_pitch_data_full['Year'] = fg_pitch_data_full['Year'].astype('int') + 1
# del fg_pitch_data_full['Team'], fg_pitch_data_full['Num']

fg_pitch_data_full.to_csv("~/Desktop/MLB_FA/fg_pitch_data.csv", index=False)
