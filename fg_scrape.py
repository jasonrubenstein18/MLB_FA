from bs4 import BeautifulSoup as bs
import requests
from selenium import webdriver
import pandas as pd
import numpy as np
import time


# Scrape Fangraphs for MLB Data (incl. but not limited to WAR, OPS, etc.)
years = [2016, 2017, 2018, 2019, 2020]

appended_data = pd.DataFrame()

urls = []

for y in years:
    url = "https://www.fangraphs.com/leaders.aspx?pos=all&stats=bat&lg=all&qual=20&type=c,6,34,35,36,-1, " \
          "23,37,38,39,-1,40,60,41,-1,201,205,200,-1,52,51,50,61,58&season=" + str(y) + "&month=0&" \
             "season1=" + str(y) + "&ind=0&team=0&rost=0&age=0&filter=&players=0&startdate=" + str(y) + \
          "-01-01&enddate=" + str(y) + "-12-31&page=1_1000"
    urls.append(url)

for url in urls:
    series_url = pd.Series(url)
    df_url = series_url.str.split('=', expand=True)
    df_url.columns = ['protocol', 'blank', 'path_one', 'path_two', 'path_three', 'path_four', 'path_five', 'path_six',
                      'year', 'path_eight', 'path_nine', 'league', 'd', 'desc', 'file_name', 'a', 'b', 'c']
    df_url["year"] = df_url["year"].apply(lambda x: x.replace("&ind", ""))
    # then make that the year reference cell
    df_url.columns = ['protocol', 'blank', 'path_one', 'path_two', 'path_three', 'path_four', 'path_five', 'path_six',
                      'path_seven', 'path_eight', 'path_nine', 'league', 'year', 'desc', 'file_name', 'a', 'b', 'c']
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
    df_fix = pd.DataFrame(np.reshape(df.values, (df.shape[0] // 22, 22)),
                          columns=['Num', 'Name', "Team", "PA", "BB%", "K%", "BB/K", "AVG", "OBP", "SLG", "OPS", "ISO",
                                   "Spd", "BABIP", "UBR", "wGDP", "wSB", "wRC", "wRAA", "wOBA", "wRC+", "WAR"])
    df_fix['Year'] = df_url['year'][0]
    appended_data = appended_data.append(df_fix, ignore_index=True)
    time.sleep(5)
    driver.close()
