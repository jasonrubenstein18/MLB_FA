import pandas as pd
import time
import numpy as np

# Scrape ESPN Free Agent Tracker for free agent contracts
years = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]

espn_salary_data = pd.DataFrame()
espn_urls = []

for y in years:
    url = "http://www.espn.com/mlb/freeagents/_/year/" + str(y)
    espn_urls.append(url)

for url in espn_urls:
    salary_url = pd.Series(url)
    espn_url = salary_url.str.split('/|-', expand=True)
    espn_url.columns = ['protocol', 'blank', 'path_one', 'path_two', 'path_three', '_', 'fix', 'year']
    espn_salary_table = pd.read_html(url)[0]
    new_header = espn_salary_table.iloc[1]  # grab the first row for the header
    espn_salary_table = espn_salary_table[2:]  # take the data less the header row
    espn_salary_table.columns = new_header  # set the header row as the df header
    espn_salary_table = espn_salary_table.rename(columns={espn_salary_table.columns[4]: "PREV TEAM"})
    espn_salary_table = espn_salary_table[espn_salary_table['PLAYER'] != "PLAYER"].reset_index(drop=True)
    espn_salary_table['Season'] = espn_url['year'][0]
    espn_salary_data = espn_salary_data.append(espn_salary_table, ignore_index=True)
    time.sleep(5)

espn_salary_data['Season'] = espn_salary_data['Season'].astype('int') + 1
espn_salary_data['YRS'] = espn_salary_data['YRS'].fillna(0)
espn_salary_data['YRS'] = espn_salary_data['YRS'].astype('int')

espn_data = espn_salary_data[(espn_salary_data['Season']) < 2022]


# Scrape MLB Trade Rumors FA Tracker for option and qualifying offer data
tr_salary_data = pd.DataFrame()
mlb_tr_urls = []

for y in years:
    url = "https://www.mlbtraderumors.com/" + str(y) + "-mlb-free-agent-tracker/"
    mlb_tr_urls.append(url)

for url in mlb_tr_urls:
    tr_salary_url = pd.Series(url)
    tr_url = tr_salary_url.str.split('/|-', expand=True)
    tr_url.columns = ['protocol', 'blank', 'path_one', 'year', 'path_two', 'path_three',
                      'path_four', 'path_five', 'blank_two']
    tr_salary_table = pd.read_html(url)[1]
    tr_salary_table = tr_salary_table.rename(columns={"Qual?": "Qual"})
    tr_salary_table['Season'] = tr_url['year'][0]
    tr_salary_data = tr_salary_data.append(tr_salary_table, ignore_index=True)
    time.sleep(5)

mlb_tr_data = tr_salary_data

# Reformat dtypes
mlb_tr_data['Season'] = mlb_tr_data['Season'].astype('int')
mlb_tr_data['Years'] = mlb_tr_data['Years'].fillna(0)
mlb_tr_data['Years'] = mlb_tr_data['Years'].astype('int')


# Fix column names
espn_data.columns = ['Player', 'Position', 'Age', 'Status', 'Prev Team', 'Team', 'Years', 'Rank', 'Salary', 'Season']
mlb_tr_data.columns = ['Player', 'Position', 'Team', 'Qual', 'Years', 'Amount', 'AAV', 'Option', 'Season']

# Select certain columns for ease of view
mlb_tr_data = mlb_tr_data[['Player', 'Team', 'Qual', 'Years', 'Amount', 'Option', 'Season']]

# Merge espn and tr data on player name, team, and season year
salary_data = pd.merge(espn_data, mlb_tr_data, how='left',
                       left_on=['Player', 'Team', 'Season'], right_on=['Player', 'Team', 'Season'])

# salary_data['Qual'] = salary_data['Qual'].fillna(0)
# print(salary_data['Qual'].unique())


def salary_formatting(df):
    # Take max years #, determined from vesting or misreporting on some outlets
    df['Years_y'] = df['Years_y'].fillna(0)
    
    df['Years'] = np.where(df['Years_x'] >= df['Years_y'],
                           df['Years_x'],
                           df['Years_y'])
    del df['Years_x'], df['Years_y']

    # Remove rows where years == 0 aka player unsigned
    df = df[(df['Years'] > 0) | (df['Season'] == max(df['Season']))].reset_index(drop=True)
    
    # Begin reformatting salary data for analysis, replace $ sign
    df["Salary"] = df["Salary"].apply(lambda x: x.replace("$", ""))
    
    # replace NA from espn with mlb_tr data
    df['Salary'].fillna(df['Amount'], inplace=True)
    
    # replace -- from ESPN with mlb_tr data
    df['Salary'] = np.where(df['Salary'] == "--",
                            df['Amount'],
                            df['Salary'])
    
    # Secondary column used to reformat salaries such as ($1.5MM)
    df['Value'] = 0
    df.loc[df['Amount'].str.contains('$', na=False), 'Value'] = 1
    
    # Refine minor league definition
    df['Salary'] = np.where((df['Value'] == 1) & (df['Salary'] == "Minor Lg"),
                            df['Amount'],
                            df['Salary'])
    
    df['fix_salary_format'] = 0
    df.loc[df['Salary'].str.contains('MM', na=False), 'fix_salary_format'] = df['Salary']
    df['fix_salary_format'] = df['fix_salary_format'].str.replace("MM", "")
    df['fix_salary_format'] = df['fix_salary_format'].str.replace("$", "")
    df['fix_salary_format'] = df['fix_salary_format'].fillna(0)
    df['fix_salary_format'] = pd.to_numeric(df['fix_salary_format'])
    df['fix_salary_format'] = np.where(df['fix_salary_format'] > 0,
                                       df['fix_salary_format'] * 1000000,
                                       df['fix_salary_format'])
    df['Salary'] = np.where(df['fix_salary_format'] > 0,
                            df['fix_salary_format'],
                            df['Salary'])
    df['Salary'] = np.where((df['Salary'] == "Minor Lg") | (df['Salary'] == "Minor"),
                            600000,
                            df['Salary'])
    df['Salary'] = df['Salary'].str.replace(",", "")
    df['Salary'] = df['Salary'].fillna(0)

    # fix "K" values
    df['fix_salary_format'] = 0
    df.loc[df['Salary'].str.contains('K', na=False), 'fix_salary_format'] = df['Salary']
    df['fix_salary_format'] = df['fix_salary_format'].astype('str')
    df['fix_salary_format'] = df['fix_salary_format'].str.replace("K", "")
    df['fix_salary_format'] = df['fix_salary_format'].str.replace("$", "")
    df['fix_salary_format'] = pd.to_numeric(df['fix_salary_format'])
    df['fix_salary_format'] = df['fix_salary_format'] * 1000
    df['Salary'] = np.where(df['fix_salary_format'] > 0,
                            df['fix_salary_format'],
                            df['Salary'])
    df = df[(df['Salary'] != "Unknown")]
    df['Salary'] = pd.to_numeric(df['Salary'])
    
    # binary; 1 = received QO
    df['Qual'] = df['Qual'].fillna(0)
    df['Qual'] = df['Qual'].replace("No", 0)
    df['Qual'] = np.where(df['Qual'] != 0,
                          1, 0)
    
    # Option either Not Incl, Club, or Opt Out
    df['Option'] = df['Option'].fillna('No')
    
    del df['fix_salary_format'], df['Amount'], df['Value'], df['Rank']
    return df


salary_data = salary_formatting(salary_data)


# Remove 0 salary players
salary_data = salary_data[(salary_data['Salary'] > 0) | (salary_data['Season'] == max(salary_data['Season']))]

salary_data.to_csv("~/Desktop/MLB_FA/salary_data.csv")
