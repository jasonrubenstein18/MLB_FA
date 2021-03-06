import pandas as pd
import numpy as np
import time
import datetime
import plotly_express as px
import requests

injury_data = pd.DataFrame()
injury_urls = []

lst = [x for x in range(0, 25300) if x % 25 == 0]

for i in lst:
    url = "http://www.prosportstransactions.com/baseball/Search/SearchResults.php?Player=&" \
          "Team=&BeginDate=&EndDate=&DLChkBx=yes&submit=Search&start=" + str(i)
    injury_urls.append(url)

k = 0
for url in injury_urls:
    k += 1
    # dl_url = pd.Series(url)
    # transaction_url = dl_url.str.split('/|-', expand=True)
    # transaction_url.columns = ['protocol', 'blank', 'path_one', 'year', 'path_two', 'path_three',
    #                            'path_four', 'path_five', 'blank_two']
    transaction_table = pd.read_html(url)[0]
    new_header = transaction_table.iloc[0]  # grab the first row for the header
    transaction_table = transaction_table[1:]  # take the data less the header row
    transaction_table.columns = new_header  # set the header row as the df header
    transaction_table['Relinquished'] = transaction_table['Relinquished'].str.replace("• ", "")
    transaction_table['Acquired'] = transaction_table['Acquired'].str.replace("• ", "")
    # transaction_table['Season'] = transaction_url['year'][0]
    injury_data = injury_data.append(transaction_table, ignore_index=True)
    print(k + " " + str(len(lst)))
    time.sleep(5)

injury_data.to_csv("~/Desktop/MLB_FA/injury_data.csv", index=False)

injury_df_working = pd.read_csv("~/Desktop/MLB_FA/injury_data.csv")


def reformatting(df):
    try:
        df[['Notes', 'Injury']] = df['Notes'].str.split(' with ', 1, expand=True)
        df[['Notes', 'Injury_2']] = df['Notes'].str.split(' recovering from ', 1, expand=True)
    except ValueError:
        pass
    df['Injury'] = np.where(df['Injury'] == "None",
                            df['Injury_2'],
                            df['Injury'])
    df['Injury'] = df['Injury'].fillna(df['Injury_2'])
    del df['Injury_2']
    return df


injury_in_sample = reformatting(injury_df_working)


# print(len(injury_df_working['Injury'].unique()))

# Keep only data that is within the analysis
def pre_process(df):
    df = df[~df.Notes.str.contains("transferred")]  # Remove transferring between DL's / IL's

    # Make single player column
    df["Player_split"] = np.nan
    df['Player_split'] = df['Player_split'].fillna(df['Acquired'])
    df['Player_split'] = df['Player_split'].fillna(df['Relinquished'])
    df['Player_split'] = df['Player_split'].str.replace(r"\(.*\)", "")
    df[['Player_split', 'Player']] = df['Player_split'].str.split(' / ', 1, expand=True)
    df['Player'] = df['Player'].fillna(df['Player_split'])
    del df['Player_split']

    # Remove non DL / IL rows
    df = df[(df['Notes'].str.contains("DL")) |
                                    (df['Notes'].str.contains("IL"))
                                    ]
    # Shift down on the player
    df['injury_end'] = df.groupby("Player")['Date'].shift(-1)

    # grandy = injury_fixed[(injury_fixed['Player'] == "Curtis Granderson")]

    # Now remove rows where Relinquished column is nan
    df = df[df['Relinquished'].notna()]
    df = df[df['injury_end'].notna()]

    # Keep data in sample
    df = df[(df['Date'] > '2010-01-01')].reset_index(drop=True)

    # Timedelta of injuries
    df[['Date','injury_end']] = df[['Date','injury_end']].apply(pd.to_datetime)  # if conversion required
    df['injury_duration'] = (df['injury_end'] - df['Date']).dt.days

    df = df[(df['injury_duration'] <= 365)]
    return df


injury_working = pre_process(injury_in_sample)

# Plot a histogram of injury duration
fig = px.histogram(injury_working, x="injury_duration", title="Histogram of DL / IL stint length since 2010")
fig.show()


def grouping(df):
    array_func = lambda x: set(x)
    injuries_array = df.pivot_table(index=['Player', 'Year'], values='Injury',
                                    aggfunc=array_func, margins=False).reset_index()
    df['Year'] = df['Date'].dt.year + 1  # Create season variable to align with year of FA
    df = df.groupby(['Year', 'Player']).agg({
        'injury_duration': sum,
    }).reset_index()

    df_merged = pd.merge(df, injuries_array, on=['Player', 'Year'], how='left')
    return df_merged


injury_final = grouping(injury_working)


# Plot a histogram of injury total
fig = px.histogram(injury_final, x="injury_total", title="Histogram of player season total DL / IL time since 2010")
fig.show()

# short_stint = injury_working[(injury_working['injury_duration'] < 10) & (injury_working['Team'] == "Mets")]

# Find % of free agent contract signers that are in this injury dataset
# Find % of free agent players who are in the fangraphs dataset

injury_final.to_csv("~/Desktop/MLB_FA/pst_injury_data.csv", index=False)


### MLB dot com transactions

mlb_injury_data = pd.DataFrame()
mlb_injury_urls = []

for year in range(2005, 2019, 1):
    r = requests.get("http://lookup-service-prod.mlb.com/json/named.transaction_all.bam?start_date={0}0101&e"
                     "nd_date={1}1231&sport_code=%27mlb%27".format(year, year+1)).json()
    transaction_data = pd.DataFrame(r['transaction_all']['queryResults']['row'])
    mlb_injury_data = mlb_injury_data.append(transaction_data, ignore_index=True)

inj_only = mlb_injury_data[(mlb_injury_data['type'] == "Status Change")]

inj_only.to_csv("~/Desktop/MLB_FA/mlb_injury_data.csv", index=False)
