import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
import plotly_express
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Read in data
batter_data = pd.read_csv("~/Desktop/MLB_FA/Data/fg_bat_data.csv")
del batter_data['Age']
print(len(batter_data))
print(batter_data.head())

pitcher_data = pd.read_csv("~/Desktop/MLB_FA/Data/fg_pitch_data.csv")
del pitcher_data['Age']
print(len(pitcher_data))
print(pitcher_data.head())

salary_data = pd.read_csv("~/Desktop/MLB_FA/Data/salary_data.csv")
print(len(salary_data))

injury_data = pd.read_csv("~/Desktop/MLB_FA/Data/injury_data_use.csv")

# Check for whether there is overlap between injury data and the salary data players
# injury_data_players = injury_data['Player'].unique()
# mutual = salary_data[salary_data['Player'].isin(injury_data_players)]  # 945 out of 1135 players included
# excl = salary_data[~salary_data['Player'].isin(injury_data_players)]

# print(len(excl['Player'].unique()))  # 129 unique players injury data omitted; use mlb.com trans for these


# Define inflation
def npv(df, rate):
    r = rate
    df['Salary'] = pd.to_numeric(df['Salary'])
    df['AAV'] = salary_data['Salary'] / df['Years']
    df['NPV'] = 0
    df['NPV'] = round(df['AAV'] * (1 - (1 / ((1 + r) ** df['Years']))) / r, 2)
    return df


salary_data = npv(salary_data, 0.05)


# Lagged metrics to see if there is carryover value / value in continuity
class Metrics:
    def lagged_batter(df):
        df['WAR'] = pd.to_numeric(df['WAR'])
        df['y_n1_war'] = df.groupby("Name")['WAR'].shift(1)
        df['y_n2_war'] = df.groupby("Name")['y_n1_war'].shift(1)
        df['y_n3_war'] = df.groupby("Name")['y_n2_war'].shift(1)
        df['y_n4_war'] = df.groupby("Name")['y_n3_war'].shift(1)
        df['y_n5_war'] = df.groupby("Name")['y_n4_war'].shift(1)
        df['y_n6_war'] = df.groupby("Name")['y_n5_war'].shift(1)

        df['wOBA'] = pd.to_numeric(df['wOBA'])
        df['y_n1_wOBA'] = df.groupby("Name")['wOBA'].shift(1)
        df['y_n2_wOBA'] = df.groupby("Name")['y_n1_wOBA'].shift(1)
        df['y_n3_wOBA'] = df.groupby("Name")['y_n2_wOBA'].shift(1)
        df['y_n4_wOBA'] = df.groupby("Name")['y_n3_wOBA'].shift(1)

        df['wRC+'] = pd.to_numeric(df['wRC+'])
        df['y_n1_wRC+'] = df.groupby("Name")['wRC+'].shift(1)
        df['y_n2_wRC+'] = df.groupby("Name")['y_n1_wRC+'].shift(1)

        df['y_n1_war_pa'] = df.groupby("Name")['WAR_PA'].shift(1)
        df['y_n2_war_pa'] = df.groupby("Name")['y_n1_war_pa'].shift(1)
        df['y_n3_war_pa'] = df.groupby("Name")['y_n2_war_pa'].shift(1)
        df['y_n4_war_pa'] = df.groupby("Name")['y_n3_war_pa'].shift(1)
        df['y_n5_war_pa'] = df.groupby("Name")['y_n4_war_pa'].shift(1)
        df['y_n6_war_pa'] = df.groupby("Name")['y_n5_war_pa'].shift(1)

        df["BB%"] = df["BB%"].apply(lambda x: x.replace("%", ""))
        df['BB%'] = pd.to_numeric(df['BB%'])
        df["K%"] = df["K%"].apply(lambda x: x.replace("%", ""))
        df['K%'] = pd.to_numeric(df['K%'])

        df.rename(columns={'BB%': 'BBpct', 'K%': 'Kpct'}, inplace=True)
        return df

    def lagged_pitcher(df):
        df['WAR'] = pd.to_numeric(df['WAR'])
        df['y_n1_war'] = df.groupby("Name")['WAR'].shift(1)
        df['y_n2_war'] = df.groupby("Name")['y_n1_war'].shift(1)
        df['y_n3_war'] = df.groupby("Name")['y_n2_war'].shift(1)
        df['y_n4_war'] = df.groupby("Name")['y_n3_war'].shift(1)
        df['y_n5_war'] = df.groupby("Name")['y_n4_war'].shift(1)
        df['y_n6_war'] = df.groupby("Name")['y_n5_war'].shift(1)

        # df['ERA-'] = pd.to_numeric(df['ERA-'])
        # df['y_n1_ERA-'] = df.groupby("Name")['ERA-'].shift(1)
        # df['y_n2_ERA-'] = df.groupby("Name")['y_n1_ERA-'].shift(1)

        df['xFIP'] = pd.to_numeric(df['xFIP'])
        df['y_n1_xFIP'] = df.groupby("Name")['xFIP'].shift(1)
        df['y_n2_xFIP'] = df.groupby("Name")['y_n1_xFIP'].shift(1)

        df['y_n1_war_tbf'] = df.groupby("Name")['WAR_TBF'].shift(1)
        df['y_n2_war_tbf'] = df.groupby("Name")['y_n1_war_tbf'].shift(1)
        df['y_n3_war_tbf'] = df.groupby("Name")['y_n2_war_tbf'].shift(1)
        df['y_n4_war_tbf'] = df.groupby("Name")['y_n3_war_tbf'].shift(1)
        df['y_n5_war_tbf'] = df.groupby("Name")['y_n4_war_tbf'].shift(1)
        df['y_n6_war_tbf'] = df.groupby("Name")['y_n5_war_tbf'].shift(1)

        df['BB%'] = df['BB%'].astype(str)
        df["BB%"] = df["BB%"].apply(lambda x: x.replace("%", ""))
        df['BB%'] = pd.to_numeric(df['BB%'])

        df['K%'] = df['K%'].astype(str)
        df["K%"] = df["K%"].apply(lambda x: x.replace("%", ""))
        df['K%'] = pd.to_numeric(df['K%'])

        df['K-BB%'] = df['K-BB%'].astype(str)
        df["K-BB%"] = df["K-BB%"].apply(lambda x: x.replace("%", ""))
        df['K-BB%'] = pd.to_numeric(df['K-BB%'])

        df['SwStr%'] = df['SwStr%'].astype(str)
        df["SwStr%"] = df["SwStr%"].apply(lambda x: x.replace("%", ""))
        df['SwStr%'] = pd.to_numeric(df['SwStr%'])

        df['LOB%'] = df['LOB%'].astype(str)
        df["LOB%"] = df["LOB%"].apply(lambda x: x.replace("%", ""))
        df['LOB%'] = pd.to_numeric(df['LOB%'])


        # df['CB%'] = df['CB%'].astype(str)
        # df["CB%"] = df["CB%"].apply(lambda x: x.replace("%", ""))
        # df['CB%'] = pd.to_numeric(df['CB%'])

        df.rename(columns={'BB%': 'BBpct', 'K%': 'Kpct', 'K-BB%': 'K_minus_BBpct', 'CB%': 'CBpct',
                           'SwStr%': 'Swstrpct'}, inplace=True)
        return df

    def fix_position(df):
        df['Position'] = np.where(df['Position'] == "OF", "CF", df['Position'])
        df['Position'] = np.where((df['Position'] == "LF") | (df['Position'] == "RF"),
                                  "Corner Outfield", df['Position'])
        df['Position'] = np.where(df['Position'] == "P", "RP", df['Position'])
        # df['Position'] = np.where(df['Position'] == "SP", 1, df['Position'])
        # df['Position'] = np.where(df['Position'] == "C", 2, df['Position'])
        # df['Position'] = np.where(df['Position'] == "1B", 3, df['Position'])
        # df['Position'] = np.where(df['Position'] == "2B", 4, df['Position'])
        # df['Position'] = np.where(df['Position'] == "3B", 5, df['Position'])
        # df['Position'] = np.where(df['Position'] == "SS", 6, df['Position'])
        # df['Position'] = np.where(df['Position'] == "Corner Outfield", 7, df['Position'])
        # df['Position'] = np.where(df['Position'] == "CF", 8, df['Position'])
        # df['Position'] = np.where(df['Position'] == "RP", 9, df['Position'])
        # df['Position'] = np.where(df['Position'] == "DH", 10, df['Position'])
        return df

    def rate_stats_batter(df):
        df['WAR_PA'] = df['WAR'] / df['PA']  # add in rate based WAR (per PA, game played, etc)
        df['oWAR_PA'] = df['oWAR'] / df['PA']

        df['WAR_PA'] = round(df['WAR_PA'], 3)
        df['oWAR_PA'] = round(df['oWAR_PA'], 3)
        return df

    def rate_stats_pitcher(df):
        df['WAR_TBF'] = df['WAR'] / df['TBF']  # add in rate based WAR (per IP, etc)
        # df['WAR_IP'] = df['WAR'] / df['IP']
        df['wFB_TBF'] = df['wFB'] / df['TBF']

        df['WAR_TBF'] = round(df['WAR_TBF'], 3)
        # df['WAR_IP'] = round(df['WAR_IP'], 3)
        df['wFB_TBF'] = round(df['wFB_TBF'], 3)
        return df

    def injury_engineering(df):
        df['two_year_inj_avg'] = 0
        df.loc[:, "two_year_inj_avg"] = (
                df.groupby("Player")["injury_duration"].shift(1) / df.groupby("Player")["injury_duration"].shift(
            2) - 1)
        df['Injury'] = df['Injury'].fillna("None")
        df['injury_duration'] = df['injury_duration'].fillna(0)
        return df

    def short_season_fix_batter(df):
        df['WAR_162'] = np.where(df['Year'] == 2021, df['WAR']*2.3, df['WAR'])
        df['PA_162'] = np.where(df['Year'] == 2021, df['PA']*2.3, df['PA'])
        df['oWAR_162'] = np.where(df['Year'] == 2021, df['oWAR'] * 2.3, df['oWAR'])
        df['dWAR_162'] = np.where(df['Year'] == 2021, df['dWAR'] * 2.3, df['dWAR'])
        return df

    def short_season_fix_pitcher(df):
        df['WAR_162'] = np.where(df['Year'] == 2021, df['WAR']*2.3, df['WAR'])
        df['IP_162'] = np.where(df['Year'] == 2021, df['IP']*2.3, df['IP'])
        return df


class NonLinearVars():
    def fg_batter_vars(df):
        df['WAR_sq'] = np.where(df['WAR'] > 0, df['WAR'] ** 2, df['WAR'] * 2)
        df['y_n1_war_sq'] = np.where(df['y_n1_war'] > 0, df['y_n1_war'] ** 2, df['y_n1_war'] * 2)
        df['y_n2_war_sq'] = np.where(df['y_n2_war'] > 0, df['y_n2_war'] ** 2, df['y_n2_war'] * 2)
        df['y_n3_war_sq'] = np.where(df['y_n3_war'] > 0, df['y_n3_war'] ** 2, df['y_n3_war'] * 2)
        df['y_n4_war_sq'] = np.where(df['y_n4_war'] > 0, df['y_n4_war'] ** 2, df['y_n4_war'] * 2)
        df['y_n5_war_sq'] = np.where(df['y_n5_war'] > 0, df['y_n5_war'] ** 2, df['y_n5_war'] * 2)
        df['y_n6_war_sq'] = np.where(df['y_n6_war'] > 0, df['y_n6_war'] ** 2, df['y_n6_war'] * 2)
        df['y_n1_wOBA_sq'] = df['y_n1_wOBA'] ** 2
        df['y_n2_wOBA_sq'] = df['y_n2_wOBA'] ** 2
        df['y_n1_wRC+_sq'] = df['y_n1_wRC+'] ** 2
        df['y_n2_wRC+_sq'] = df['y_n2_wRC+'] ** 2
        return df

    def fg_pitcher_vars(df):
        df['WAR_sq'] = df['WAR'] **2
        df['y_n1_war_sq'] = np.where(df['y_n1_war'] > 0, df['y_n1_war'] ** 2, df['y_n1_war'] * 2)
        df['y_n2_war_sq'] = np.where(df['y_n2_war'] > 0, df['y_n2_war'] ** 2, df['y_n2_war'] * 2)
        df['y_n3_war_sq'] = np.where(df['y_n3_war'] > 0, df['y_n3_war'] ** 2, df['y_n3_war'] * 2)
        df['y_n4_war_sq'] = np.where(df['y_n4_war'] > 0, df['y_n4_war'] ** 2, df['y_n4_war'] * 2)
        df['y_n5_war_sq'] = np.where(df['y_n5_war'] > 0, df['y_n5_war'] ** 2, df['y_n5_war'] * 2)
        df['y_n6_war_sq'] = np.where(df['y_n6_war'] > 0, df['y_n6_war'] ** 2, df['y_n6_war'] * 2)
        # df['ERA-_sq'] = df['ERA-'] **2
        # df['y_n1_ERA-_sq'] = df['y_n1_ERA-'] **2
        # df['y_n2_ERA-_sq'] = df['y_n2_ERA-'] **2
        df['xFIP_sq'] = df['xFIP'] **2
        df['y_n1_xFIP_sq'] = df['y_n1_xFIP'] **2
        df['y_n2_xFIP_sq'] = df['y_n2_xFIP'] **2
        return df

    def salary_vars(df):
        # df['Age'] = df['Age'].astype('int')
        df['Age_sq'] = df['Age'] ** 2
        df['Age_log'] = np.log(df['Age'])

        return df


# Attach the injury data to the players, merge on player and year
def merge_injuries(salary_df, injury_df):
    merged_df = pd.merge(salary_df, injury_df, how='left', left_on=['Player', 'Season'], right_on=['Player', 'Year'])
    del merged_df['Year']
    return merged_df


# MA
print(len(salary_data))
salary_data = merge_injuries(salary_data, injury_data)
print(len(salary_data))
salary_data['injury_duration'] = salary_data['injury_duration'].fillna(0)
salary_data = Metrics.injury_engineering(salary_data)


# Lag
batter_data = Metrics.short_season_fix_batter(batter_data)
batter_data = Metrics.rate_stats_batter(batter_data)
batter_data = Metrics.lagged_batter(batter_data)
pitcher_data = Metrics.short_season_fix_pitcher(pitcher_data)
pitcher_data = Metrics.rate_stats_pitcher(pitcher_data)
pitcher_data = Metrics.lagged_pitcher(pitcher_data)

# Position fix
salary_data = Metrics.fix_position(salary_data)

# Non Linears
batter_data = NonLinearVars.fg_batter_vars(batter_data)
pitcher_data = NonLinearVars.fg_pitcher_vars(pitcher_data)
salary_data = NonLinearVars.salary_vars(salary_data)

# Merge data sets (one pitcher, one batter)
batter_merged = pd.merge(batter_data, salary_data, left_on=['Name', 'Year'], right_on=['Player', 'Season'])
batter_merged = batter_merged[(batter_merged['Position'] != "SP") & (batter_merged['Position'] != "RP")]  # remove P's
print(len(batter_merged))

pitcher_merged = pd.merge(pitcher_data, salary_data, left_on=['Name', 'Year'], right_on=['Player', 'Season'])
pitcher_merged = pitcher_merged[(pitcher_merged['Position'] == "SP") | (pitcher_merged['Position'] == "RP")]  # keep P's
print(len(pitcher_merged))

# Begin modeling
# train_data_batter = batter_merged[(batter_merged['Year'] != max(batter_merged['Year']))]
# train_data_pitcher = pitcher_merged[(pitcher_merged['Year'] != max(pitcher_merged['Year']))]

train_data_batter = batter_merged.loc[~batter_merged['NPV'].isnull()]
train_data_pitcher = pitcher_merged.loc[~pitcher_merged['NPV'].isnull()]
test_data_batter = batter_merged[
    # (batter_merged['Year'] == max(batter_merged['Year']))
    # &
    (np.isnan(batter_merged['NPV']))]
test_data_pitcher = pitcher_merged[
    # (pitcher_merged['Year'] == max(pitcher_merged['Year']))
    # &
    (np.isnan(pitcher_merged['NPV']))]

train_data_batter.to_csv('~/Desktop/MLB_FA/Data/train_data_batter.csv', index=False)
train_data_pitcher.to_csv('~/Desktop/MLB_FA/Data/train_data_pitcher.csv', index=False)
test_data_batter.to_csv('~/Desktop/MLB_FA/Data/test_data_batter.csv', index=False)
test_data_pitcher.to_csv('~/Desktop/MLB_FA/Data/test_data_pitcher.csv', index=False)

fit = ols('NPV ~ C(Position) + WAR_sq + WAR + Age', data=train_data_batter).fit()
fit.summary()  # 0.597 r-sq, 0.587 adj r-sq

# Plot NPV / WAR to see nonlinear relationship
plot_data = train_data_batter[(train_data_batter['Year'] > 2010)]
fig = plotly_express.scatter(plot_data, x="dWAR", y="NPV", color='Position',
                             hover_data=['Player', 'Position', 'Year', 'Prev Team'],
                             title="dWAR, NPV Colored By Position (since {})".format(min(plot_data['Year'])))
fig.show()

# Plot WAR / Rate WAR
plot_data = batter_data[(batter_data['Year'] == 2021) & (batter_data['PA'] > 100)]
fig = plotly_express.scatter(plot_data, x="PA", y="dWAR", color='Name')
fig.update_layout(
    hoverlabel=dict(
        bgcolor="white",
        font_size=10,
        font_family="Arial"
    )
)
fig.show()

# remove linear WAR
# Let's add a season factor and qualifying offer
fit = ols('NPV ~ C(Position) + C(Season) + WAR_sq + Age + Qual + WAR_PA', data=train_data_batter).fit()
fit.summary()

# Getting better, but there's more unexplained variance. Let's try log of Age and prior season's WAR
# Log Age
fit = ols('NPV ~ C(Position) + C(Season) + y_n1_war_sq +  WAR_sq + Age_log + Qual + WAR_PA + y_n1_war_pa',
          data=train_data_batter).fit()
fit.summary()

# Still marginally improving. Up to around 50% of the variance explained.
# WAR is a counting stat, let's add in base-running UBR, non-log Age
# UBR
fit = ols('NPV ~ C(Position) + y_n1_war_sq +  WAR_sq + Age + UBR + Qual', data=train_data_batter).fit()
fit.summary()

# Try some new variables (e.g. OPS, ISO, wRC+, wOBA, y_n2_war_sq, etc)
fit = ols('NPV ~ C(Position) + y_n2_war_sq + y_n1_war_sq +  WAR_sq + Age + UBR + Qual + wOBA + ISO',
          data=train_data_batter).fit()
fit.summary()

# Now let's consider only deals signed for multiple-years
train_data_batter_multiyear = train_data_batter[(train_data_batter['Years'] > 1)]
fit = ols('NPV ~ C(Position) + y_n1_war_sq +  WAR_sq + Age + UBR + Qual', data=train_data_batter_multiyear).fit()
fit.summary()

# Single year only
train_data_batter_single = train_data_batter[(train_data_batter['Years'] == 1)]
fit = ols('NPV ~ C(Position) + y_n1_war_sq +  WAR_sq + Age + Qual', data=train_data_batter_single).fit()
fit.summary()

# So what are team's using to assess these single year contracts?
fit = ols('NPV ~ ISO + WAR_sq + y_n1_war_sq + y_n2_war_sq + wGDP + BABIP + Qual', data=train_data_batter_single).fit()
fit.summary()

# Now add injury duration
fit = ols('NPV ~ ISO + WAR_sq + y_n1_war_sq + y_n2_war_sq + injury_duration + Qual', data=train_data_batter).fit()
fit.summary()

# Kitchen sink
fit_rate = ols('NPV ~ BBpct + Kpct + AVG + OBP + SLG + OPS + ISO + Spd + BABIP + UBR + wGDP + wSB + wRC + '
               'wRAA + wOBA + WAR + dWAR + oWAR + Year + WAR_PA + oWAR_PA + y_n1_war + y_n2_war + y_n3_war + '
               'y_n4_war + y_n5_war + y_n6_war + y_n1_wOBA + y_n2_wOBA + y_n3_wOBA + y_n4_wOBA + '
               'y_n1_war_pa + y_n2_war_pa + y_n3_war_pa + y_n4_war_pa + y_n5_war_pa + y_n6_war_pa +'
               'WAR_sq + y_n1_war_sq + y_n2_war_sq + y_n3_war_sq + y_n4_war_sq + y_n5_war_sq + y_n6_war_sq + '
               'y_n1_wOBA_sq + y_n2_wOBA_sq + Position + Age + Qual + injury_duration', data=train_data_batter).fit()
fit_rate.summary()

# Remove unwanted vars
fit_rate = ols('NPV ~ Kpct + Year + y_n1_war +'
               'y_n1_wOBA + y_n2_war_pa + WAR_sq + y_n1_war_sq +'
               'Age + Qual', data=train_data_batter).fit()
fit_rate.summary()

# PITCHERS
train_data_pitcher['pos_dummy'] = np.where(train_data_pitcher['Position'] == "SP", 1, 0)
fit = ols('NPV ~ WAR_sq + Age + Qual + pos_dummy + FBv + Kpct + y_n1_war_sq', data=train_data_pitcher).fit()
fit.summary()

# Predict WAR
fit = ols('WAR ~ FBv + Kpct + BBpct + FIP + IP + wFB + pos_dummy', data=train_data_pitcher).fit()
fit.summary()

# Let's add in injury duration
train_data_pitcher['injury_duration_log'] = np.log(train_data_pitcher['injury_duration'])
fit = ols('NPV ~ WAR_sq + Age + Qual + injury_duration + pos_dummy', data=train_data_pitcher).fit()
fit.summary()

# Add FBv
fit = ols('NPV ~ WAR_sq + Age + Qual + injury_duration + FBv + pos_dummy', data=train_data_pitcher).fit()
fit.summary()

# Kpct
fit = ols('NPV ~ WAR_sq + Age + Qual + injury_duration + FBv + Kpct + pos_dummy + BBpct', data=train_data_pitcher).fit()
fit.summary()

# CBv
fit = ols('NPV ~ Age + Qual + injury_duration + FBv + Kpct + CBv + pos_dummy', data=train_data_pitcher).fit()
fit.summary()

# Rate stats
fit_rate = ols(
    'NPV ~ Age + WAR_TBF + y_n1_war_tbf + y_n2_war_tbf + FBv + xFIP_sq + pos_dummy + injury_duration + Qual',
    data=train_data_pitcher).fit()
fit_rate.summary()

multi_year_pitcher = train_data_pitcher[(train_data_pitcher['Years'] > 1)]
fit_rate_multi = ols(
    'NPV ~ Age + WAR_TBF + y_n1_war_tbf + y_n2_war_tbf + FBv + xFIP_sq + pos_dummy + injury_duration',
    data=multi_year_pitcher).fit()
fit_rate_multi.summary()

# Change position and Season to random effect

batter_grp = batter_merged.groupby(['Season']).agg({
    'NPV': sum,
    'WAR': sum,
    'Name': 'nunique'
}).reset_index()

batter_grp['NPV'] = batter_grp['NPV'] / 1000000

fig = plotly_express.bar(batter_grp, x="Season", y="NPV",
                         color_continuous_scale=plotly_express.colors.qualitative.D3,
                         title="Yearly total NPV and total WAR")
fig.add_trace(go.Scatter(x=batter_grp['Season'], y=batter_grp['WAR'], line=dict(color='red'), name='WAR'),
              row=1, col=1)
fig.show()


# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Bar(x=batter_grp['Season'], y=batter_grp['NPV'], name="NPV total"),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=batter_grp['Season'], y=batter_grp['WAR'], name="WAR total"),
    secondary_y=True,
)

# Add figure title
fig.update_layout(
    title_text="Yearly total NPV and total WAR"
)

# Set x-axis title
fig.update_xaxes(title_text="Off-Season Year")

# Set y-axes titles
fig.update_yaxes(title_text="<b>NPV</b> total ($ Millions)", secondary_y=False)
fig.update_yaxes(title_text="<b>WAR</b> total", secondary_y=True)

fig.show()
