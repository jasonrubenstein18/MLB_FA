import pandas as pd
import numpy as np
from sklearn import linear_model
import statsmodels.api as sm
from statsmodels.formula.api import ols
import plotly_express

# Read in data
batter_data = pd.read_csv("~/Desktop/MLB_FA/fg_bat_data.csv")
del batter_data['Unnamed: 0']
print(len(batter_data))
print(batter_data.head())

pitcher_data = pd.read_csv("~/Desktop/MLB_FA/fg_pitch_data.csv")
print(len(pitcher_data))
print(pitcher_data.head())

salary_data = pd.read_csv("~/Desktop/MLB_FA/salary_data.csv")
del salary_data['Unnamed: 0']
print(len(salary_data))

injury_data = pd.read_csv("~/Desktop/MLB_FA/injury_data_use.csv")

# Check for whether there is overlap between injury data and the salary data players
injury_data_players = injury_data['Player'].unique()
mutual = salary_data[salary_data['Player'].isin(injury_data_players)]  # 945 out of 1135 players included
excl = salary_data[~salary_data['Player'].isin(injury_data_players)]

print(len(excl['Player'].unique()))  # 129 unique players injury data omitted; need to leverage mlb.com trans for these


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

        df['CB%'] = pd.to_numeric(df['CB%'])

        df.rename(columns={'BB%': 'BBpct', 'K%': 'Kpct', 'K-BB%': 'K_minus_BBpct', 'CB%': 'CBpct'}, inplace=True)
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

        return df

    def rate_stats_pitcher(df):
        df['WAR_TBF'] = df['WAR'] / df['TBF']  # add in rate based WAR (per IP, etc)

        return df

    def injury_engineering(df):
        df['two_year_inj_avg'] = 0
        df.loc[:, "two_year_inj_avg"] = (
                df.groupby("Player")["injury_duration"].shift(1) / df.groupby("Player")["injury_duration"].shift(
            2) - 1)
        return df


class NonLinearVars():
    def fg_batter_vars(df):
        df['WAR_sq'] = df['WAR'] ** 2
        df['y_n1_war_sq'] = df['y_n1_war'] ** 2
        df['y_n2_war_sq'] = df['y_n2_war'] ** 2
        df['y_n3_war_sq'] = df['y_n3_war'] ** 2
        df['y_n4_war_sq'] = df['y_n4_war'] ** 2
        df['y_n5_war_sq'] = df['y_n5_war'] ** 2
        df['y_n6_war_sq'] = df['y_n6_war'] ** 2
        df['y_n1_wOBA_sq'] = df['y_n1_wOBA'] ** 2
        df['y_n2_wOBA_sq'] = df['y_n2_wOBA'] ** 2
        df['y_n1_wRC+_sq'] = df['y_n1_wRC+'] ** 2
        df['y_n2_wRC+_sq'] = df['y_n2_wRC+'] ** 2
        return df

    def fg_pitcher_vars(df):
        df['WAR_sq'] = df['WAR'] **2
        df['y_n1_war_sq'] = df['y_n1_war'] **2
        df['y_n2_war_sq'] = df['y_n2_war'] **2
        df['y_n3_war_sq'] = df['y_n3_war'] **2
        df['y_n3_war_sq'] = df['y_n4_war'] **2
        df['y_n3_war_sq'] = df['y_n5_war'] **2
        df['y_n3_war_sq'] = df['y_n6_war'] **2
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


print(len(salary_data))
salary_data = merge_injuries(salary_data, injury_data)
print(len(salary_data))

# Lag
batter_data = Metrics.rate_stats_batter(batter_data)
batter_data = Metrics.lagged_batter(batter_data)
pitcher_data = Metrics.rate_stats_pitcher(pitcher_data)
pitcher_data = Metrics.lagged_pitcher(pitcher_data)

# Position fix
salary_data = Metrics.fix_position(salary_data)

# MA
injury_data = Metrics.injury_engineering(injury_data)

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
train_data_batter = batter_merged[(batter_merged['Year'] != max(batter_merged['Year']))]
train_data_pitcher = pitcher_merged[(pitcher_merged['Year'] != max(pitcher_merged['Year']))]

test_data_batter = batter_merged[(batter_merged['Year'] == max(batter_merged['Year'])) &
                                 (np.isnan(batter_merged['NPV']))]
test_data_pitcher = pitcher_merged[(pitcher_merged['Year'] == max(pitcher_merged['Year'])) &
                                   (np.isnan(pitcher_merged['NPV']))]

fit = ols('NPV ~ C(Position) + WAR_sq + WAR + Age', data=train_data_batter).fit()
fit.summary()  # 0.682 r-sq, 0.674 adj r-sq

# Plot NPV / WAR to see nonlinear relationship
plot_data = train_data_batter[(train_data_batter['Year'] > 2015)]
fig = plotly_express.scatter(plot_data, x="WAR", y="NPV",
                             hover_data=['Player', 'Position'])
fig.show()

# Plot WAR / WAR_PA
plot_data = batter_data[(batter_data['Year'] == 2021) & (batter_data['PA'] > 5)]
fig = plotly_express.scatter(plot_data, x="WAR", y="WAR_PA", color='Name',
                             hover_data=['Name', 'Year'])
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


# PITCHERS
fit = ols('NPV ~ C(Position) + WAR_sq + Age + Qual', data=train_data_pitcher).fit()
fit.summary()

# Let's add in injury duration
train_data_pitcher['injury_duration_log'] = np.log(train_data_pitcher['injury_duration'])
fit = ols('NPV ~ C(Position) + WAR_sq + Age + Qual + injury_duration', data=train_data_pitcher).fit()
fit.summary()

# Add FBv
fit = ols('NPV ~ C(Position) + WAR_sq + Age + Qual + injury_duration + FBv', data=train_data_pitcher).fit()
fit.summary()

# Kpct
fit = ols('NPV ~ Age + Qual + injury_duration + FBv + Kpct', data=train_data_pitcher).fit()
fit.summary()

# CBv
fit = ols('NPV ~ Age + Qual + injury_duration + FBv + Kpct + CBv', data=train_data_pitcher).fit()
fit.summary()

print(list(train_data_pitcher))


# Clean up more metrics as seen in Class


# Out of sample / in sample (pre 2021)

X_train = train_data_batter[['WAR', 'WAR_sq', 'Age', 'Position']]
Y_train = train_data_batter['NPV']
regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# prediction with sklearn
WAR = 3
WAR_sq = WAR ** 2
AGE = 30
POSITION = 3
print('Predicted NPV: \n', regr.predict([[WAR, WAR_sq, AGE, POSITION]]))
print('Accuracy Score: ', regr.score(X_train, Y_train))

# with statsmodels
X = sm.add_constant(X_train)  # adding a constant

model = sm.OLS(Y_train, X_train).fit()
predictions = model.predict(X_train)

print_model = model.summary()
print(print_model)
