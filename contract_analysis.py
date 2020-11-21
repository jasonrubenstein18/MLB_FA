import pandas as pd
import math
import numpy as np

# Read in data
batter_data = pd.read_csv("~/Desktop/MLB_FA/fg_bat_data.csv")
del batter_data['Unnamed: 0']
print(len(batter_data))
print(batter_data.head())

pitcher_data = pd.read_csv("~/Desktop/MLB_FA/fg_pitch_data.csv")
del pitcher_data['Unnamed: 0']
print(len(pitcher_data))
print(pitcher_data.head())

salary_data = pd.read_csv("~/Desktop/MLB_FA/salary_data.csv")
del salary_data['Unnamed: 0']
print(len(salary_data))


# Define inflation
def npv(df, rate):
    r = rate
    df['Salary'] = pd.to_numeric(df['Salary'])
    df['AAV'] = salary_data['Salary'] / df['Years']
    df['NPV'] = 0
    df['NPV'] = round(df['AAV'] * (1 - (1 / ((1 + r) ** df['Years']))) / r, 2)
    return df


salary_data = npv(salary_data, 0.5)


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

        df['wRC+'] = pd.to_numeric(df['wRC+'])
        df['y_n1_wRC+'] = df.groupby("Name")['wRC+'].shift(1)
        df['y_n2_wRC+'] = df.groupby("Name")['y_n1_wRC+'].shift(1)
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

# Lag
batter_data = Metrics.lagged_batter(batter_data)
pitcher_data = Metrics.lagged_pitcher(pitcher_data)

# Non Linears
batter_data = NonLinearVars.fg_batter_vars(batter_data)
pitcher_data = NonLinearVars.fg_pitcher_vars(pitcher_data)
salary_data = NonLinearVars.salary_vars(salary_data)

# Merge data sets (one pitcher, one batter)
batter_merged = pd.merge(batter_data, salary_data, left_on=['Name', 'Year'], right_on=['Player', 'Season'])
print(len(batter_merged))

pitcher_merged = pd.merge(pitcher_data, salary_data, left_on=['Name', 'Year'], right_on=['Player', 'Season'])
print(len(pitcher_merged))

