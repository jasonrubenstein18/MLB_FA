import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
import numpy as np
from numpy import mean
from numpy import std
from numpy import absolute

# Read Data
train_data_batter = pd.read_csv("~/Desktop/MLB_FA/Data/train_data_batter.csv")
train_data_pitcher = pd.read_csv("~/Desktop/MLB_FA/Data/train_data_pitcher.csv")
salary_data = pd.read_csv("~/Desktop/MLB_FA/Data/salary_data.csv")
injury_data = pd.read_csv("~/Desktop/MLB_FA/Data/injury_data_use.csv")
# batter_data = pd.read_csv("~/Desktop/MLB_FA/Data/fg_bat_data.csv")
# pitcher_data = pd.read_csv("~/Desktop/MLB_FA/Data/fg_pitch_data.csv")


# Batters
batter_model_data = train_data_batter[['NPV', 'Kpct', 'wOBA', 'Year', 'WAR_sq',
                                       'y_n1_war_sq', 'Position',
                                       'y_n2_war_sq', 'Age', 'Qual', 'injury_duration']]  # .join(pos_dummies)

batter_model_data = batter_model_data.dropna()

X_train = batter_model_data[['Kpct', 'Year', 'WAR_sq',
                             'y_n1_war_sq',
                             # '1B', '2B', '3B', 'SS', 'C', 'Corner Outfield', 'CF', 'DH',
                             'Age', 'Qual', 'injury_duration']]
Y_train = batter_model_data['NPV']


# Linear regression
regr = linear_model.LinearRegression()

regr.fit(X_train, Y_train)
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


# prediction with sklearn
def predict_batter_contract(player):
    model_data = train_data_batter[['AAV', 'NPV', 'Kpct', 'wOBA', 'Year', 'WAR_sq',
                                    'y_n1_war_sq', 'Position',
                                    'y_n2_war_sq', 'Age', 'Qual', 'injury_duration']]  # .join(pos_dummies)

    model_data = model_data.dropna()

    X_train = model_data[['Kpct', 'Year', 'WAR_sq',
                          'y_n1_war_sq',
                          # '1B', '2B', '3B', 'SS', 'C', 'Corner Outfield', 'CF', 'DH',
                          'Age', 'Qual', 'injury_duration']]
    Y_train = model_data['NPV']
    regr = linear_model.LinearRegression()

    regr.fit(X_train, Y_train)
    # print('Intercept: \n', regr.intercept_)
    # print('Coefficients: \n', regr.coef_)
    Y_train_aav = model_data['AAV']
    regr_aav = linear_model.LinearRegression()

    regr_aav.fit(X_train, Y_train_aav)

    WAR_PA = batter_data[(batter_data['Name'] == player)]['WAR_PA'].iloc[[-1]].reset_index(drop=True)
    WAR_PA_sq = WAR_PA ** 2
    WAR = batter_data[(batter_data['Name'] == player)]['WAR_162'].iloc[[-1]].reset_index(drop=True)
    WAR_SQ = np.where(WAR > 0, WAR**2, WAR*2)
    DWAR = batter_data[(batter_data['Name'] == player)]['dWAR_162'].iloc[[-1]].reset_index(drop=True)
    Y_N1_WAR = batter_data[(batter_data['Name'] == player)]['y_n1_war'].iloc[[-1]].reset_index(drop=True)
    Y_N1_WAR_SQ = np.where(Y_N1_WAR > 0, Y_N1_WAR**2, Y_N1_WAR*2)
    Y_N2_WAR = batter_data[(batter_data['Name'] == player)]['y_n2_war'].iloc[[-1]].reset_index(drop=True)
    Y_N2_WAR_SQ = Y_N2_WAR ** 2
    WOBA = batter_data[(batter_data['Name'] == player)]['wOBA'].iloc[[-1]].reset_index(drop=True)
    Y_N2_WOBA = batter_data[(batter_data['Name'] == player)]['y_n2_wOBA'].iloc[[-1]].reset_index(drop=True)
    AVG = batter_data[(batter_data['Name'] == player)]['AVG'].iloc[[-1]].reset_index(drop=True)
    try:
        INJURY = injury_data[(injury_data['Player'] == player)]['injury_duration'].iloc[[-1]].reset_index(drop=True)
    except IndexError:
        INJURY = 0
    KPCT = batter_data[(batter_data['Name'] == player)]['Kpct'].iloc[[-1]].reset_index(drop=True)
    AGE = salary_data[(salary_data['Player'] == player)]['Age'].iloc[[-1]].reset_index(drop=True)
    QUAL = salary_data[(salary_data['Player'] == player)]['Qual'].iloc[[-1]].reset_index(drop=True)
    POSITION = salary_data[(salary_data['Player'] == player)]['Position'].iloc[[-1]].reset_index(drop=True)

    YEAR = 2020

    # POSITION = 3
    # IP = 139
    # FBv = 93.3
    # INJURY = 23
    print('Player Name: \n', str(player))
    print('Predicted NPV: \n', regr.predict([[KPCT, YEAR, WAR_SQ,
                                              Y_N1_WAR_SQ, AGE, QUAL, INJURY]]))
    print('Predicted AAV: \n', regr_aav.predict([[KPCT, YEAR, WAR_SQ,
                                                  Y_N1_WAR_SQ, AGE, QUAL, INJURY]]))
    print('Accuracy Score: ', regr_aav.score(X_train, Y_train_aav))


predict_batter_contract("Yadier Molina")  # Fringier one year guys AAV is more accurate than NPV

# All Pitchers
train_data_pitcher['pos_dummy'] = np.where(train_data_pitcher['Position'] == "SP", 1, 0)
test_data_pitcher['pos_dummy'] = np.where(test_data_pitcher['Position'] == "SP", 1, 0)
pitcher_model_data = train_data_pitcher[['NPV', 'WAR_sq', 'Age', 'Qual', 'pos_dummy', 'FBv', 'Kpct', 'y_n1_war_sq',
                                         'injury_duration', 'Name']]
pitcher_model_data['FBv'].fillna(value=pitcher_model_data['FBv'].mean(), inplace=True)
pitcher_model_data['y_n1_war_sq'].fillna(value=pitcher_model_data['y_n1_war_sq'].mean(), inplace=True)

X_train = pitcher_model_data[['WAR_sq', 'Age', 'Qual', 'pos_dummy', 'FBv', 'Kpct', 'y_n1_war_sq', 'injury_duration']]
Y_train = pitcher_model_data['NPV']

# Linear Regression
regr = linear_model.LinearRegression()

regr.fit(X_train, Y_train)
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


# prediction with sklearn
def predict_pitcher_contract(player):
    train_data_pitcher['pos_dummy'] = np.where(train_data_pitcher['Position'] == "SP", 1, 0)
    model_data = train_data_pitcher[['NPV', 'AAV', 'WAR_sq', 'Age', 'Qual', 'pos_dummy', 'FBv', 'Kpct', 'y_n1_war_sq',
                                     'injury_duration']]
    model_data['FBv'].fillna(value=model_data['FBv'].mean(), inplace=True)
    model_data['y_n1_war_sq'].fillna(value=model_data['y_n1_war_sq'].mean(), inplace=True)

    X_train = model_data[['WAR_sq', 'Age', 'Qual', 'pos_dummy', 'FBv', 'Kpct', 'y_n1_war_sq', 'injury_duration']]
    Y_train = model_data['NPV']
    regr = linear_model.LinearRegression()

    regr.fit(X_train, Y_train)
    # print('Intercept: \n', regr.intercept_)
    # print('Coefficients: \n', regr.coef_)

    Y_train_aav = model_data['AAV']
    regr_aav = linear_model.LinearRegression()

    regr_aav.fit(X_train, Y_train_aav)

    WAR = pitcher_data[(pitcher_data['Name'] == player)]['WAR_162'].iloc[[-1]].reset_index(drop=True)
    WAR_SQ = np.where(WAR > 0, WAR**2, WAR*2)
    Y_N1_WAR = pitcher_data[(pitcher_data['Name'] == player)]['y_n1_war'].iloc[[-1]].reset_index(drop=True)
    Y_N1_WAR_SQ = np.where(Y_N1_WAR > 0, Y_N1_WAR**2, Y_N1_WAR*2)
    AGE = salary_data[(salary_data['Player'] == player)]['Age'].iloc[[-1]].reset_index(drop=True)
    KPCT = pitcher_data[(pitcher_data['Name'] == player)]['Kpct'].iloc[[-1]].reset_index(drop=True)
    QUAL = salary_data[(salary_data['Player'] == player)]['Qual'].iloc[[-1]].reset_index(drop=True)
    try:
        INJURY = injury_data[(injury_data['Player'] == player)]['injury_duration'].iloc[[-1]].reset_index(drop=True)
    except IndexError:
        INJURY = 0
    FBv = pitcher_data[(pitcher_data['Name'] == player)]['FBv'].iloc[[-1]].reset_index(drop=True)
    pos_dummy = 0
    YEAR = 2020

    # POSITION = 3
    # IP = 139
    # FBv = 93.3
    # INJURY = 23
    print('Player Name: \n', str(player))
    print('Predicted NPV: \n', regr.predict([[WAR_SQ, AGE, QUAL, pos_dummy, FBv, KPCT, Y_N1_WAR_SQ, INJURY]]))
    print('Predicted AAV: \n', regr_aav.predict([[WAR_SQ, AGE, QUAL, pos_dummy, FBv, KPCT, Y_N1_WAR_SQ, INJURY]]))
    print('Accuracy Score: ', regr.score(X_train, Y_train))


predict_pitcher_contract("Liam Hendriks")


# Ridge regression -- pitcher
ridge_data = train_data_pitcher[['NPV', 'WAR_sq', 'Age', 'Qual', 'pos_dummy',
                                 'FBv', 'Kpct', 'y_n1_war_sq', 'injury_duration']]
ridge_data = ridge_data.dropna().reset_index(drop=True)
X_train = ridge_data[['WAR_sq', 'Age', 'Qual', 'pos_dummy', 'FBv', 'Kpct', 'y_n1_war_sq', 'injury_duration']]
Y_train = ridge_data['NPV']


model = Ridge(alpha=0)
cv = RepeatedKFold(n_splits=10, n_repeats=6, random_state=1)
scores = cross_val_score(model, X_train, Y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))

model.fit(X_train, Y_train)


def find_pitcher(player):
    data = test_data_pitcher[(test_data_pitcher['Player'] == player)]
    data = data[['WAR_sq', 'Age', 'Qual', 'pos_dummy', 'FBv', 'Kpct', 'y_n1_war_sq', 'injury_duration']]
    row = data.values[-1].tolist()
    yhat = model.predict([row])
    print('Predicted: %.3f' % yhat)


find_pitcher("Liam Hendriks")
