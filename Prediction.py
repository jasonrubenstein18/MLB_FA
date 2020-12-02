import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np

# Read Data
train_data_batter = pd.read_csv("~/Desktop/MLB_FA/Data/train_data_batter.csv")
train_data_pitcher = pd.read_csv("~/Desktop/MLB_FA/Data/train_data_pitcher.csv")
salary_data = pd.read_csv("~/Desktop/MLB_FA/Data/salary_data.csv")
injury_data = pd.read_csv("~/Desktop/MLB_FA/Data/injury_data_use.csv")

# Batters
model_data = train_data_batter[['NPV', 'Kpct', 'wOBA', 'Year', 'WAR_sq', 'y_n1_war',
                                'y_n2_war', 'y_n1_war_sq', 'Position', 'dWAR_162',
                                'y_n2_war_sq', 'Age', 'Qual', 'injury_duration']]  # .join(pos_dummies)

print(len(model_data))
model_data['injury_duration'] = model_data['injury_duration'].fillna(0)
model_data = model_data.dropna()
print(len(model_data))

X_train = model_data[['Kpct', 'wOBA', 'Year', 'WAR_sq', 'y_n1_war',
                      'y_n2_war', 'y_n1_war_sq', 'dWAR_162',
                      # '1B', '2B', '3B', 'SS', 'C', 'Corner Outfield', 'CF', 'DH',
                      'y_n2_war_sq', 'Age', 'Qual', 'injury_duration']]
Y_train = model_data['NPV']
regr = linear_model.LinearRegression()

regr.fit(X_train, Y_train)
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


# prediction with sklearn
def predict_batter_contract(player):
    WAR_PA = batter_data[(batter_data['Name'] == player)]['WAR_PA'].iloc[[-1]].reset_index(drop=True)
    WAR_PA_sq = WAR_PA ** 2
    WAR = batter_data[(batter_data['Name'] == player)]['WAR_162'].iloc[[-1]].reset_index(drop=True)
    WAR_SQ = WAR ** 2
    DWAR = batter_data[(batter_data['Name'] == player)]['dWAR_162'].iloc[[-1]].reset_index(drop=True)
    Y_N1_WAR = batter_data[(batter_data['Name'] == player)]['y_n1_war'].iloc[[-1]].reset_index(drop=True)
    Y_N1_WAR_SQ = Y_N1_WAR ** 2
    Y_N2_WAR = batter_data[(batter_data['Name'] == player)]['y_n2_war'].iloc[[-1]].reset_index(drop=True)
    Y_N2_WAR_SQ = Y_N2_WAR ** 2
    WOBA = batter_data[(batter_data['Name'] == player)]['wOBA'].iloc[[-1]].reset_index(drop=True)
    Y_N2_WOBA = batter_data[(batter_data['Name'] == player)]['y_n2_wOBA'].iloc[[-1]].reset_index(drop=True)
    AVG = batter_data[(batter_data['Name'] == player)]['AVG'].iloc[[-1]].reset_index(drop=True)
    AGE = salary_data[(salary_data['Player'] == player)]['Age'].iloc[[-1]].reset_index(drop=True)
    KPCT = batter_data[(batter_data['Name'] == player)]['Kpct'].iloc[[-1]].reset_index(drop=True)
    QUAL = salary_data[(salary_data['Player'] == player)]['Qual'].iloc[[-1]].reset_index(drop=True)
    INJURY = injury_data[(injury_data['Player'] == player)]['injury_duration'].iloc[[-1]].reset_index(drop=True)
    POSITION = salary_data[(salary_data['Player'] == player)]['Position'].iloc[[-1]].reset_index(drop=True)
    YEAR = 2020

    # POSITION = 3
    # IP = 139
    # FBv = 93.3
    # INJURY = 23
    print('Predicted NPV: \n', regr.predict([[KPCT, WOBA, YEAR, WAR_SQ, Y_N1_WAR,
                                              Y_N2_WAR, Y_N1_WAR_SQ, DWAR,
                                              Y_N2_WAR_SQ, AGE, QUAL, INJURY]]))
    print('Accuracy Score: ', regr.score(X_train, Y_train))


predict_batter_contract("George Springer")


# with statsmodels
X = sm.add_constant(X_train)  # adding a constant

model = sm.OLS(Y_train, X_train).fit()
predictions = model.predict(X_train)

print_model = model.summary()
print(print_model)


# Pitchers
train_data_pitcher['pos_dummy'] = np.where(train_data_pitcher['Position'] == "SP", 1, 0)
# fit = ols('NPV ~ WAR_sq + Age + Qual + pos_dummy + FBv + Kpct + y_n1_war_sq', data=train_data_pitcher).fit()
# fit.summary()

model_data = train_data_pitcher[['NPV', 'WAR_sq', 'Age', 'Qual', 'pos_dummy', 'FBv', 'Kpct', 'y_n1_war_sq']]
model_data['FBv'].fillna(value=model_data['FBv'].mean(), inplace=True)
# model_data['injury_duration'].fillna(0, inplace=True)
model_data['y_n1_war_sq'].fillna(value=model_data['y_n1_war_sq'].mean(), inplace=True)

X_train = model_data[['WAR_sq', 'Age', 'Qual', 'pos_dummy', 'FBv', 'Kpct', 'y_n1_war_sq']]
Y_train = model_data['NPV']
regr = linear_model.LinearRegression()

regr.fit(X_train, Y_train)
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


# prediction with sklearn
def predict_pitcher_contract(player):
    WAR = pitcher_data[(pitcher_data['Name'] == player)]['WAR_162'].iloc[[-1]].reset_index(drop=True)
    WAR_SQ = WAR ** 2
    Y_N1_WAR = pitcher_data[(pitcher_data['Name'] == player)]['y_n1_war'].iloc[[-1]].reset_index(drop=True)
    Y_N1_WAR_SQ = Y_N1_WAR ** 2
    AGE = salary_data[(salary_data['Player'] == player)]['Age'].iloc[[-1]].reset_index(drop=True)
    KPCT = pitcher_data[(pitcher_data['Name'] == player)]['Kpct'].iloc[[-1]].reset_index(drop=True)
    QUAL = salary_data[(salary_data['Player'] == player)]['Qual'].iloc[[-1]].reset_index(drop=True)
    INJURY = injury_data[(injury_data['Player'] == player)]['injury_duration'].iloc[[-1]].reset_index(drop=True)
    FBv = pitcher_data[(pitcher_data['Name'] == player)]['FBv'].iloc[[-1]].reset_index(drop=True)
    pos_dummy = 0
    YEAR = 2020

    # POSITION = 3
    # IP = 139
    # FBv = 93.3
    # INJURY = 23
    print('Predicted NPV: \n', regr.predict([[WAR_SQ, AGE, QUAL, pos_dummy, FBv, KPCT, Y_N1_WAR_SQ]]))
    print('Accuracy Score: ', regr.score(X_train, Y_train))


predict_pitcher_contract("Trevor May")


# with statsmodels
X = sm.add_constant(X_train)  # adding a constant

model = sm.OLS(Y_train, X_train).fit()
predictions = model.predict(X_train)

print_model = model.summary()
print(print_model)
