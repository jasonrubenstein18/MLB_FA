import numpy as np
from numpy import loadtxt
from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import graphviz


train_data_pitcher = pd.read_csv("~/Desktop/MLB_FA/Data/train_data_pitcher.csv")
train_data_pitcher['pos_dummy'] = np.where(train_data_pitcher['Position'] == "SP", 1, 0)
train_data_pitcher['WAR_162_sq'] = train_data_pitcher['WAR_162']**2
test_data_pitcher = pd.read_csv("~/Desktop/MLB_FA/Data/test_data_pitcher.csv")
test_data_pitcher['pos_dummy'] = np.where(test_data_pitcher['Position'] == "SP", 1, 0)
test_data_pitcher['WAR_162_sq'] = test_data_pitcher['WAR_162']**2
pitcher_data = pd.read_csv("~/Desktop/MLB_FA/Data/fg_pitch_data.csv")

train_data_batter = pd.read_csv("~/Desktop/MLB_FA/Data/train_data_batter.csv")
test_data_batter = pd.read_csv("~/Desktop/MLB_FA/Data/test_data_batter.csv")
batter_data = pd.read_csv("~/Desktop/MLB_FA/Data/fg_bat_data.csv")


# Pitchers
# split data into X and Y
model_data = train_data_pitcher[['NPV', 'WAR_162_sq', 'Age', 'Qual', 'pos_dummy', 'FBv', 'Kpct', 'y_n1_war_sq']]
X = model_data[['WAR_162_sq', 'Age', 'Qual', 'pos_dummy', 'FBv', 'Kpct', 'y_n1_war_sq']]
Y = model_data[['NPV']]

# Matrix creation
X, Y = model_data.iloc[:,model_data.columns != "NPV"],model_data.iloc[:,0]
data_dmatrix = xgb.DMatrix(data=X,label=Y)


# split data into train and test sets
seed = 7
test_size = 0.5
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# Create model
xg_reg = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1,
                          max_depth=5, alpha=10, n_estimators=10)
xg_reg.fit(X_train,Y_train)
preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(Y_test, preds))
print("RMSE: %f" % rmse)

# Cross validate, to reduce RMSE
params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
          'max_depth': 5, 'alpha': 10}

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50, early_stopping_rounds=10, metrics="rmse", as_pandas=True, seed=123)
cv_results.head()
print((cv_results["test-rmse-mean"]).tail(1))

# Visualize boosters and importance
xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)

# xgb.plot_tree(xg_reg,num_trees=0)
# plt.rcParams['figure.figsize'] = [100, 20]
# plt.show()
# plt.close()

xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()
