import numpy as np
import pandas
from readData import readData
from readData_cv import readData_cv
from readData_cv_nogroup import readData_cv_ng
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from utils import calculateR2
from utils import huber
from utils import logcosh
from sklearn.model_selection._search import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import shap
np.random.seed(80)
shap.initjs()

#Read excel data
K = 10
X_folds, y_folds = readData_cv(K)
X_test = X_folds[K-1]
y_test = y_folds[K-1]

rmse_err = np.zeros((3, K-1))
R2_err = np.zeros((3, K-1))
mae_err = np.zeros((3, K-1))
huber_err = np.zeros((3, K-1))
logcosh_err = np.zeros((3, K-1))
for i in range(K-1):
    #Generate training set and validation set
    X_valid = X_folds[i]
    y_valid = y_folds[i]
    X_train = np.zeros((0, 10))
    y_train = np.zeros((0,))
    for j in range(K-1):
        if(j != i):
            X_train = np.append(X_train, X_folds[j], axis=0)
            y_train = np.append(y_train, y_folds[j], axis=0)

    regr = GradientBoostingRegressor(n_estimators=100, max_depth=4)

    regr.fit(X_train, y_train)

    y_pred_train = regr.predict(X_train)
    y_pred_valid = regr.predict(X_valid)
    y_pred_test = regr.predict(X_test)

    rmse_err[0][i] = np.sqrt(np.sum((y_pred_train - y_train) ** 2) / y_pred_train.size)
    rmse_err[1][i] = np.sqrt(np.sum((y_pred_valid - y_valid) ** 2) / y_pred_valid.size)
    rmse_err[2][i] = np.sqrt(np.sum((y_pred_test - y_test) ** 2) / y_pred_test.size)
    R2_err[0][i] = calculateR2(y_train, y_pred_train)
    R2_err[1][i] = calculateR2(y_valid, y_pred_valid)
    R2_err[2][i] = calculateR2(y_test, y_pred_test)
    mae_err[0][i] = np.sum(abs(y_pred_train - y_train)) / y_pred_train.size
    mae_err[1][i] = np.sum(abs(y_pred_valid - y_valid)) / y_pred_valid.size
    mae_err[2][i] = np.sum(abs(y_pred_test - y_test)) / y_pred_test.size
    huber_err[0][i] = huber(y_train, y_pred_train, 0.1)/ y_pred_train.size
    huber_err[1][i] = huber(y_valid, y_pred_valid, 0.1)/ y_pred_valid.size
    huber_err[2][i] = huber(y_test, y_pred_test, 0.1)/ y_pred_test.size
    logcosh_err[0][i] = logcosh(y_train, y_pred_train)/ y_pred_train.size
    logcosh_err[1][i] = logcosh(y_valid, y_pred_valid)/ y_pred_valid.size
    logcosh_err[2][i] = logcosh(y_test, y_pred_test)/ y_pred_test.size

    print("Train RMSE: ", rmse_err[0][i], "Valid RMSE: ", rmse_err[1][i], "Test RMSE: ", rmse_err[2][i])
    print("Train R2: ", R2_err[0][i], "Valid R2: ", R2_err[1][i], "Test R2: ", R2_err[2][i])
    print("Train mae: ", mae_err[0][i], "Valid mae: ", mae_err[1][i], "Test mae: ", mae_err[2][i])
    print("Train huber: ", huber_err[0][i], "Valid huber: ", huber_err[1][i], "Test huber: ", huber_err[2][i])
    print("Train logcosh: ", logcosh_err[0][i], "Valid logcosh: ", logcosh_err[1][i], "Test logcosh: ",
          logcosh_err[2][i])
print("")
print("Summary:")
print("    Train RMSE: ", np.mean(rmse_err[0]),  "Test RMSE: ", np.mean(rmse_err[2]))
print("    Train R2: ", np.mean(R2_err[0]),  "Test R2: ", np.mean(R2_err[2]))
print("    Train mae: ", np.mean(mae_err[0]),"Test mae: ", np.mean(mae_err[2]))
print("    Train huber: ", np.mean(huber_err[0]),  "Test huber: ", np.mean(huber_err[2]))
print("    Train logcosh: ", np.mean(logcosh_err[0]),  "Test logcosh: ", np.mean(logcosh_err[2]))

'''''
y_pred_train = regr.predict(X_train)
y_pred_valid = regr.predict(X_valid)
y_pred_test = regr.predict(X_test)

#predict train, valid, test
sheet_data_train = np.zeros((1686, 10))
sheet_data_train[:, 0:8] = X_train[:,0:8]
sheet_data_train[:,8] = y_pred_train[0:None]
sheet_data_train[:,9] = y_train[0:None]
np.savetxt("GBDT_pred_train.csv", sheet_data_train, delimiter=",")

sheet_data_valid = np.zeros((212, 10))
sheet_data_valid[:, 0:8] = X_valid[:,0:8]
sheet_data_valid[:,8] = y_pred_valid[0:None]
sheet_data_valid[:,9] = y_valid[0:None]
np.savetxt("GBDT_pred_valid.csv", sheet_data_valid, delimiter=",")

sheet_data_test = np.zeros((300, 10))
sheet_data_test[:, 0:8] = X_test[:,0:8]
sheet_data_test[:,8] = y_pred_test[0:None]
sheet_data_test[:,9] = y_test[0:None]
np.savetxt("GBDT_pred_test.csv", sheet_data_test, delimiter=",")
'''