import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.grid_search import GridSearchCV

def add_dummies(indata):
    dummies = pd.get_dummies(indata['Depth'])
    outdata = pd.concat([indata, dummies], axis=1)
    outdata.drop(['PIDN','Depth'], axis=1, inplace = True)
    return outdata

def cv_optimize(X_train, Y_train):
    clf = svm.SVR(kernel = 'linear')
    list_C = np.logspace(-4, 3, num=8)
    #list_epsilon = np.logspace(-7, 0, num=8)
    list_epsilon = [0.1]

    parameters = {"C": list_C, "epsilon": list_epsilon}
    gs = GridSearchCV(clf, param_grid = parameters, cv = 10, n_jobs = 8)
    gs.fit(X_train, Y_train)
    print "gs.best_params_ = {0}, gs.best_score_ = {1}".format(gs.best_params_, gs.best_score_)
    return gs.best_params_, gs.best_score_

def fit_clf(X_train, Y_train):
    bp, bs = cv_optimize(X_train, Y_train)
    clf = svm.SVR(kernel = 'linear', C = bp['C'], epsilon = bp['epsilon'])
    clf.fit(X_train, Y_train)
    return clf

if __name__ == '__main__':

    df_train = pd.read_csv("..\\..\\..\\data\\training.csv")
    df_train = add_dummies(df_train)
    df_test = pd.read_csv("..\\..\\..\\data\\sorted_test.csv")
    df_output = pd.DataFrame(df_test[['PIDN']])
    df_test = add_dummies(df_test)

    yCols = ['Ca', 'P', 'pH', 'SOC', 'Sand']
    df_trainX = df_train.copy()
    for colName in yCols:
        del df_trainX[colName]
    X_train = df_trainX.values



    for colName in yCols:
        df_trainY = df_train[colName]
        Y_train = df_trainY.values

        X_train = X_train[0:100, :]
        Y_train = Y_train[0:100]

        clf = fit_clf(X_train, Y_train)
        print clf.score(X_train, Y_train)
        predicted_test = clf.predict(df_test.values)
        df_output[colName] = predicted_test

    df_output.to_csv("..\\..\\..\\data\\output.csv", index = False)








