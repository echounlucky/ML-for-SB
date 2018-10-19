import os
os.chdir('C:\\Users\\k_mun\\Documents\\Tennis\\Machine learning') # Change directory to where you have saved your data

import pandas
from sklearn import linear_model
from sklearn.externals import joblib

data = 'TennisData3.csv' # Change the file name to the csv file you have saved in the directory above
dataset = pandas.read_csv(data, header=None)

array = dataset.values
X = array[:,0:-1]
Y = array[:,-1]
X_train, Y_train = X, Y

lr = linear_model.LogisticRegression()
lr.fit(X_train, Y_train)

joblib.dump(lr, 'Predictor.pkl')
