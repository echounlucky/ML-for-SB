# As with UV_logistic_regression, delete the hash at the start of lines 2 and 3 for the assignment. 
# import os
# os.chdir('C:\\Folder_name\\Sub_folder_name') 

import pandas
from sklearn import linear_model
from sklearn.externals import joblib

data = "https://raw.githubusercontent.com/echounlucky/ML-for-SB/master/TennisData.csv" 
# For the assignment, change the URL to the csv file you have saved in the home directory set up above, e.g. 'NBAData.csv'
dataset = pandas.read_csv(data, header=None)

array = dataset.values
X = array[:,0:-1]
Y = array[:,-1]
X_train, Y_train = X, Y

lr = linear_model.LogisticRegression()
lr.fit(X_train, Y_train)

joblib.dump(lr, 'Predictor.pkl')
