# For the assignment, delete the hash at the start of lines 4 and 5. 
# Running the code in these lines tells pyton to change the home directory for this file.
# Change the directory name in line 5 to the folder where you have saved the [NBA] data for the assignment. 
# import os
# os.chdir('C:\\Folder_name\\Sub_folder_name') 

import pandas
from sklearn import linear_model
from sklearn.externals import joblib

data = "https://raw.githubusercontent.com/echounlucky/ML-for-SB/master/TennisData.csv" 
# For the assignment, change the file name to the csv file you have saved in the directory above, e.g. 'NBAData.csv'
dataset = pandas.read_csv(data, header=None)

array = dataset.values
X = array[:,0:-1]
Y = array[:,-1]
X_train, Y_train = X, Y

lr = linear_model.LogisticRegression()
lr.fit(X_train, Y_train)

joblib.dump(lr, 'Predictor.pkl')
