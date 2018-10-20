# As with UV_logistic_regression, delete the hash at the start of lines 2 and 3 for the assignment. 
# import os
# os.chdir('C:\\Folder_name\\Sub_folder_name')

import pandas
from sklearn.externals import joblib

lr = joblib.load('Predictor.pkl')

data = 'https://raw.githubusercontent.com/echounlucky/ML-for-SB/master/TennisDataUpcomingMatches.csv' 
# For the assignment, replace the URL with the name of the file name that you saved in the folder above, e.g. 'NBAData.csv' 
dataset = pandas.read_csv(data, header=None)

probabilities = lr.predict_proba(dataset)
prediction = lr.predict(dataset)

print(prediction)
print(probabilities)
