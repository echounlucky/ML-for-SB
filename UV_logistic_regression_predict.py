import os
os.chdir('C:\\Users\\k_mun\\Documents\\Tennis\\Machine learning') # Change directory to where you have saved your data

import pandas
from sklearn.externals import joblib

lr = joblib.load('Predictor.pkl')

data = 'UpcomingMatches.csv' # Save data on about 10 matches under this file name in the directory above
dataset = pandas.read_csv(data, header=None)

probabilities = lr.predict_proba(dataset)
prediction = lr.predict(dataset)

print(prediction)
print(probabilities)
