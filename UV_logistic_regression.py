import os
os.chdir('C:\\ML_for_SB\\SportsData') # Change directory to where you have saved your data

import pandas
import sklearn
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

data = 'TennisData.csv' # Change the file name to the csv file you have saved in the directory above
dataset = pandas.read_csv(data, header=None)

array = dataset.values
X = array[:,0:-1] # X becomes the whole dataset except the rightmost column
Y = array[:,-1] # Y is the target variable, which must be in the rightmost column
seed = 7
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.20, random_state=seed)

lr = linear_model.LogisticRegression()
lr.fit(X_train, Y_train)
prob_pos_lr = lr.predict_proba(X_test)
prediction = lr.predict(X_test)

print("Accuracy score: %1.3f" % accuracy_score(Y_test, prediction), '\n')
print("Confusion matrix:" '\n', confusion_matrix(Y_test, prediction), '\n')

for i in range(10):
    if i == 0:
        print("Prediction:" '\t' "Result:")
    print(prediction[i], '\t\t', Y_test[i])

