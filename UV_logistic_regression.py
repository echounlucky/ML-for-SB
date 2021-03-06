# For the assignment, delete the hash at the start of lines 4 and 5. 
# Running the code in these lines tells pyton to change the home directory for this file.
# Change the directory name in line 5 to the folder where you have saved the [NBA] data for the assignment. 
# import os
# os.chdir('C:\\Folder_name\\Sub_folder_name')

import pandas
import sklearn
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

data = 'https://raw.githubusercontent.com/echounlucky/ML-for-SB/master/TennisData.csv' 
# For the assignment, replace the URL with the name of the file that you saved in the folder above, e.g. 'NBAData.csv' 
dataset = pandas.read_csv(data, header=None)

array = dataset.values
X = array[:,0:-1] # X becomes the whole dataset except the rightmost column
Y = array[:,-1] # Y is the target variable, which is in the rightmost column
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
        print("Prediction:" '\t' "P0 prob:" '\t' "P1 prob:" '\t' "Result:")
    print(prediction[-i], '\t\t', "%1.3f" % prob_pos_lr[-i, 0], '\t\t', "%1.3f" % prob_pos_lr[-i, 1], '\t\t', Y_test[-i])
