# http://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html
# http://machinelearningmastery.com/how-to-load-data-in-python-with-scikit-learn/
# http://stackoverflow.com/questions/2864842/common-elements-comparison-between-2-lists

import numpy as np
import urllib
from sklearn.tree import DecisionTreeRegressor
import utils as ut
# import matplotlib.pyplot as plt

# import training data
url_train = "https://raw.githubusercontent.com/wdehrich/facebook-comment-volume-prediction/master/Dataset/Dataset/Training/Features_Variant_1.csv"
# download the file
raw_data = urllib.urlopen(url_train)
# load the CSV file as a numpy matrix
data_set_train = np.loadtxt(raw_data, delimiter=",", skiprows=1)
# separate the data from the target attributes
X = data_set_train[:, 0:52]
y = data_set_train[:, 53]

# Fit regression model
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_2.fit(X, y)

url_test = list()
url_test.append("https://raw.githubusercontent.com/wdehrich/facebook-comment-volume-prediction/master/Dataset/Dataset/Testing/TestSet/Test_Case_1.csv")
url_test.append("https://raw.githubusercontent.com/wdehrich/facebook-comment-volume-prediction/master/Dataset/Dataset/Testing/TestSet/Test_Case_2.csv")
url_test.append("https://raw.githubusercontent.com/wdehrich/facebook-comment-volume-prediction/master/Dataset/Dataset/Testing/TestSet/Test_Case_3.csv")
url_test.append("https://raw.githubusercontent.com/wdehrich/facebook-comment-volume-prediction/master/Dataset/Dataset/Testing/TestSet/Test_Case_4.csv")
url_test.append("https://raw.githubusercontent.com/wdehrich/facebook-comment-volume-prediction/master/Dataset/Dataset/Testing/TestSet/Test_Case_5.csv")
url_test.append("https://raw.githubusercontent.com/wdehrich/facebook-comment-volume-prediction/master/Dataset/Dataset/Testing/TestSet/Test_Case_6.csv")
url_test.append("https://raw.githubusercontent.com/wdehrich/facebook-comment-volume-prediction/master/Dataset/Dataset/Testing/TestSet/Test_Case_7.csv")
url_test.append("https://raw.githubusercontent.com/wdehrich/facebook-comment-volume-prediction/master/Dataset/Dataset/Testing/TestSet/Test_Case_8.csv")
url_test.append("https://raw.githubusercontent.com/wdehrich/facebook-comment-volume-prediction/master/Dataset/Dataset/Testing/TestSet/Test_Case_9.csv")
url_test.append("https://raw.githubusercontent.com/wdehrich/facebook-comment-volume-prediction/master/Dataset/Dataset/Testing/TestSet/Test_Case_10.csv")

for i in range(len(url_test)):
    current_url_test = url_test[i]
    # download the file
    raw_data = urllib.urlopen(current_url_test)
    # load the CSV file as a numpy matrix
    data_set_test = np.loadtxt(raw_data, delimiter=",")

    # Predict
    X_test = data_set_test[:, 0:52]
    y = data_set_test[:, 53]
    y_2 = regr_2.predict(X_test)

    hit_at_ten_predicted2 = ut.get_hits_at_10(y, y_2)
    print 'hit_at_ten_predicted2 =', hit_at_ten_predicted2
