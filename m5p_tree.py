# http://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html
# http://machinelearningmastery.com/how-to-load-data-in-python-with-scikit-learn/
# http://stackoverflow.com/questions/2864842/common-elements-comparison-between-2-lists

import numpy as np
import urllib
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/wdehrich/facebook-comment-volume-prediction/master/Dataset/Dataset/Training/Features_Variant_1.csv"
# download the file
raw_data = urllib.urlopen(url)
# load the CSV file as a numpy matrix
dataset = np.loadtxt(raw_data, delimiter=",", skiprows=1)
# separate the data from the target attributes
X = dataset[:, 0:52]
y = dataset[:, 53]

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X, y)
regr_2.fit(X, y)

url = "https://raw.githubusercontent.com/wdehrich/facebook-comment-volume-prediction/master/Dataset/Dataset/Testing/TestSet/Test_Case_1.csv"
# download the file
raw_data = urllib.urlopen(url)
# load the CSV file as a numpy matrix
dataset = np.loadtxt(raw_data, delimiter=",")
print(dataset.shape)

# Predict
X_test = dataset[:, 0:52]
y = dataset[:, 53]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

order_actual = y.argsort()
order_predicted1 = y_1.argsort()
order_predicted2 = y_2.argsort()

topten_actual = order_actual[90:]
topten_predicted1 = order_predicted1[90:]
topten_predicted2 = order_predicted2[90:]
print(topten_actual)

commonalities_predicted1 = np.intersect1d(topten_predicted1, topten_actual)
hit_at_ten_predicted1 = len(list(commonalities_predicted1))

commonalities_predicted2 = np.intersect1d(topten_predicted2, topten_actual)
hit_at_ten_predicted2 = len(list(commonalities_predicted2))

print 'hit_at_ten_predicted1 =', hit_at_ten_predicted1
print 'hit_at_ten_predicted2 =', hit_at_ten_predicted2