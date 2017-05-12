# http://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html
# http://machinelearningmastery.com/how-to-load-data-in-python-with-scikit-learn/

import numpy as np
import urllib
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/wdehrich/facebook-comment-volume-prediction/master/Dataset/Dataset/Training/Features_Variant_1.csv"
# download the file
raw_data = urllib.urlopen(url)
# load the CSV file as a numpy matrix
dataset = np.loadtxt(raw_data, delimiter=",", skiprows=1)
print(dataset.shape)
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

print(y)
print(y_1)
print(y_2)

'''
# Plot the results
plt.figure()
plt.scatter(X, y, c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
'''