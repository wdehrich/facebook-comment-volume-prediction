import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import urllib
from sklearn.tree import DecisionTreeRegressor


# requires an input array of at least size 10
def get_hits_at_ten(actual, predicted):
    order_actual = np.argsort(actual)
    order_predicted = np.argsort(predicted)
    top_ten_actual = order_actual[-10:]
    top_ten_predicted = order_predicted[-10:]
    commonalities = np.intersect1d(top_ten_predicted, top_ten_actual)
    hits_at_ten = len(commonalities)
    return hits_at_ten

# Evaluation of mean square error on the classifier
def mse(target, predicted):
	"""
	compute the mean square error of the classifier
	-inputs:
		target: the actual class labels
		predicted: prediction from the classifier given the attributes
	-output:
		mse: the mean square error
	"""
	return mean_squared_error(target, predicted)

# actual-predicted ratio plot
def ratio(target, predicted, factor):
    """
    plot the actual vs predicted value, and compute the percentage within the factor
    -inputs:
        target: the actual class labels
        predicted: prediction from the classifier given the attributes
        factor: specify the number of within factor
    -output:
        a plot of predicted against target values, with ideal line
        percentage of values within the factor
    """
    plt.plot(target,predicted,'ro')
    upper = max(max(target),max(predicted)) * 1.1
    plt.plot([0, upper],[0, upper],'b-')
    predicted[predicted == 0] = 1
    r = target*1.0/predicted
    print 'Within the factor of %d: ' % factor, len(r[(r <= factor) & (r >= 1/factor)])*1.0/len(r)
    plt.show()


# test the functions
def tester():
	y_true = np.array([3.0, 1, 2, 7])
	y_pred = np.array([4, 0.0, 2, 8])
	ratio(y_true, y_pred)

#tester()


def run_get_hits_at_ten(url_train, url_test):
    # download training data file
    raw_data = urllib.urlopen(url_train)
    # load the CSV file as a numpy matrix
    data_set_train = np.loadtxt(raw_data, delimiter=",", skiprows=1)
    # separate the data from the target attributes
    num_attributes = len(data_set_train[0]) - 1
    print(num_attributes)
    X = data_set_train[:, 0:(num_attributes - 1)]
    y = data_set_train[:, num_attributes]

    # Fit regression model
    regressor = DecisionTreeRegressor(max_depth=5)
    regressor.fit(X, y)

    num_url_test = len(url_test)
    hits_at_ten = list()

    for i in range(num_url_test):
        current_url_test = url_test[i]
        # download the testing data file
        raw_data = urllib.urlopen(current_url_test)
        # load the CSV file as a numpy matrix
        data_set_test = np.loadtxt(raw_data, delimiter=",")

        # Predict
        X_test = data_set_test[:, 0:(num_attributes - 1)]
        y = data_set_test[:, num_attributes]
        y_predicted = regressor.predict(X_test)

        # Get Hits@10 measurement
        hits_at_ten_current = get_hits_at_ten(y, y_predicted)
        hits_at_ten.append(hits_at_ten_current)

    hits_at_ten_average = float(sum(hits_at_ten) / float(num_url_test))
    print 'hits_at_ten =', hits_at_ten
    print 'hit_at_ten_average =', hits_at_ten_average

    importances = regressor.feature_importances_
    print(importances)
    importances_indexes = np.argsort(importances)
    print(importances_indexes)
    print(np.argmax(importances))

    x = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Avg.']
    y_pos = np.arange(len(x))
    hits_at_ten.append(hits_at_ten_average)
    scores = np.copy(hits_at_ten)

    plt.bar(y_pos, scores, align='center', alpha=0.5)
    plt.xticks(y_pos, x)
    plt.ylabel('Score')
    plt.xlabel('Test')
    plt.title('Hits@10')
    plt.show()


def get_accuracy(data_set_train, data_sets_test):
    print("Executing function get_accuracy")
    num_attributes = len(data_set_train[0]) - 1
    print(num_attributes)

    # separate the data from the target attributes
    X = data_set_train[:, 0:num_attributes]
    #X = data_set_train[:, :-1]
    y = data_set_train[:, num_attributes]

    # Fit regression model
    regressor = DecisionTreeRegressor(max_depth=5)
    regressor.fit(X, y)
    print("created tree")

    hits_at_ten = list()
    for i in range(len(data_sets_test)):
        # Predict
        X_test = data_sets_test[i][:, 0:num_attributes]
        y = data_sets_test[i][:, num_attributes]
        y_predicted = regressor.predict(X_test)

        # Get Hits@10 measurement
        hits_at_ten_current = get_hits_at_ten(y, y_predicted)
        hits_at_ten.append(hits_at_ten_current)

    hits_at_ten_average = float(sum(hits_at_ten) / float(len(data_sets_test)))
    print 'hits_at_ten =', hits_at_ten
    print 'hit_at_ten_average =', hits_at_ten_average
    return