from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error
import numpy as np
import urllib
import utils as ut
import matplotlib.pyplot as plt


def get_hits_at_ten(actual, predicted):
    order_actual = np.argsort(actual)
    order_predicted = np.argsort(predicted)
    top_ten_actual = order_actual[-10:]
    top_ten_predicted = order_predicted[-10:]
    commonalities = np.intersect1d(top_ten_predicted, top_ten_actual)
    hits_at_ten = len(commonalities)
    return hits_at_ten


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
    plt.plot(np.array(target), np.array(predicted), 'ro')
    upper = max(max(target), max(predicted)) * 1.1
    plt.plot([0, upper], [0, upper], 'b-')
    predicted[predicted == 0] = 1
    r = target * 1.0 / predicted
    # print 'Within the factor of %d: ' % factor, len(r[(r <= factor) & (r >= 1/factor)])*1.0/len(r)
    # plt.show()
    return len(r[(r <= factor) & (r >= 1 / factor)]) * 1.0 / len(r)


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


# url_train, url_test = ut.get_urls("features-1")
# url_train, url_test = ut.get_urls("features-2")
# url_train, url_test = ut.get_urls("features-3")

data_set_train, data_sets_test = ut.get_data()
#columns = [30, 53]
#data_set_train_selected, data_sets_test_selected = ut.select_data(data_set_train, data_sets_test, columns)
#data_set_train_selected[:, 0] = np.square(data_set_train_selected[:, 0])
#for i in range(len(data_sets_test_selected)):
#    data_sets_test_selected[i][:, 0] = np.square(data_sets_test_selected[i][:, 0])

# download training data file
# raw_data = urllib.urlopen(url_train)
# load the CSV file as a numpy matrix
# data_set_train = np.loadtxt(raw_data, delimiter=",", skiprows=1)
# separate the data from the target attributes
num_attributes = len(data_set_train[0]) - 1

X = data_set_train[:, 0:num_attributes]
y = data_set_train[:, num_attributes]
clf = GaussianNB()
clf.fit(X, y)
hits_at_ten = list()
mean_square_error = list()
ratios = list()
# download the testing data file
for i in range(len(data_sets_test)):
    # current_url_test = url_test[i]
    # raw_data = urllib.urlopen(current_url_test)
    # load the CSV file as a numpy matrix
    # data_set_test = np.loadtxt(raw_data, delimiter=",",skiprows=1)

    # Predict
    X_test = data_sets_test[i][:, 0:num_attributes]
    y_test = data_sets_test[i][:, num_attributes]
    y_predicted = clf.predict(X_test)
    # Get Hits@10 measurement
    hits_at_ten_current = get_hits_at_ten(y_test, y_predicted)
    hits_at_ten.append(hits_at_ten_current)

    mean_square_error_current = mse(y_test, y_predicted)
    mean_square_error.append(mean_square_error_current)

    # Get ratio
    ratio_current = ratio(y_test, y_predicted, 2)
    ratios.append(ratio_current)

    hits_at_ten_average = float(sum(hits_at_ten) / float(len(data_sets_test)))
    mean_square_error_average = float(sum(mean_square_error)) / float(len(mean_square_error))
    ratios_average = float(sum(ratios)) / float(len(ratios))
print 'hits_at_ten =', hits_at_ten
print 'hit_at_ten_average =', hits_at_ten_average
print 'mean_square_error_average = ', mean_square_error_average
print 'ratios_average = ', ratios_average

x = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Avg.']
y_pos = np.arange(len(x))
hits_at_ten.append(hits_at_ten_average)
scores = np.copy(hits_at_ten)

plt.bar(y_pos, scores, align='center', alpha=0.5)
plt.xticks(y_pos, x)
plt.ylabel('Score')
plt.xlabel('Test')
plt.title('Hits@10')
# plt.show()
