import numpy as np
import csv
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


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

# read csv files
def parse(filename, header = True):
    '''
    takes a filename and returns list of features(list) and its labels(list), assuming
    labels are at the last column. If header is true (as default), it skips the header.
    '''
    # initialize variables
    label = []
    attr = [] 
    csvfile = open(filename,'rb')
    fileToRead = csv.reader(csvfile)
    if header:
    	headers = fileToRead.next()

    # iterate through rows of actual data
    for row in fileToRead:
        attr.append(row)
    for row in attr:
    	label.append(row.pop())

    return np.array(attr, dtype = np.float64), np.array(label, dtype = np.float64)

# test the functions
def tester():
	y_true = np.array([3.0, 1, 2, 7])
	y_pred = np.array([4, 0.0, 2, 8])
	ratio(y_true, y_pred)

#tester()