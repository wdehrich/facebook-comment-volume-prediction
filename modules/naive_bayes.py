from sklearn.naive_bayes import GaussianNB
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

# url_train, url_test = ut.get_urls("features-1")
# url_train, url_test = ut.get_urls("features-2")
url_train, url_test = ut.get_urls("features-3")
# download training data file
raw_data = urllib.urlopen(url_train)
# load the CSV file as a numpy matrix
data_set_train = np.loadtxt(raw_data, delimiter=",", skiprows=1)
# separate the data from the target attributes
num_attributes = len(data_set_train[0]) - 1
X = data_set_train[:, 0:(num_attributes - 1)]
y = data_set_train[:, num_attributes]
clf = GaussianNB()
clf.fit(X,y)
hits_at_ten = list()
num_url_test = len(url_test)
# download the testing data file
for i in range(num_url_test):
	current_url_test = url_test[i]
	raw_data = urllib.urlopen(current_url_test)
	# load the CSV file as a numpy matrix
	data_set_test = np.loadtxt(raw_data, delimiter=",",skiprows=1)

	# Predict
	X_test = data_set_test[:, 0:(num_attributes - 1)]
	y_test = data_set_test[:, num_attributes]
	y_predicted = clf.predict(X_test)
	# Get Hits@10 measurement
	hits_at_ten_current = get_hits_at_ten(y_test, y_predicted)
	hits_at_ten.append(hits_at_ten_current)

	hits_at_ten_average = float(sum(hits_at_ten)/float(num_url_test))
print 'hits_at_ten =', hits_at_ten
print 'hit_at_ten_average =', hits_at_ten_average

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
