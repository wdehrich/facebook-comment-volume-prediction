from sklearn import svm
import numpy as np
import urllib
import utils as ut
import matplotlib.pyplot as plt
url_prefix = "https://raw.githubusercontent.com/wdehrich/facebook-comment-volume-prediction/master/Dataset/Dataset/"
# training data URL
url_train = url_prefix + "Training/Features_Variant_1.csv"
# testing data URLs
url_test = list()
url_test.append(url_prefix + "Testing/TestSet/Test_Case_1.csv")
url_test.append(url_prefix + "Testing/TestSet/Test_Case_2.csv")
url_test.append(url_prefix + "Testing/TestSet/Test_Case_3.csv")
url_test.append(url_prefix + "Testing/TestSet/Test_Case_4.csv")
url_test.append(url_prefix + "Testing/TestSet/Test_Case_5.csv")
url_test.append(url_prefix + "Testing/TestSet/Test_Case_6.csv")
url_test.append(url_prefix + "Testing/TestSet/Test_Case_7.csv")
url_test.append(url_prefix + "Testing/TestSet/Test_Case_8.csv")
url_test.append(url_prefix + "Testing/TestSet/Test_Case_9.csv")
url_test.append(url_prefix + "Testing/TestSet/Test_Case_10.csv")

# download training data file
raw_data = urllib.urlopen(url_train)
# load the CSV file as a numpy matrix
data_set_train = np.loadtxt(raw_data, delimiter=",", skiprows=1)
# separate the data from the target attributes
X = data_set_train[:, 0:52]
y = data_set_train[:, 53]
clf = svm.SVR(kernel='linear')
clf.fit(X,y)
num_url_test = len(url_test)
hits_at_ten = list()

for i in range(num_url_test):
    current_url_test = url_test[i]
    # download the testing data file
    raw_data = urllib.urlopen(current_url_test)
    # load the CSV file as a numpy matrix
    data_set_test = np.loadtxt(raw_data, delimiter=",")

    # Predict
    X_test = data_set_test[:, 0:52]
    y_test = data_set_test[:, 53]
    y_predicted = clf.predict(X_test)
    # Get Hits@10 measurement
    hits_at_ten_current = ut.get_hits_at_ten(y_test, y_predicted)
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
