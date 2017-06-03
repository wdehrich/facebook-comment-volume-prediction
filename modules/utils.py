import numpy as np
import csv
import urllib


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

    # iterate through rows of act0ual data
    for row in fileToRead:
        attr.append(row)
    for row in attr:
        label.append(row.pop())

    return np.array(attr, dtype = np.float64), np.array(label, dtype = np.float64)


def get_urls(directory):
    url_prefix = "https://raw.githubusercontent.com/wdehrich/facebook-comment-volume-prediction/master/dataset-transformed/"
    url_prefix += directory
    # training data URL
    url_train = url_prefix + "/training/Features_Variant_1.csv"
    # testing data URLs
    url_test = list()
    url_test.append(url_prefix + "/testing/Test_Case_1.csv")
    url_test.append(url_prefix + "/testing/Test_Case_2.csv")
    url_test.append(url_prefix + "/testing/Test_Case_3.csv")
    url_test.append(url_prefix + "/testing/Test_Case_4.csv")
    url_test.append(url_prefix + "/testing/Test_Case_5.csv")
    url_test.append(url_prefix + "/testing/Test_Case_6.csv")
    url_test.append(url_prefix + "/testing/Test_Case_7.csv")
    url_test.append(url_prefix + "/testing/Test_Case_8.csv")
    url_test.append(url_prefix + "/testing/Test_Case_9.csv")
    url_test.append(url_prefix + "/testing/Test_Case_10.csv")
    return url_train, url_test


def get_data():
    url_prefix = "https://raw.githubusercontent.com/wdehrich/facebook-comment-volume-prediction/master/data/"

    url_train = url_prefix + "training_set.csv"

    urls_test = list()
    urls_test.append(url_prefix + "testing_set_1.csv")
    urls_test.append(url_prefix + "testing_set_2.csv")
    urls_test.append(url_prefix + "testing_set_3.csv")
    urls_test.append(url_prefix + "testing_set_4.csv")
    urls_test.append(url_prefix + "testing_set_5.csv")
    urls_test.append(url_prefix + "testing_set_6.csv")
    urls_test.append(url_prefix + "testing_set_7.csv")
    urls_test.append(url_prefix + "testing_set_8.csv")
    urls_test.append(url_prefix + "testing_set_9.csv")
    urls_test.append(url_prefix + "testing_set_10.csv")

    num_url_test = len(urls_test)

    # download training data file
    raw_data = urllib.urlopen(url_train)
    # load the CSV file as a numpy matrix
    data_set_train = np.loadtxt(raw_data, delimiter=",", skiprows=1)

    data_sets_test = list()
    for i in range(num_url_test):
        url_test_current = urls_test[i]
        # download the testing data file
        raw_data = urllib.urlopen(url_test_current)
        # load the CSV file as a numpy matrix
        data_set_test_current = np.loadtxt(raw_data, delimiter=",")
        data_sets_test.append(data_set_test_current)

    return data_set_train, data_sets_test


def select_data(data_set_train, data_sets_test, columns):
    data_sets_test_selected = list()

    for i in range(len(data_sets_test)):
        data_sets_test_selected.append(data_sets_test[i][:, columns])

    data_set_train_selected = data_set_train[:, columns]

    return data_set_train_selected, data_sets_test_selected
