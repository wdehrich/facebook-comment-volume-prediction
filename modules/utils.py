import numpy as np
import csv


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
