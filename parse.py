import csv
import numpy as np

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

def parse_tester():
	attr, label = parse('E:/Test_Case_1.csv', False)
	print attr

#parse_tester()