import csv

def parse(filename):
    '''
    takes a filename and returns list of features(list) and its labels(list), assuming
    labels are at the last column.
    '''
    # initialize variables
    label = []
    attr = [] 
    csvfile = open(filename,'rb')
    fileToRead = csv.reader(csvfile)
    headers = fileToRead.next()

    # iterate through rows of actual data
    for row in fileToRead:
        attr.append(row)
    for row in attr:
    	label.append(row.pop())

    return attr,label

def parse_tester():
	attr = []
	label = []
	attr, label = parse('E:/facebook_datasets.csv')
	print attr[0]

parse_tester()