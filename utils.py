import numpy as np

# requires an input array of length 100
def get_hits_at_10(actual, predicted):
    order_actual = np.argsort(actual)
    order_predicted = np.argsort(predicted)
    top_ten_actual = order_actual[90:]
    top_ten_predicted = order_predicted[90:]
    commonalities = np.intersect1d(top_ten_predicted, top_ten_actual)
    hits_at_10 = len(commonalities)
    return hits_at_10