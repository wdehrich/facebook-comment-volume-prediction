import numpy as np

def get_hits_at_10(actual, predicted):
    order_actual = np.argsort(actual)
    order_predicted = np.argsort(predicted)
    top_ten_actual = order_actual[90:]
    top_ten_predicted = order_predicted[90:]
    commonalities = np.intersect1d(top_ten_predicted, top_ten_actual)
    hits_at_10 = len(commonalities)
    return hits_at_10

#get_hits_at_10([1, 2, 3], [4, 5, 6])


'''
order_actual = y.argsort()
order_predicted1 = y_1.argsort()
order_predicted2 = y_2.argsort()

topten_actual = order_actual[90:]
topten_predicted1 = order_predicted1[90:]
topten_predicted2 = order_predicted2[90:]
print(topten_actual)

commonalities_predicted1 = np.intersect1d(topten_predicted1, topten_actual)
hit_at_ten_predicted1 = len(list(commonalities_predicted1))

commonalities_predicted2 = np.intersect1d(topten_predicted2, topten_actual)
hit_at_ten_predicted2 = len(list(commonalities_predicted2))
'''