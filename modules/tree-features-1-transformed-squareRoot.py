import numpy as np
import utils as ut
import accuracy as ac

data_set_train, data_sets_test = ut.get_data()

columns = [30, 53]
data_set_train_selected, data_sets_test_selected = ut.select_data(data_set_train, data_sets_test, columns)

data_set_train_selected[:, 0] = np.power(data_set_train_selected[:, 0], 1/2)

for i in range(len(data_sets_test_selected)):
    data_sets_test_selected[i][:, 0] = np.power(data_sets_test_selected[i][:, 0], 1/2)

ac.get_accuracy(data_set_train_selected, data_sets_test_selected)
