import utils as ut
import accuracy as ac

data_set_train, data_sets_test = ut.get_data()
ac.get_accuracy(data_set_train, data_sets_test)
