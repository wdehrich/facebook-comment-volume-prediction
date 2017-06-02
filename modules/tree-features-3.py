import utils as ut

# https://raw.githubusercontent.com/wdehrich/facebook-comment-volume-prediction/master/dataset-transformed/3-features/training/Features_Variant_1.csv
url_prefix = "https://raw.githubusercontent.com/wdehrich/facebook-comment-volume-prediction/master/dataset-transformed/3-features/"
# training data URL
url_train = url_prefix + "training/Features_Variant_1.csv"
# testing data URLs
url_test = list()
url_test.append(url_prefix + "testing/Test_Case_1.csv")
url_test.append(url_prefix + "testing/Test_Case_2.csv")
url_test.append(url_prefix + "testing/Test_Case_3.csv")
url_test.append(url_prefix + "testing/Test_Case_4.csv")
url_test.append(url_prefix + "testing/Test_Case_5.csv")
url_test.append(url_prefix + "testing/Test_Case_6.csv")
url_test.append(url_prefix + "testing/Test_Case_7.csv")
url_test.append(url_prefix + "testing/Test_Case_8.csv")
url_test.append(url_prefix + "testing/Test_Case_9.csv")
url_test.append(url_prefix + "testing/Test_Case_10.csv")

ut.run_get_hits_at_ten(url_train, url_test)
