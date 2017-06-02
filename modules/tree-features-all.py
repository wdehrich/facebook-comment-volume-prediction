import utils as ut

url_train, url_test = ut.get_urls("features-all")

ut.run_get_hits_at_ten(url_train, url_test)
