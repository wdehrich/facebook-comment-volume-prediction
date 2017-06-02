import utils as ut
import accuracy as ac

url_train, url_test = ut.get_urls("features-2")

ac.run_get_hits_at_ten(url_train, url_test)
