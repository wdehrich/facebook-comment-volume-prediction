import neural_net as nn
import numpy as np
import utils

attr, label = utils.parse('E:/Features_Variant_1.csv')
attr_test, target = utils.parse('E:/Test_Case_1.csv', False)
mlp, scaler = nn.mlp_train(attr, label)
#print nn.mlp_test(mlp, attr_test, target, scaler)
print utils.get_hits_at_ten(target, mlp.predict(scaler.transform(attr_test)))