import neural_net as nn
import numpy as np
import parse

attr, label = parse.parse('E:/Features_Variant_1.csv')
attr_test, target = parse.parse('E:/Test_Case_1.csv', False)
mlp, scaler = nn.mlp_train(attr, label)
print nn.mlp_test(mlp, attr_test, target, scaler)