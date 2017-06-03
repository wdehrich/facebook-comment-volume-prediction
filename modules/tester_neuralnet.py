import matplotlib.pyplot as plt
import neural_net as nn
import numpy as np
import accuracy as acc
import utils

# get data
attr, label = utils.parse('E:/Features_Variant_1.csv')
attr_test, target = utils.parse('E:/Test_Case_all.csv', True)

# train
mlp, scaler = nn.mlp_train(attr, label)

# mean square error
print 'mean square error: ', acc.mse(target,mlp.predict(scaler.transform(attr_test)))

# actual-perdicted ratio
#acc.ratio(target, mlp.predict(scaler.transform(attr_test)), 2)

# hits@10
hit_10 = []
for i in np.arange(0,1000,100):
	hit_10.append(acc.get_hits_at_ten(target[i:i+100],mlp.predict(scaler.transform(attr_test[i:i+100]))))
print 'Hit@10 average: ', np.mean(hit_10)
hit_10.append(np.mean(hit_10))

x = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Avg.']
plt.bar(np.arange(len(hit_10)), hit_10, align='center', alpha=0.5)
plt.xticks(np.arange(len(hit_10)), x)
axes = plt.gca()
axes.set_ylim([0,10])
plt.ylabel('Score')
plt.xlabel('Test')
plt.title('Hits@10')
plt.show()