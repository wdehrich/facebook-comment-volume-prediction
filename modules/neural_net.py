from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler 
import numpy as np

# trainer of Multi-Layer Perceptron (MLP)

def mlp_train(x, y, k = 10):
	"""
	training a Multi-Layer Perceptron (MLP) using k-fold cross validation
	- inputs:
		x: set of features/attributes(array size: (n_samples,n_features))
		y: target values/class labels(array size: (n_samples,))
		k: number of folds of cross validation (10 as default)
	- outputs:
		classifier: the MLP classifier (object)
		scaler: the scaling of training data
	"""
	# scaling the features
	scaler = StandardScaler()
	scaler.fit(x)
	classifier = MLPRegressor(alpha = 1e-4, hidden_layer_sizes = (150,5,), 
		random_state = 12, max_iter = 500, activation = 'relu',
		verbose = True, early_stopping = True, learning_rate_init = 0.001)
	classifier.fit(scaler.transform(x), y)
	return classifier, scaler


# tester of Multi-Layer Perceptron (MLP)

def mlp_test(mlp, test_features, test_labels, scaler, type = 1):
	"""
	test the accuracy / mean square error of a Multi-Layer Perceptron (MLP)
	- inputs:
		mlp: the MLP classifier
		test_features: set of features of test data
		test_labels: class labels of test data
		type: regression(1) or classification(0)
	- outputs:
		accuracy: the accuracy of the MLP
	"""
	predict_labels = mlp.predict(scaler.transform(test_features))
	print predict_labels
	if type == 0:
		return ((predict_labels-test_labels) == 0).sum()/len(test_labels)
	else:
		return np.power(predict_labels-test_labels,2).sum()/len(test_features[0])

def mlp_tester():
	x = [[1,2],[9,10],[2,3],[8,9]]
	y = [0,1,0,1]
	mlp = mlp_train(x,y)
	print mlp
	print mlp_test(mlp, [[1,2],[9,10],[1,2],[9,10]], [0,1,0,1], 0)

#mlp_tester()