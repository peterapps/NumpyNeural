import numpy as np
import neural
import random


#my_seed = 95
#random.seed(my_seed)
#np.random.seed(my_seed)

# Load data
def load_data(path):
	f = open(path, 'r')
	features = []
	labels = []
	rows = f.readlines()
	for row in rows[0:200]:
		values = [float(x) for x in row.split(',')]
		features.append(values[1:]) # Ignore first column
		labels.append(values[0:1]) # Only label
	return np.array(features, dtype=np.float128).T, np.array(labels, dtype=np.float128).T

# Split data in training and testing
X_train, y_train = load_data('mnist_train.csv')
X_test, y_test = load_data('mnist_test.csv')

# First train
nn = neural.NeuralNetwork([784, 100, 1],activations=['sigmoid', 'sigmoid'])

nn.train(X_train, y_train, epochs=100, batch_size=200, lr = 0.1)

nn.evaluate(X_test, y_test)
nn.evaluate(X_train, y_train)