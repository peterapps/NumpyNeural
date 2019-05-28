# https://www.kaggle.com/sukeshpabba/linear-regression-with-boston-housing-data/data
# https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html

import numpy as np
import neural
import random


#my_seed = 95
#random.seed(my_seed)
#np.random.seed(my_seed)

# Load data
f = open('boston_housing.csv', 'r')
features = []
labels = []
rows = f.readlines()
for row in rows:
	values = [float(x) for x in row.split(',')]
	features.append(values[:-2]) # Ignore last two columns (median value and bias)
	labels.append(values[-2:-1]) # Only median value

# Split data in training and testing
X_train, X_test, y_train, y_test = [], [], [], []
for i in range(len(features)):
	if random.random() > 0.25:
		X_train.append(features[i])
		y_train.append(labels[i])
	else:
		X_test.append(features[i])
		y_test.append(labels[i])
X_train = np.array(X_train, dtype=np.float128).T
y_train = np.array(y_train, dtype=np.float128).T
X_test = np.array(X_test, dtype=np.float128).T
y_test = np.array(y_test, dtype=np.float128).T

print(X_train.shape)
print(y_train.shape)


# First train
nn = neural.NeuralNetwork([13, 8, 5, 1],activations=['sigmoid', 'sigmoid', 'sigmoid'])

nn.train(X_train, y_train, epochs=1000, batch_size=64, lr = 10)

nn.evaluate(X_test, y_test)
