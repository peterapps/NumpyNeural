import numpy as np
import neural
import random


#my_seed = 95
#random.seed(my_seed)
#np.random.seed(my_seed)

# Load data
f = open('seeds_dataset.csv', 'r')
features = []
labels = []
rows = f.readlines()
for row in rows:
	values = [float(x) for x in row.split(',')]
	features.append(values[:-1]) # Ignore last column
	label = int(values[-1])
	if label == 1:
		labels.append([1, 0, 0])
	elif label == 2:
		labels.append([0, 1, 0])
	else:
		labels.append([0, 0, 1])

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
nn = neural.NeuralNetwork([7, 5, 3],activations=['sigmoid', 'sigmoid'])

nn.train(X_train, y_train, epochs=1000, batch_size=64, lr = 0.1)

# Evaluate
print(y_test)
_, output = nn.feed_forward(X_test)
y_prime = [x.index(max(x)) for x in output]
percent_errors = []
for i in range(len(y)):
	percent_errors.append(abs((y_prime[i] - y[i]) / y[i]))

mean_error = sum(percent_errors) / len(percent_errors)
print("Mean percent error: {0:.2f}%".format(float((mean_error * 100).astype(str))))
print("Max error: {0:.2f}%".format(float((max(percent_errors) * 100).astype(str))))
print("Min error: {0:.2f}%".format(float((min(percent_errors) * 100).astype(str))))
print("Accuracy: {0:.2f}%".format(float(((1-mean_error) * 100).astype(str))))