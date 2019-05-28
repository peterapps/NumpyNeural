# https://medium.com/@a.mirzaei69/implement-a-neural-network-from-scratch-with-python-numpy-back_propagation-e82b70caa9bb

import numpy as np

class NeuralNetwork(object):
	def __init__(self, layers = [2 , 10, 1], activations=['sigmoid', 'sigmoid']):
		assert(len(layers) == len(activations)+1) # Make sure the number of activation functions is equal to each input layer
		self.layers = layers
		self.activations = activations
		# Initialize weights and biases
		self.weights = []
		self.biases = []
		for i in range(len(layers)-1):
			self.weights.append(np.random.randn(layers[i+1], layers[i]))
			self.biases.append(np.random.randn(layers[i+1], 1))

	def feed_forward(self, x):
		# Return the feed forward value for x
		a = np.copy(x)
		z_s = []
		a_s = [a]
		for i in range(len(self.weights)):
			activation_function = self.get_activation_func(self.activations[i])
			z_s.append(self.weights[i].dot(a) + self.biases[i])
			a = activation_function(z_s[-1])
			a_s.append(a)
		return (z_s, a_s)

	def back_propagation(self,y, z_s, a_s):
		dw = []  # dJ/dW
		db = []  # dJ/dB
		deltas = [None] * len(self.weights)  # delta = dJ/dZ, which is the error for each layer
		# Insert the last layer's error
		deltas[-1] = (y-a_s[-1]) * (self.get_derivative_func(self.activations[-1]))(z_s[-1])
		# Perform back propagation
		for i in reversed(range(len(deltas) - 1)):
			deltas[i] = self.weights[i + 1].T.dot(deltas[i + 1]) * (self.get_derivative_func(self.activations[i])(z_s[i]))		
		batch_size = y.shape[1]
		db = [d.dot(np.ones((batch_size,1))) / float(batch_size) for d in deltas]
		dw = [d.dot(a_s[i].T) / float(batch_size) for i, d in enumerate(deltas)]
		# Return the derivatives with respect to weight matrix and biases
		return dw, db
	
	def train(self, x, y, batch_size=10, epochs=100, lr = 0.01):
		# Update weights and biases based on the output
		for e in range(epochs): 
			i=0
			while(i<len(y)):
				x_batch = x[i:i+batch_size]
				y_batch = y[i:i+batch_size]
				i = i + batch_size
				z_s, a_s = self.feed_forward(x_batch)
				dw, db = self.back_propagation(y_batch, z_s, a_s)
				self.weights = [w+lr*dweight for w, dweight in zip(self.weights, dw)]
				self.biases = [w+lr*dbias for w, dbias in zip(self.biases, db)]
				if e % 100 == 0: print("Epoch {}/{}: loss = {}".format(e, epochs, np.linalg.norm(a_s[-1]-y_batch) ))
	
	def evaluate(self, _X, _y):
		y = _y.flatten()
		_, output = self.feed_forward(_X)
		y_prime = output[-1].flatten()

		percent_errors = []
		for i in range(len(y)):
			percent_errors.append(abs((y_prime[i] - y[i]) / y[i]))

		mean_error = sum(percent_errors) / len(percent_errors)
		print("Mean percent error: {0:.2f}%".format(float((mean_error * 100).astype(str))))
		print("Max error: {0:.2f}%".format(float((max(percent_errors) * 100).astype(str))))
		print("Min error: {0:.2f}%".format(float((min(percent_errors) * 100).astype(str))))
		print("Accuracy: {0:.2f}%".format(float(((1-mean_error) * 100).astype(str))))
	
	@staticmethod
	def get_activation_func(name):
		if name == 'sigmoid':
			return lambda x : np.exp(x)/(1+np.exp(x))
		elif name == 'relu':
			def relu(x):
				y = np.copy(x)
				y[y<0] = 0
				return y
			return relu
		else: # Linear
			return lambda x: x
	
	@staticmethod
	def get_derivative_func(name):
		if name == 'sigmoid':
			sig = lambda x : np.exp(x)/(1+np.exp(x))
			return lambda x : sig(x)*(1-sig(x)) 
		elif name == 'relu':
			def relu_diff(x):
				y = np.copy(x)
				y[y >= 0] = 1
				y[y < 0] = 0
				return y
			return relu_diff
		else: # Linear
			return lambda x: 1
