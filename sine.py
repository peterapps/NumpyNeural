import numpy as np
import neural
import matplotlib.pyplot as plt
	
nn = neural.NeuralNetwork([1, 100, 1],activations=['sigmoid', 'sigmoid'])
X = 2*np.pi*np.random.rand(1000).reshape(1, -1)
y = np.sin(X)/2 + 0.5

nn.train(X, y, epochs=1000, batch_size=64, lr = 1)
_, a_s = nn.feed_forward(X)

nn.evaluate(X, y)

plt.scatter(X.flatten(), y.flatten())
plt.scatter(X.flatten(), a_s[-1].flatten())
plt.show()