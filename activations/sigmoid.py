import numpy as np

class Sigmoid(object):

    def forward(self, z):
        """The sigmoid function."""
        return 1.0 / (1.0 + np.exp(-z))

    def backward(self, z):
        """Derivative of the sigmoid function."""
        return self.forward(z) * (1 - self.forward(z))
