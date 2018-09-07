import numpy as np

class Softmax(object):

    def forward(self, z):
        """Compute the softmax of vector x."""
        z = z - np.max(z)
        exp_z = np.exp(z)
        softmax_z = exp_z / np.sum(exp_z)
        return softmax_z

    def backward(self, z):
        """Derivative of the sigmoid function."""
        return self.forward(z) * (1 - self.forward(z))
