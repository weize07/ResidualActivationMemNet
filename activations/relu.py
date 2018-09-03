import numpy as np

class Relu(object):

    def forward(self, z):
        return np.maximum(0, z)

    def backward(self, z):
        z[z <= 0] = 0
        z[z > 0] = 1
        return z
