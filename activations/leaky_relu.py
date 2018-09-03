import numpy as np

class LRelu(object):

    def forward(self, z):
        for i in range(len(z)):
            if z[i] <= 0:
                z[i] = 0.1 * z[i]
        return z

    def backward(self, z):
        z[z <= 0] = 0.1
        z[z > 0] = 1
        return z
