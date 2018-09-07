import numpy as np

class LRelu(object):

    def forward(self, z):
        for i in range(len(z)):
            if z[i] <= 0:
                z[i] = 0.1 * z[i]
        return z

    def backward(self, z):
        for i in range(len(z)):
            if z[i] <= 0:
                z[i] = 0.1
            else:
                z[i] = 1
        return z
