import numpy as np

class Relu(object):

    def forward(self, z):
        for i in range(len(z)):
            if z[i][0] <= 0:
                z[i][0] = 0
        return z

    def backward(self, z):
        for i in range(len(z)):
            if z[i][0] <= 0:
                z[i][0] = 0
            else:
                z[i][0] = 1
        return z
        # z[z <= 0] = 0
        # z[z > 0] = 1
        # return z
