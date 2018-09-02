class Relu(object):

    def forward(self, z):
        print(z)
        return max(0, z)

    def backward(self, z):
        if z > 0:
            return 1
        return 0
