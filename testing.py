import random
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))  # function sigmoid


# initialise the network


class Network:
    def __init__(self, neurones):
        self.neurone = neurones

    def init(self, neurones):
        layers = [[0.0 for j in range(i)] for i in neurones]

        connection_layer = [[neurones[z], neurones[z + 1]] for z in range(len(layers) - 1)]

        w = []
        for h in range(len(connection_layer)):
            w.append([])
            for i in range(connection_layer[h][1]):
                w[h].append([])
                for j in range(connection_layer[h][0]):
                    # w[h][i].append(0.5)  # => 0.5 for the test phase
                    w[h][i].append(random.random())

        return layers, w

    def propagate(self, layers, w):
        for h in range(len(layers) - 1):
            for i in range(len(layers[h + 1])):
                layers[h + 1][i] = 0
                for j in range(len(layers[h])):
                    layers[h + 1][i] += w[h][i][j] * layers[h][j]
                layers[h + 1][i] = sigmoid(layers[h + 1][i])

        return layers

    def learn(self, layers, w, d, t):  # d = delta / t = list of target

        e_list = []
        for i in range(len(w)):
            e_list.append([])
        for i in range(len(layers[-1])):
            e = t[i] - layers[-1][i]
            e_list[-1].append(e)
        for k in reversed(range(len(w) - 1)):
            for i in range(len(layers[k + 1])):
                e = 0
                w_total = 0
                for j in range(len(layers[k + 2])):
                    w_total += w[k][i][j]
                    e += w[k][i][j] * e_list[k + 1][j]
                e = e / w_total
                e_list[k].append(e)

        for i in reversed(range(len(w))):
            for j in range(len(layers[i])):
                for k in range(len(layers[i + 1])):
                    w[i][k][j] -= d * (e_list[i][k] * -1) * layers[i][k] * (1 - layers[i][k]) * layers[i][j]

        return w


d = 0.5
network_1 = Network([3, 3, 3, 3, 3])
l, w = network_1.init([3, 3, 3])
print("layers = " + str(l) + " w = " + str(w))

print(" ")
network_1.propagate(l, w)
print("layers = " + str(l))

print(" ")
network_1.learn(l, w, d, [1, 0, 1])
print("layers = " + str(l))

for i in range(10):
    network_1.learn(l, w, d, [10, 0, 0])
network_1.propagate(l, w)
print("layers a 12 learn =  " + str(l))
print("w =" + str(w))
