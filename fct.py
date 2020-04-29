import random
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))  # function sigmoid


# initialise the network


class Network:
    def __init__(self, neurones):
        self.neurones = neurones

        self.layers = [[0.0 for j in range(i)] for i in self.neurones]

        self.connection_layer = [[self.neurones[z], self.neurones[z + 1]] for z in range(len(self.layers) - 1)]

        self.w = []
        for h in range(len(self.connection_layer)):
            self.w.append([])
            for i in range(self.connection_layer[h][1]):
                self.w[h].append([])
                for j in range(self.connection_layer[h][0]):
                    # w[h][i].append(0.5)  # => 0.5 for the test phase
                    self.w[h][i].append(random.random())

    def propagate(self):
        for h in range(len(self.layers) - 1):
            for i in range(len(self.layers[h + 1])):
                self.layers[h + 1][i] = 0
                for j in range(len(self.layers[h])):
                    self.layers[h + 1][i] += self.w[h][i][j] * self.layers[h][j]
                self.layers[h + 1][i] = sigmoid(self.layers[h + 1][i])

    def learn(self, d, t):  # d = delta / t = list of target

        e_list = []
        for i in range(len(self.w)):
            e_list.append([])
        for i in range(len(self.layers[-1])):
            e = t[i] - self.layers[-1][i]
            e_list[-1].append(e)
        for k in reversed(range(len(self.w) - 1)):
            for i in range(len(self.layers[k + 1])):
                e = 0
                w_total = 0
                for j in range(len(self.layers[k + 2])):
                    w_total += self.w[k][i][j]
                    e += self.w[k][i][j] * e_list[k + 1][j]
                e = e / w_total
                e_list[k].append(e)

        for i in reversed(range(len(self.w))):
            for j in range(len(self.layers[i])):
                for k in range(len(self.layers[i + 1])):
                    self.w[i][k][j] -= d * (e_list[i][k] * -1) * self.layers[i][k] * (1 - self.layers[i][k]) * self.layers[i][j]

