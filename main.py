
from fct import *

d = 0.5
test = Network([4, 4, 2])
print("layers = " + str(test.layers))
print(" w = " + str(test.w))
test.propagate()
print("layers = " + str(test.layers))
print(" ")
test.learn(d, [0, 0])
print("w =" + str(test.w))
print("layers = " + str(test.layers))

test.learn(d, [0, 0])

print("w à 2 learn = " + str(test.w))
print("layers a 2 learn =  " + str(test.layers))
for i in range(10):
    test.learn(d, [0, 0])
test.propagate()
print("layers a 12 learn =  " + str(test.layers))

# I did this a long time ago
# I don't think this is working but i was a great experience
