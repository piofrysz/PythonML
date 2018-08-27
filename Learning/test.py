import numpy as np
import matplotlib.pyplot as plt

testX = np.array([1, 5, 2, 6, 3, 4])
testY = np.array([1, 3, 5, 7, 2, 1])

X = np.arange(4, 8)
Y = np.arange(3, 7)
a, b = np.meshgrid(X, Y)

print(a)
print(b)
print(X)
print(Y)
plt.contourf(a, b, [[1, -1, -1, 1], [1, -1, -1, 1],
                    [1, 2, 2, 1], [-1, -1, -1, -1]])

# plt.scatter(testX, testY)
plt.show()


class Test():
    def __init__(self):
        pass


print("TEST")
