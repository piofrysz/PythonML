import numpy as np
import matplotlib.pyplot as plt
import CH1.DecisionPlot as decplt
import CH1.Data as dt


class Perceptron(object):

    def __init__(self, eta=0.1, n_iter=10, random_seed=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_seed = random_seed
        self.errors_ = []
        self.w_ = []

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        #        rgen = np.random.RandomState(self.random_seed)
        #        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)

        return self


# percetron
ppn = Perceptron()
ppn.fit(dt.X, dt.y)

# basic plot
plt.scatter(dt.X[:50, 0], dt.X[:50, 1], color='red', marker='o', label='Setosa')
plt.scatter(dt.X[50:100, 0], dt.X[50:100, 1], color='blue', marker='x', label='Versicolor')

plt.xlabel('Długość działki [cm]')
plt.ylabel('Długość płatka [cm]')
plt.legend(loc='upper left')

plt.show()

# errors plot
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')

plt.xlabel('Epoki')
plt.ylabel('Liczba aktualizacji')

plt.show()

# decision plot
decplt.plot_decision_regions(dt.X, dt.y, classifier=ppn)

plt.xlabel('Długość działki [cm]')
plt.ylabel('Długość płatka [cm]')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()
