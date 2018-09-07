import numpy as np
import matplotlib.pyplot as plt
import CH1.Data as dt
import CH1.DecisionPlot as decplt

class AdalineSGD(object):

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.cost_ = []
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            np.random.seed(random_state)

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)

    def _shuffle(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialized_weights(self, m):
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        output = self.net_input(xi)
        error = target - output
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost

    def fit(self, X, y):
        self._initialized_weights(X.shape[1])

        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)

            cost_temp = []
            for xi, target in zip(X, y):
                cost_temp.append(self._update_weights(xi, target))

            avg_cost = sum(cost_temp)/len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        if not self.w_initialized:
            self._initialized_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self


ada = AdalineSGD(n_iter=15, random_state=1)

# standard
X_std = np.copy(dt.X)
X_std[:,0] = (dt.X[:,0] - dt.X[:,0].mean()) / dt.X[:,0].std()
X_std[:,1] = (dt.X[:,1] - dt.X[:,1].mean()) / dt.X[:,1].std()

ada.fit(X_std, dt.y)
decplt.plot_decision_regions(X_std, dt.y, classifier=ada)
plt.title('Adaline - Stochastyczny spadek wzdłuż gradientu')
plt.xlabel('Długość działki [standaryzowana]')
plt.ylabel('Długość płatka [standaryzowana]')
plt.legend(loc='upper left')
plt.show()
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epoki')
plt.ylabel('Średni koszt')
plt.show()
