import numpy as np
import matplotlib.pyplot as plt
import CH1.Data as dt
import CH1.DecisionPlot as decplt

class AdalineGD(object):
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        self.cost_ = []

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])

        for i in range(self.n_iter):
            outputs = self.net_input(X)
            errors = (y - outputs)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum()/2.0
            self.cost_.append(cost)
        return self

# cost plot
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

# eta = 0.01
ada1 = AdalineGD(0.01, 10).fit(dt.X, dt.y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epoki')
ax[0].set_ylabel('Log(suma kwadratów błędów)')
ax[0].set_title('Adaline - współczynnik uczenia 0.01')

# eta = 0.0001
ada2 = AdalineGD(0.0001, 10).fit(dt.X, dt.y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epoki')
ax[1].set_ylabel('Suma kwadratów błędów')
ax[1].set_title('Adaline - współczynnik uczenia 0.0001')

plt.show()

# standard
X_std = np.copy(dt.X)
X_std[:,0] = (dt.X[:,0] - dt.X[:,0].mean()) / dt.X[:,0].std()
X_std[:,1] = (dt.X[:,1] - dt.X[:,1].mean()) / dt.X[:,1].std()

ada = AdalineGD(n_iter=15, eta=0.01)
ada.fit(X_std, dt.y)

# decision plot
decplt.plot_decision_regions(X_std, dt.y, classifier=ada)
plt.title('Adaline - Gradient prosty')
plt.xlabel('Długość działki [standaryzowana]')
plt.ylabel('Długość płatka [standaryzowana]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# cost plot
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epoki')
plt.ylabel('Suma kwadratów błędów')

plt.tight_layout()
plt.show()
