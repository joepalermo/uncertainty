import numpy as np
from sklearn.model_selection import train_test_split

def generate_linear_data(size, noise_scale=0.1):
    x = np.random.uniform(-5,5,size=size)
    y = 2*x + 1 + np.random.normal(0, noise_scale, size)
    return x,y

class BLG:

    def __init__(self, n_features, covariance_scale=1):
        self.n_features = n_features
        self.covariance_scale = covariance_scale
        self.prior_covariance_mat = covariance_scale * np.eye(n_features)

    def train(self, X,y):
        self.X = X
        self.y = y
        self.A = np.matmul(X, X.T) + np.linalg.inv(self.prior_covariance_mat)
        self.A_inv = np.linalg.inv(self.A)

    def predict(self, x_star):
        mean = np.matmul(np.matmul(x_star.T, self.A_inv), np.matmul(self.X, self.y))
        variance = np.matmul(np.matmul(x_star.T, self.A_inv),x_star)
        return mean, variance


# generate data
size = 50
n_features = 1
x, y = generate_linear_data(size)
print(x.shape, y.shape)

# split train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# reshape input data
X_train = np.expand_dims(x_train, 1).T
X_test = np.expand_dims(x_test, 1).T
y_train = np.expand_dims(y_train, 1)
y_test = np.expand_dims(y_test, 1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

blg = BLG(n_features)
blg.train(x_train, y_train)
mean, variance = blg.predict(np.array([[15]]))
print(mean, variance)