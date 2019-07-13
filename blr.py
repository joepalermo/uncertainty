from functools import reduce
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

matmul_list = lambda mats: reduce(np.matmul, mats)

def generate_linear_data(size, noise_scale=0.1):
    x = np.random.uniform(-0.1,0.1,size=size)
    y = 2*x + 1 + np.random.normal(0, noise_scale, size)
    return x,y

def plot(x_test, y_test, mean_preds, positive_2_sigma_preds, negative_2_sigma_preds):
    # plot test data
    plt.scatter(x_test, y_test)
    # plot the predicted mean and variance
    plt.plot(x_test, mean_preds)
    plt.plot(np.sort(x_test), np.sort(positive_2_sigma_preds))
    plt.plot(np.sort(x_test), np.sort(negative_2_sigma_preds))
    plt.show()

class BLG:

    def __init__(self, n_features, covariance_scale=1):
        self.n_features = n_features
        self.covariance_scale = covariance_scale
        self.prior_covariance_mat = covariance_scale * np.eye(n_features)

    # conditioning (compute the posterior distribution of the weights)
    def train(self, X,y):
        self.X = X
        self.y = y
        self.A = np.matmul(X, X.T) + np.linalg.inv(self.prior_covariance_mat)
        self.A_inv = np.linalg.inv(self.A)
        self.w = matmul_list([self.A_inv, self.X, self.y])

    # marginalization (marginalize over the weights to obtain predictive distributions)
    def predict(self, x_star):
        mean_preds = np.matmul(x_star.T, self.w)
        covariance_preds = matmul_list([x_star.T, self.A_inv, x_star])
        # extract the diagonal entries of the diagonal predicted covariance matrix
        variance_preds = np.diag(covariance_preds)
        return mean_preds.flatten(), variance_preds


# generate data
size = 500
n_features = 1
x, y = generate_linear_data(size)
# print(x.shape, y.shape)

# split train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# reshape input data
X_train = np.expand_dims(x_train, 1).T
X_test = np.expand_dims(x_test, 1).T
y_train = np.expand_dims(y_train, 1)
y_test = np.expand_dims(y_test, 1)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# add bias term to inputs
train_bias_inputs = np.expand_dims(np.ones(len(x_train)), 1).T
test_bias_inputs = np.expand_dims(np.ones(len(x_test)), 1).T
X_train = np.concatenate([train_bias_inputs, X_train])
X_test = np.concatenate([test_bias_inputs, X_test])
# print(X_train.shape, X_test.shape)

# training
blg = BLG(n_features)
blg.train(X_train, y_train)

# inference
mean_preds, variance_preds = blg.predict(X_test)

# prepare 2-sigma error bars
std_preds = np.sqrt(variance_preds)
positive_2_sigma_preds = mean_preds + 2 * std_preds
negative_2_sigma_preds = mean_preds - 2 * std_preds

# plot
plot(x_test, y_test, mean_preds, positive_2_sigma_preds, negative_2_sigma_preds)
