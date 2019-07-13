import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from utils import generate_linear_data, plot_regression
from BayesianLinearRegressor import BayesianLinearRegressor

# generate data
domain = (-1,1)
size = 500
n_features = 1
noise_scale = 0.1
x, y = generate_linear_data(domain, noise_scale, size)
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
blr = BayesianLinearRegressor(n_features, 1)
blr.train_(X_train, y_train)

# inference
mean_preds, variance_preds = blr.predict_(X_test)

# prepare 2-sigma error bars
std_preds = np.sqrt(variance_preds)
positive_2_sigma_preds = mean_preds + 2 * std_preds
negative_2_sigma_preds = mean_preds - 2 * std_preds

# plot
plot_regression(x_test, y_test, mean_preds, positive_2_sigma_preds, negative_2_sigma_preds)
