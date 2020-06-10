import numpy as np
np.random.seed(0)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from utils import generate_linear_data, plot_regression
from sklearn.metrics import mean_squared_error
from bayesian_linear_regression.BayesianLinearRegressor import BayesianLinearRegressor

# generate data
size = (500, 1)
domain = (0, 0.5)
noise_scale = 0.5
x, y = generate_linear_data(size, domain, noise_scale)

# split train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# reshape input data
if x_train.ndim == 1:
    X_train = np.expand_dims(x_train, 1).T
    X_test = np.expand_dims(x_test, 1).T
else:
    X_train = x_train.T
    X_test = x_test.T
y_train = np.expand_dims(y_train, 1)
y_test = np.expand_dims(y_test, 1)

# add bias term to inputs
train_bias_inputs = np.expand_dims(np.ones(len(x_train)), 1).T
test_bias_inputs = np.expand_dims(np.ones(len(x_test)), 1).T
X_train = np.concatenate([train_bias_inputs, X_train])
X_test = np.concatenate([test_bias_inputs, X_test])

# training
n_features = size[1] if type(size) == tuple else 1
blr = BayesianLinearRegressor(n_features, noise_scale, invert_feature_mat=False)
blr.train(X_train, y_train)

# inference
mean_preds, variance_preds = blr.predict(X_test)
print('test MSE: ', mean_squared_error(y_test[:,0], mean_preds))

# prepare 2-sigma error bars
std_preds = np.sqrt(variance_preds)
positive_2_sigma_preds = mean_preds + 2 * std_preds
negative_2_sigma_preds = mean_preds - 2 * std_preds

# plot
if n_features == 1:
    train = (x_train.flatten(), y_train.flatten())
    test = (x_test.flatten(), y_test.flatten())
    plot_regression(train, test, mean_preds, positive_2_sigma_preds, negative_2_sigma_preds)
