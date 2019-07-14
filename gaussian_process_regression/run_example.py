import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from utils import generate_sinosoidal_data, plot_regression
from gaussian_process_regression.gp_utils import *

# set data parameters
train_domain = (-5, 5)
test_domain = (-2, 8)
size = 100
n_features = 1
noise_scale = 0.1

# generate data
x_train, y_train = generate_sinosoidal_data(train_domain, noise_scale, size)
x_test, y_test = generate_sinosoidal_data(test_domain, noise_scale, size)

# reshape input data
X_train = np.expand_dims(x_train, 1).T
X_test = np.expand_dims(x_test, 1).T
y_train = np.expand_dims(y_train, 1)
y_test = np.expand_dims(y_test, 1)

# add bias term to inputs
train_bias_inputs = np.expand_dims(np.ones(len(x_train)), 1).T
test_bias_inputs = np.expand_dims(np.ones(len(x_test)), 1).T
X_train = np.concatenate([train_bias_inputs, X_train])
X_test = np.concatenate([test_bias_inputs, X_test])

# inference
mean_preds, variance_preds = gaussian_process_inference(X_test, X_train, y_train, kernel=squared_exponential_kernel)

# prepare 2-sigma error bars
std_preds = np.sqrt(variance_preds)
positive_2_sigma_preds = mean_preds + 2 * std_preds
negative_2_sigma_preds = mean_preds - 2 * std_preds

# plot
train_data = (x_train, y_train)
test_data = (x_test, y_test)
plot_regression(train_data, test_data, mean_preds, positive_2_sigma_preds, negative_2_sigma_preds)
