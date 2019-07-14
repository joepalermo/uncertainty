from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

matmul_list = lambda mats: reduce(np.matmul, mats)

def squared_exponential_kernel(x1,x2):
    return np.exp(-0.5*np.linalg.norm(x1-x2)**2)

def generate_linear_data(domain, noise_scale, size):
    x = np.random.uniform(domain[0],domain[1], size=size)
    y = 2*x + 1 + np.random.normal(0, noise_scale, size)
    return x,y

def generate_sinosoidal_data(domain, noise_scale, size):
    x = np.random.uniform(domain[0],domain[1], size=size)
    y = np.sin(x) + np.random.normal(0, noise_scale, size)
    return x,y

def plot_regression(train, test, mean_preds, positive_2_sigma_preds, negative_2_sigma_preds):
    # extract training and test data
    x_train, y_train = train
    x_test, y_test = test
    # plot test data
    plt.scatter(x_train, y_train)
    plt.scatter(x_test, y_test)
    # sort examples by position on the x-axis so that line plots work
    sort_order = np.argsort(x_test)
    x_test = x_test[sort_order]
    mean_preds = mean_preds[sort_order]
    positive_2_sigma_preds = positive_2_sigma_preds[sort_order]
    negative_2_sigma_preds = negative_2_sigma_preds[sort_order]
    # plot the predicted mean and variance
    plt.plot(x_test, mean_preds)
    plt.plot(x_test, positive_2_sigma_preds)
    plt.plot(x_test, negative_2_sigma_preds)
    plt.show()