from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

matmul_list = lambda mats: reduce(np.matmul, mats)

def generate_linear_data(domain, noise_scale, size):
    x = np.random.uniform(domain[0],domain[1], size=size)
    y = 2*x + 1 + np.random.normal(0, noise_scale, size)
    return x,y

def plot_regression(x_test, y_test, mean_preds, positive_2_sigma_preds, negative_2_sigma_preds):
    # plot test data
    plt.scatter(x_test, y_test)
    # plot the predicted mean and variance
    plt.plot(x_test, mean_preds)
    plt.plot(np.sort(x_test), np.sort(positive_2_sigma_preds))
    plt.plot(np.sort(x_test), np.sort(negative_2_sigma_preds))
    plt.show()