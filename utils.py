import functools
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


def matmul_list(mats):
    return functools.reduce(np.matmul, mats)


def generate_linear_data(size, domain, noise_scale, domain_type=None):
    if type(size) == tuple:
        return generate_multi_dim_linear_data(size, domain, noise_scale)
    else:
        num_examples = size
        return generate_uni_dim_linear_data(num_examples, domain, noise_scale, domain_type=domain_type)


def generate_uni_dim_linear_data(num_examples, domain, noise_scale, domain_type=None):
    if domain_type == 'log_normal':
        x = np.clip(np.random.lognormal(0, 3, size=num_examples), domain[0], domain[1])
        x = x[x < domain[1]]
    elif domain_type == 'bimodal':
        x1 = np.random.beta(1, 5, size=num_examples // 2)
        x2 = np.random.beta(5, 1, size=num_examples // 2)
        x = np.concatenate([x1, x2])
    else:
        x = np.random.uniform(domain[0], domain[1], size=num_examples)
    eps = np.random.normal(0, noise_scale, len(x))
    y = 2*x + 1 + eps
    return x, y


def generate_multi_dim_linear_data(size, domain, noise_scale):
    n_features = size[1]
    x = np.random.uniform(domain[0], domain[1], size=size)
    w = np.random.randn(n_features)
    w = np.expand_dims(w, axis=0)
    y_ = np.sum(w * x, axis=1)
    y = 2*y_ + 1 + np.random.normal(0, noise_scale, len(x))
    return x, y


def generate_sinosoidal_data(domain, noise_scale, size):
    x = np.random.uniform(domain[0], domain[1], size=size)
    y = np.sin(x) + np.random.normal(0, noise_scale, size)
    return x, y


def plot_line(xs, ys):
    plt.scatter(xs, ys)
    plt.plot(xs, ys)
    plt.show()


def plot_regression(train, test, mean_preds, positive_2_sigma_preds, negative_2_sigma_preds, show_train=False):
    # extract training and test data
    x_train, y_train = train
    x_test, y_test = test
    # plot test data
    if show_train:
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


def sample_without_replacement(df, n_samples):
    sampled_indices = np.random.choice(df.index, replace=False, size=n_samples)
    return df.loc[sampled_indices], df.drop(sampled_indices, axis=0)