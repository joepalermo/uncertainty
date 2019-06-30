import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse

# data generation utility ----------------------------------------------------------------------------------------------

def f(x):
    size = len(x)
    return x + np.random.randn(size)

def generate_input_data(train_size, test_size, plot=False):
    inpt = np.linspace(-10,10,num=train_size+test_size)
    out = f(inpt)
    # extract train
    train_inpt, train_out = inpt[0:train_size],out[0:train_size]
    # extract test
    test_inpt, test_out = inpt[train_size:], out[train_size:]
    if plot:
        plt.scatter(train_inpt, train_out, c='blue') # plot train
        plt.scatter(test_inpt, test_out, c='red')  # plot test
        plt.show()
    return train_inpt, train_out, test_inpt, test_out

# plotting utilities ---------------------------------------------------------------------------------------------------

def plot(train_inpt, train_out, test_inpt, test_out, preds, std_pos, std_neg):
    plt.plot(train_inpt, train_out, c='blue', linewidth=1)  # plot train
    plt.plot(test_inpt, test_out, c='red', linewidth=1)  # plot test
    plt.plot(test_inpt, preds, c='green', linewidth=1) # plot predictions
    plt.plot(test_inpt, std_pos, c='black', linewidth=1)
    plt.plot(test_inpt, std_neg, c='black', linewidth=1)
    plt.show()

def plot_kernel(kernel):
    mat1, mat2 = np.meshgrid(np.linspace(-20,20,10), np.linspace(-5,5,10))
    pairs = list(zip(mat1.flatten(), mat2.flatten()))
    covariances = [kernel(pair[0], pair[1]) for pair in pairs]
    covariance_matrix = np.reshape(covariances, mat1.shape)
    plt.imshow(covariance_matrix, cmap='hot', interpolation='nearest')
    plt.show()

# kernel utilities -----------------------------------------------------------------------------------------------------

# length=0.8, variance=1
def rbf_kernel(t1,t2, hparams):
    length = hparams['length']
    variance = hparams['variance']
    return variance * np.exp(-np.power(np.linalg.norm(t1-t2),2)/(2*(length**2)))

# sigma_b=0.8, sigma=0.3, offset=0
def linear_kernel(t1,t2, hparams):
    sigma_b = hparams['sigma_b']
    sigma = hparams['sigma']
    offset = hparams['offset']
    return (sigma_b**2) + (sigma**2) * (t1-offset) * (t2-offset)

# core Gaussian process functions --------------------------------------------------------------------------------------

def get_covariance_matrix(inp1, inp2, hparams, verbose=False):
    mat1, mat2 = np.meshgrid(inp1, inp2)
    pairs = list(zip(mat1.flatten(), mat2.flatten()))
    kernel = hparams['kernel']
    covariances = [kernel(pair[0], pair[1], hparams) for pair in pairs]
    covariance_matrix = np.reshape(covariances, mat1.shape).T
    if verbose:
        print(pairs)
        print(covariances)
        print(covariance_matrix)
    return covariance_matrix

def compute_conditioned_mean(test_inpt, train_inpt, train_out, hparams):
    test_train_cov = get_covariance_matrix(test_inpt, train_inpt, hparams)
    train_cov = get_covariance_matrix(train_inpt, train_inpt, hparams)
    return np.matmul(np.matmul(test_train_cov,
                               np.linalg.inv(train_cov)),
                     np.reshape(train_out, (len(train_out),1))).flatten()

def compute_conditioned_covariance(test_inpt, train_inpt, hparams):
    test_cov = get_covariance_matrix(test_inpt, test_inpt, hparams)
    test_train_cov = get_covariance_matrix(test_inpt, train_inpt, hparams)
    train_test_cov = get_covariance_matrix(train_inpt, test_inpt, hparams)
    train_cov = get_covariance_matrix(train_inpt, train_inpt, hparams)
    return test_cov - np.matmul(np.matmul(test_train_cov, np.linalg.inv(train_cov)), train_test_cov)

def sample_predictions(test_inpt, n_samples, hparams):
    mean = compute_conditioned_mean(test_inpt, train_inpt, train_out, hparams)
    cov = compute_conditioned_covariance(test_inpt, train_inpt, hparams)
    preds = np.random.multivariate_normal(mean, cov, size=n_samples).flatten()
    test_inpts = np.concatenate([test_inpt]*n_samples)
    preds = np.concatenate([preds]*n_samples)
    return test_inpts, preds

def predict(test_inpt, hparams):
    mean = compute_conditioned_mean(test_inpt, train_inpt, train_out, hparams)
    cov = compute_conditioned_covariance(test_inpt, train_inpt, hparams)
    return mean, np.diagonal(cov)

def tune_hparams(length_range, variance_range, n_trials):
    # prepare grid search
    grid_size = int(n_trials**(1/2))
    mat1, mat2 = np.meshgrid(np.linspace(length_range[0], length_range[1], grid_size),
                            np.linspace(variance_range[0], variance_range[1], grid_size))
    pairs = list(zip(mat1.flatten(), mat2.flatten()))
    scores = list()
    for pair in pairs:
        hparams = {'kernel': rbf_kernel,'length': pair[0], 'variance': pair[1]}
        mean, _ = predict(test_inpt, hparams)
        scores.append(mse(test_out, mean))
    print(scores)
    print(pairs)
    min_score, min_score_index = np.min(scores), np.argmin(scores)
    min_params = pairs[min_score_index]
    return min_params, min_score

# run code -------------------------------------------------------------------------------------------------------------

# generate dataset
train_inpt, train_out, test_inpt, test_out = generate_input_data(25,5, plot=False)

min_params, min_score = tune_hparams((0.01,3), (0.01,3), n_trials=100)
print(min_params, min_score)

# # marginalize to obtain the mean and variance for each test example
# hparams = {'kernel': rbf_kernel, 'length': 1.04, 'variance': 0.34}
# mean, variance = predict(test_inpt, hparams)
# std_pos = mean + 2*variance
# std_neg = mean - 2*variance
#
# # plot
# plot(train_inpt, train_out, test_inpt, test_out, mean, std_pos, std_neg)