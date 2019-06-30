import numpy as np
import matplotlib.pyplot as plt

# data generation utility ----------------------------------------------------------------------------------------------

def generate_input_data(train_size, test_size, plot=False):
    inpt = np.random.uniform(-5,5,size=train_size+test_size)
    out = inpt**5 - 3*inpt**4 - 5*inpt**3 + 6*inpt**2 + 1 + np.random.randn(train_size+test_size)
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

def plot(train_inpt, train_out, test_inpt, test_out, preds):
    plt.scatter(train_inpt, train_out, c='blue')  # plot train
    plt.scatter(test_inpt, test_out, c='red')  # plot test
    plt.scatter(test_inpt, preds, c='green') # plot predictions
    plt.show()

def plot_kernel(kernel):
    mat1, mat2 = np.meshgrid(np.linspace(-20,20,10), np.linspace(-5,5,10))
    pairs = list(zip(mat1.flatten(), mat2.flatten()))
    covariances = [kernel(pair[0], pair[1]) for pair in pairs]
    covariance_matrix = np.reshape(covariances, mat1.shape)
    plt.imshow(covariance_matrix, cmap='hot', interpolation='nearest')
    plt.show()

# kernel utilities -----------------------------------------------------------------------------------------------------

def rbf_kernel(t1,t2, length=1, variance=0.8):
    return variance * np.exp(-np.power(np.linalg.norm(t1-t2),2)/(2*(length**2)))

def linear_kernel(t1,t2, sigma_b=0.8, sigma=0.3, offset=0):
    return (sigma_b**2) + (sigma**2) * (t1-offset) * (t2-offset)

# core Gaussian process functions --------------------------------------------------------------------------------------

def get_covariance_matrix(inp1, inp2, verbose=False):
    mat1, mat2 = np.meshgrid(inp1, inp2)
    pairs = list(zip(mat1.flatten(), mat2.flatten()))
    covariances = [kernel(pair[0], pair[1]) for pair in pairs]
    covariance_matrix = np.reshape(covariances, mat1.shape).T
    if verbose:
        print(pairs)
        print(covariances)
        print(covariance_matrix)
    return covariance_matrix

def compute_conditioned_mean(test_inpt, train_inpt, train_out):
    test_train_cov = get_covariance_matrix(test_inpt, train_inpt)
    train_cov = get_covariance_matrix(train_inpt, train_inpt)
    return np.matmul(np.matmul(test_train_cov,
                               np.linalg.inv(train_cov)),
                     np.reshape(train_out, (len(train_out),1))).flatten()

def compute_conditioned_covariance(test_inpt, train_inpt):
    test_cov = get_covariance_matrix(test_inpt, test_inpt)
    test_train_cov = get_covariance_matrix(test_inpt, train_inpt)
    train_test_cov = get_covariance_matrix(train_inpt, test_inpt)
    train_cov = get_covariance_matrix(train_inpt, train_inpt)
    return test_cov - np.matmul(np.matmul(test_train_cov, np.linalg.inv(train_cov)), train_test_cov)

def sample_predictions(test_inpt, n_samples):
    mean = compute_conditioned_mean(test_inpt, train_inpt, train_out)
    cov = compute_conditioned_covariance(test_inpt, train_inpt)
    preds = np.random.multivariate_normal(mean, cov, size=n_samples).flatten()
    test_inpts = np.concatenate([test_inpt]*n_samples)
    preds = np.concatenate([preds])
    return test_inpts, preds

def predict(test_inpt):
    mean = compute_conditioned_mean(test_inpt, train_inpt, train_out)
    cov = compute_conditioned_covariance(test_inpt, train_inpt)
    return mean, np.diagonal(cov)

# run code -------------------------------------------------------------------------------------------------------------

# set kernel
kernel = rbf_kernel

# generate dataset
train_inpt, train_out, test_inpt, test_out = generate_input_data(15,3, plot=False)

# sample predictions
n_samples = 1
test_inpts, preds = sample_predictions(test_inpt, n_samples)
test_outs = np.concatenate([test_out]*n_samples)
plot(train_inpt, train_out, test_inpts, test_outs, preds)

# marginalize to obtain the mean and variance for each test example
mean, cov = predict(test_inpt)
