import numpy as np
import matplotlib.pyplot as plt

# data generation utility ----------------------------------------------------------------------------------------------

def generate_input_data(train_size, test_size, plot=False):
    inpt = np.random.uniform(-5,5,size=train_size+test_size)
    # out = inpt**5 - 3*inpt**4 - 5*inpt**3 + 6*inpt**2 + 1 + np.random.randn(train_size+test_size)
    out = inpt
    # extract train
    y_inpt, y_out = inpt[0:train_size],out[0:train_size]
    # extract test
    x_inpt, x_out = inpt[train_size:], out[train_size:]
    if plot:
        plt.scatter(y_inpt, y_out, c='blue') # plot train
        plt.scatter(x_inpt, x_out, c='red')  # plot test
        plt.show()
    return y_inpt, y_out, x_inpt, x_out

# plotting utilities ---------------------------------------------------------------------------------------------------

def plot(y_inpt, y_out, x_inpt, x_out, preds):
    plt.scatter(y_inpt, y_out, c='blue')  # plot train
    plt.scatter(x_inpt, x_out, c='red')  # plot test
    plt.scatter(x_inpt, preds, c='green') # plot predictions
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

def compute_conditioned_mean(x_inpt, y_inpt, y_out):
    cov_xy = get_covariance_matrix(x_inpt, y_inpt)
    cov_yy = get_covariance_matrix(y_inpt, y_inpt)
    return np.matmul(np.matmul(cov_xy,
                               np.linalg.inv(cov_yy)),
                     np.reshape(y_out, (len(y_out),1))).flatten()

def compute_conditioned_covariance(x_inpt, y_inpt):
    cov_xx = get_covariance_matrix(x_inpt, x_inpt)
    cov_xy = get_covariance_matrix(x_inpt, y_inpt)
    cov_yx = get_covariance_matrix(y_inpt, x_inpt)
    cov_yy = get_covariance_matrix(y_inpt, y_inpt)
    return cov_xx - np.matmul(np.matmul(cov_xy, np.linalg.inv(cov_yy)), cov_yx)

def sample_predictions(x_inpt, n_samples):
    mean = compute_conditioned_mean(x_inpt, y_inpt, y_out)
    cov = compute_conditioned_covariance(x_inpt, y_inpt)
    preds = np.random.multivariate_normal(mean, cov, size=n_samples).flatten()
    x_inpts = np.concatenate([x_inpt]*n_samples)
    preds = np.concatenate([preds])
    return x_inpts, preds

def predict(x_inpt):
    mean = compute_conditioned_mean(x_inpt, y_inpt, y_out)
    cov = compute_conditioned_covariance(x_inpt, y_inpt)
    return mean, np.diagonal(cov)

# run code -------------------------------------------------------------------------------------------------------------

# set kernel
kernel = rbf_kernel

# generate dataset
y_inpt, y_out, x_inpt, x_out = generate_input_data(20,5, plot=False)

# sample predictions
n_samples = 1
x_inpts, preds = sample_predictions(x_inpt, n_samples)
x_outs = np.concatenate([x_out]*n_samples)
plot(y_inpt, y_out, x_inpts, x_outs, preds)

# marginalize to obtain the mean and variance for each test example
mean, cov = predict(x_inpt)
