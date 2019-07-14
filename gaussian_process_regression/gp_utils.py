import numpy as np
from utils import matmul_list

def squared_exponential_kernel(x1,x2):
    return np.exp(-0.5*np.linalg.norm(x1-x2)**2)

def gaussian_process_inference(X_test, X_train, y_train, kernel):
    n_train_examples = len(X_train.T)
    n_test_examples = len(X_test.T)
    k1 = np.array([[kernel(X_test.T[j, :], X_train[:, i])
                    for i in range(n_train_examples)]
                   for j in range(n_test_examples)])
    k2 = np.array([[kernel(X_train.T[j, :], X_train[:, i])
                    for i in range(n_train_examples)]
                   for j in range(n_train_examples)])
    k3 = np.array([[kernel(X_test.T[j, :], X_test[:, i])
                    for i in range(n_test_examples)]
                   for j in range(n_test_examples)])
    k4 = np.array([[kernel(X_train.T[j, :], X_test[:, i])
                    for i in range(n_test_examples)]
                   for j in range(n_train_examples)])
    k2_mod = np.linalg.inv(k2 + np.eye(n_train_examples))
    mean_preds = matmul_list([k1, k2_mod, y_train])
    covariance_preds = k3 - matmul_list([k1, k2_mod, k4])
    # extract the diagonal entries of the diagonal predicted covariance matrix
    variance_preds = np.diag(covariance_preds)
    return mean_preds.flatten(), variance_preds