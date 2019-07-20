import numpy as np
from utils import matmul_list

class BayesianLinearRegressor:
    '''
    Bayesian Linear Regressor. Two variants of training are available that make different tradeoffs.
    :param n_features: The number of features in the data (minus the bias term which is added automatically)
    :param invert_feature_mat: If equal to True computes the posterior distribution by inverting a square
    matrix of size equal to the number of features, otherwise computes it by inverting a square matrix of
    size equal to the number of training examples.
    '''

    def __init__(self, n_features, noise_scale, invert_feature_mat=True):
        self.n_features = n_features + 1  # add one to account for the bias term
        self.prior = np.eye(self.n_features)
        self.invert_feature_mat = invert_feature_mat
        self.noise_scale = noise_scale

    def train(self, X, y):
        '''Train bayesian linear regression. This effectively performs conditioning, i.e. computing the
        posterior distribution of the weights.
        :param X: inputs
        :param Y: outputs
        '''
        if self.invert_feature_mat:
            self.X = X
            self.y = y

            self.A = np.matmul(X, X.T) + np.linalg.inv(self.prior)
            self.A_inv = np.linalg.inv(self.A)
            self.w = matmul_list([self.A_inv, self.X, self.y])
        else:
            self.X = X
            self.y = y
            self.K = matmul_list([X.T, self.prior, X])
            self.B = matmul_list([self.prior,
                                  X,
                                  np.linalg.inv(self.K + np.eye(len(self.K)))
                                  ])
            self.w = matmul_list([self.B, y])
            self.C = matmul_list([self.B, self.X.T, self.prior])

    def predict(self, X_test):
        '''Performs inference using a trained bayesian linear regression model. This effectively performs
        marginalization over the posterior distribution of the weights to obtain the predictive
        distribution for each input).
        :param X_test: inputs
        '''
        if self.invert_feature_mat:
            mean_preds = np.matmul(X_test.T, self.w)
            covariance_preds = matmul_list([X_test.T, self.A_inv, X_test])
        else:
            mean_preds = np.matmul(X_test.T, self.w)
            covariance_preds = matmul_list([X_test.T, self.prior, X_test]) - matmul_list([X_test.T, self.C, X_test])
        # extract the diagonal entries of the diagonal predicted covariance matrix
        variance_preds = np.diag(covariance_preds)
        return mean_preds.flatten(), variance_preds
