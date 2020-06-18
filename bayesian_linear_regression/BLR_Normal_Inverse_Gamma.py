import numpy as np
from utils import matmul_list
from scipy.stats import invgamma


class BLR_Normal_Inverse_Gamma:
    '''
    Bayesian Linear Regressor.
    :param n_features: The number of features in the data (minus the bias term which is added automatically)
    '''

    def __init__(self, n_features, noise_scale=0.25):
        self.n_features = n_features + 1  # add one to account for the bias term
        # priors
        self.a = 6
        self.b = 6
        self.lambda_ = noise_scale
        self.mu = np.zeros(self.n_features)
        self.cov = (1.0 / self.lambda_) * np.eye(self.n_features)
        self.precision = self.lambda_ * np.eye(self.n_features)


    def train(self, X, y):
        '''Train bayesian linear regression. This effectively performs conditioning, i.e. computing the
        posterior distribution of the weights.
        :param X: inputs
        :param y: outputs
        '''
        # some terms are removed as we assume prior mu_0 = 0
        s = np.matmul(X, X.T)
        precision_post = s + self.lambda_ * np.eye(self.n_features)
        cov_post = np.linalg.inv(precision_post)
        mu_post = np.dot(cov_post, np.dot(X, y))
        a_post = self.a + X.T.shape[0] / 2.0
        b_upd = 0.5 * (np.dot(y.T, y) - np.dot(mu_post.T, np.dot(precision_post, mu_post)))
        b_post = self.b + b_upd
        # store new posterior distributions
        self.mu = mu_post
        self.cov = cov_post
        self.precision = precision_post
        self.a = a_post
        self.b = b_post

    def predict(self, X):
        '''Performs inference using a trained bayesian linear regression model. This effectively performs
        marginalization over the posterior distribution of the weights to obtain the predictive
        distribution for each input).
        :param X_test: inputs'''
        # (1,k) (k,n) = (1,n)
        mean_preds = np.matmul(self.mu.T, X)
        # (1) * (n,n) + (n,k) * (k,k) * (k,n) = (n,n)
        covariance_preds = self.b / self.a * (np.eye(len(X.T)) + matmul_list([X.T, self.cov, X]))
        # extract the diagonal entries of the diagonal predicted covariance matrix
        variance_preds = np.diag(covariance_preds)
        return mean_preds.flatten(), variance_preds

    def predict_from_samples(self, X, n_samples=1000):
        '''Performs inference using a trained bayesian linear regression model.
        We characterize the predictive distribution for each input by sampling the posterior many times, and computing
        statistics about the predictive distribution.
        :param X: inputs
        :param n_samples: the number of samples to use in the approximation'''
        sampled_preds = list()
        sampled_sigma2s = list()
        for _ in range(n_samples):
            # sample sigma2, and beta conditional on sigma2
            sampled_sigma2 = self.b * invgamma.rvs(self.a)
            sampled_sigma2s.append(sampled_sigma2)
            sampled_beta = np.random.multivariate_normal(self.mu.flatten(), sampled_sigma2 * self.cov, size=len(X.T))
            sampled_pred = np.sum(sampled_beta * X.T, axis=1)
            sampled_preds.append(np.expand_dims(sampled_pred, 1))
        sampled_preds = np.concatenate(sampled_preds, axis=1)
        mean_preds = np.mean(sampled_preds, axis=1)
        variance_preds = np.mean(sampled_sigma2s) + np.var(sampled_preds, axis=1)
        # extract the diagonal entries of the diagonal predicted covariance matrix
        return mean_preds, variance_preds