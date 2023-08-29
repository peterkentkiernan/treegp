import numpy as np
import copy
from scipy import optimize
from scipy.linalg import cholesky, cho_solve
from .kernels import IndexingArray

class log_likelihood(object):
    """Return and optimize (if requested) the log likelihood of gaussian process.

    :param X:      Coordinates of the field.  (n_samples, 1 or 2)
    :param y:      Values of the field.  (n_samples)
    :param y_err:  Error of y. (n_samples)
    """
    def __init__(self, X, y, y_err, gradient=False):
        self.X = X
        self.ndata = len(self.X[:,0])
        self.y = y
        self.y_err = y_err
        self.gradient = gradient

    def log_likelihood(self, kernel):
        """
        Return of log likehood of gaussian process
        for given hyperparameters.

        :param kernel: Sklearn kernel object.
        """
        try:
            if self.gradient:
                K,cov_grad = kernel.__call__(self.X,eval_gradient=True)
            else:
                K = kernel.__call__(self.X) + np.eye(len(self.y))*self.y_err**2
            L = cholesky(K, overwrite_a=True, lower=False)
            alpha = cho_solve((L, False), self.y, overwrite_b=False)
            chi2 = np.dot(self.y, alpha)
            log_det = np.sum(2.*np.log(np.diag(L)))

            log_likelihood = -0.5 * chi2
            log_likelihood -= (self.ndata / 2.) * np.log((2. * np.pi))
            log_likelihood -= 0.5 * log_det
        except np.linalg.LinAlgError:
            log_likelihood = -np.inf
            if self.gradient:
                self.last_grad = np.zeros(cov_grad.shape[-1])
            return log_likelihood
        
        self.last_logl = log_likelihood
        
        if self.gradient:
            inv_chol = np.linalg.inv(L)
            inv_cov = inv_chol.T @ inv_chol
            jac_log_likelihood = -0.5 * (inv_cov - np.einsum('i,k,ij,hk->jh',self.y,self.y,inv_cov,inv_cov,optimize=True))
            if isinstance(cov_grad,IndexingArray):
                cov_grad.multiply(jac_log_likelihood[...,np.newaxis])
                self.last_grad = cov_grad.sum_last_axis()
            else:
                self.last_grad = np.einsum('ijk,ij->k',jac_log_likelihood,cov_grad,optimize=True)
        return log_likelihood

    def optimizer(self, kernel):
        """
        Fit hyperparameter using maximum likelihood fit.
        Used minimization with L-BFGS-B method from scipy.

        :param kernel: sklearn.gaussian_process kernel.
        """
        self.last_call = None
        def _minus_logl(param, k=kernel):
            if np.all(param == self.last_call):
                return -1 * self.last_logl
            self.last_call = param
            kernel = k.clone_with_theta(param)
            log_l = self.log_likelihood(kernel)
            return -log_l
        
        def _grad_minus_logl(param, k=kernel):
            if not np.all(param == self.last_call):
                self.log_likelihood(kernel)
            return -1 * self.last_grad
        
        p0 = kernel.theta
        
        if self.gradient:
            results_bfgs = optimize.minimize(_minus_logl, p0, method="L-BFGS-B", jac=_grad_minus_logl)
        else:
            results_bfgs = optimize.minimize(_minus_logl, p0, method="L-BFGS-B")
        results = results_bfgs['x']
        kernel = kernel.clone_with_theta(results)
        self._kernel = copy.deepcopy(kernel)
        self._logL = self.log_likelihood(self._kernel)
        return kernel
