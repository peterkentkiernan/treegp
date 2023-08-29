"""
.. module:: kernel
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.gaussian_process.kernels import StationaryKernelMixin, NormalizedKernelMixin, Kernel
from sklearn.gaussian_process.kernels import Hyperparameter
from sklearn.gaussian_process.kernels import _check_length_scale
import copy


def eval_kernel(kernel):
    """
    Some import trickery to get all subclasses 
    of sklearn.gaussian_process.kernels.Kernel
    into the local namespace without doing 
    "from sklearn.gaussian_process.kernels import *"
    and without importing them all manually.

    Example:
    kernel = eval_kernel("RBF(1)") instead of
    kernel = sklearn.gaussian_process.kernels.RBF(1)
    """
    def recurse_subclasses(cls):
        out = []
        for c in cls.__subclasses__():
            out.append(c)
            out.extend(recurse_subclasses(c))
        return out
    clses = recurse_subclasses(Kernel)
    for cls in clses:
        module = __import__(cls.__module__, globals(), locals(), cls)
        execstr = "{0} = module.{0}".format(cls.__name__)
        exec(execstr, globals(), locals())

    from numpy import array

    try:
        k = eval(kernel)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Failed to evaluate kernel string {0!r}.  "
                           "Original exception: {1}".format(kernel, e))

    if isinstance(k.theta, property):
        raise TypeError("String provided was not initialized properly")
    return k

class AnisotropicRBF(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """ A GaussianProcessRegressor Kernel representing a radial basis function (essentially a
    squared exponential or Gaussian) but with arbitrary anisotropic covariance.
    While the parameter for this kernel, an inverse covariance matrix, can be specified directly
    with the `invLam` kwarg, it may be more convenient to instead specify a characteristic
    scale-length for each axis using the `scale_length` kwarg.  Note that a list or array is
    required so that the dimensionality of the kernel can be determined from its length.
    For optimization, it's necessary to reparameterize the inverse covariance matrix in such a way
    as to ensure that it's always positive definite.  To this end, we define `theta` (abbreviated
    `th` below) such that
    invLam = L * L.T
    L = [[exp(th[0])  0              0           ...    0                 0           ]
         [th[n]       exp(th[1])]    0           ...    0                 0           ]
         [th[n+1]     th[n+2]        exp(th[3])  ...    0                 0           ]
         [...         ...            ...         ...    ...               ...         ]
         [th[]        th[]           th[]        ...    exp(th[n-2])      0           ]
         [th[]        th[]           th[]        ...    th[n*(n+1)/2-1]   exp(th[n-1])]]
    I.e., the inverse covariance matrix is Cholesky-decomposed, exp(theta[0:n]) lie on the diagonal
    of the Cholesky matrix, and theta[n:n*(n+1)/2] lie in the lower triangular part of the Cholesky
    matrix.  This parameterization invertably maps all valid n x n covariance matrices to
    R^(n*(n+1)/2).  I.e., the range of each theta[i] is -inf...inf.
    :param  invLam:  Inverse covariance matrix of radial basis function.  Exactly one of invLam and
                     scale_length must be provided.
    :param  scale_length:  Axes-aligned scale lengths of the kernel.  len(scale_length) must be the
                     same as the dimensionality of the kernel, even if the scale length is the same
                     for each axis (i.e., even if the kernel is isotropic).  Exactly one of invLam
                     and scale_length must be provided.
    :param  bounds:  Optional keyword indicating fitting bounds on *theta*.  Can either be a
                     2-element iterable, which will be taken to be the min and max value for every
                     theta element, or an [ntheta, 2] array indicating bounds on each of ntheta
                     elements.
    """
    def __init__(self, invLam=None, scale_length=None, bounds=(-5,5)):
        if scale_length is not None:
            if invLam is not None:
                raise TypeError("Cannot set both invLam and scale_length in AnisotropicRBF.")
            invLam = np.diag(1./np.array(scale_length)**2)

        self.ndim = invLam.shape[0]
        self.ntheta = self.ndim*(self.ndim+1)//2
        self._d = np.diag_indices(self.ndim)
        self._t = np.tril_indices(self.ndim, -1)
        self.set_params(invLam)
        bounds = np.array(bounds)
        if bounds.ndim == 1:
            bounds = np.repeat(bounds[None, :], self.ntheta, axis=0)
        assert bounds.shape == (self.ntheta, 2)
        self._bounds = bounds

    def __call__(self, X, Y=None, eval_gradient=False):
        from scipy.spatial.distance import pdist, cdist, squareform
        X = np.atleast_2d(X)

        if Y is None:
            dists = pdist(X, metric='mahalanobis', VI=self.invLam)
            K = np.exp(-0.5 * dists**2)
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            dists = cdist(X, Y, metric='mahalanobis', VI=self.invLam)
            K = np.exp(-0.5 * dists**2)

        if eval_gradient:
            if self.hyperparameter_cholesky_factor.fixed:
                return K, np.empty((X.shape[0], X.shape[0], 0))
            else:
                # dK_pq/dth_k = -0.5 * K_pq *
                #               ((x_p_i-x_q_i) * dInvLam_ij/dth_k * (x_q_j - x_q_j))
                # dInvLam_ij/dth_k = dL_ij/dth_k * L_ij.T  +  L_ij * dL_ij.T/dth_k
                # dL_ij/dth_k is a matrix with all zeros except for one element.  That element is
                # L_ij if k indicates one of the theta parameters landing on the Cholesky diagonal,
                # and is 1.0 if k indicates one of the thetas in the lower triangular region.
                L_grad = np.zeros((self.ntheta, self.ndim, self.ndim), dtype=float)
                L_grad[(np.arange(self.ndim),)+self._d] = self._L[self._d]
                L_grad[(np.arange(self.ndim, self.ntheta),)+self._t] = 1.0

                half_invLam_grad = np.dot(L_grad, self._L.T)
                invLam_grad = half_invLam_grad + np.transpose(half_invLam_grad, (0, 2, 1))

                dX = X[:, np.newaxis, :] - X[np.newaxis, :, :]
                dist_grad = np.einsum("ijk,lkm,ijm->ijl", dX, invLam_grad, dX)
                K_gradient = -0.5 * K[:, :, np.newaxis] * dist_grad
                return K, K_gradient
        else:
            return K

    @property
    def hyperparameter_cholesky_factor(self):
        return Hyperparameter("CholeskyFactor", "numeric", (1e-5, 1e5), int(self.ntheta))

    def get_params(self, deep=True):
        return {"invLam":self.invLam}

    def set_params(self, invLam=None):
        if invLam is not None:
            self.invLam = invLam
            self._L = np.linalg.cholesky(self.invLam)
            self._theta = np.hstack([np.log(self._L[self._d]), self._L[self._t]])

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        self._theta = theta
        self._L = np.zeros_like(self.invLam)
        self._L[np.diag_indices(self.ndim)] = np.exp(theta[:self.ndim])
        self._L[np.tril_indices(self.ndim, -1)] = theta[self.ndim:]
        self.invLam = np.dot(self._L, self._L.T)

    def __repr__(self):
        return "{0}(invLam={1!r})".format(self.__class__.__name__, self.invLam)

    @property
    def bounds(self):
        return self._bounds


class VonKarman(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """VonKarman kernel.

    Parameters
    -----------
    length_scale : float or array with shape (n_features,), default: 1.0
        The length scale of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of l defines the length-scale of the respective feature dimension.

    length_scale_bounds : pair of floats >= 0, default: (1e-5, 1e5)
        The lower and upper bound on length_scale
    """
    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return Hyperparameter("length_scale", "numeric",
                                  self.length_scale_bounds,
                                  len(self.length_scale))
        return Hyperparameter(
            "length_scale", "numeric", self.length_scale_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        from scipy.spatial.distance import pdist, cdist, squareform
        from scipy import special

        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)
        if Y is None:
            dists = pdist(X, metric='euclidean')
            Filter = (dists != 0.)
            K = np.zeros_like(dists)
            K[Filter] = ((dists[Filter]/length_scale)**(5./6.) *
                         special.kv(5./6.,2*np.pi*dists[Filter]/length_scale))
            K = squareform(K)

            lim0 = special.gamma(5./6.) / (2 * (np.pi**(5./6.)))
            np.fill_diagonal(K, lim0)
            K /= lim0
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")

            dists = cdist(X, Y, metric='euclidean')
            Filter = (dists != 0.)
            K = np.zeros_like(dists)
            K[Filter] = ((dists[Filter]/length_scale)**(5./6.) *
                       special.kv(5./6.,2*np.pi*dists[Filter]/length_scale))
            lim0 = special.gamma(5./6.) / (2 * (np.pi**(5./6.)))
            if np.sum(Filter) != len(K[0])*len(K[:,0]):
                K[~Filter] = lim0
            K /= lim0

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = (K * squareform(dists))[:, :, np.newaxis]
                return K, K_gradient
            elif self.anisotropic:
                raise ValueError(
                    "Gradient can only be evaluated with isotropic VonKarman kernel for the moment.")
        else:
            return K

    def __repr__(self):
        if self.anisotropic:
            return "{0}(length_scale=[{1}])".format(
                self.__class__.__name__, ", ".join(map("{0:.3g}".format,
                                                   self.length_scale)))
        else:  # isotropic
            return "{0}(length_scale={1:.3g})".format(
                self.__class__.__name__, np.ravel(self.length_scale)[0])

class AnisotropicVonKarman(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """ A GaussianProcessRegressor Kernel representing a Von-Karman correlation function
    with an arbitrary anisotropic covariance. While the parameter for this kernel,
    an inverse covariance matrix, can be specified directly with the `invLam` kwarg,
    it may be more convenient to instead specify a characteristic scale-length for each axis
    using the `scale_length` kwarg.  Note that a list or array is required so that the dimensionality
    of the kernel can be determined from its length. For optimization, it's necessary to reparameterize
    the inverse covariance matrix in such a way as to ensure that it's always positive definite.
    To this end, we define `theta` (abbreviated `th` below) such that
    invLam = L * L.T
    L = [[exp(th[0])  0              0           ...    0                 0           ]
         [th[n]       exp(th[1])]    0           ...    0                 0           ]
         [th[n+1]     th[n+2]        exp(th[3])  ...    0                 0           ]
         [...         ...            ...         ...    ...               ...         ]
         [th[]        th[]           th[]        ...    exp(th[n-2])      0           ]
         [th[]        th[]           th[]        ...    th[n*(n+1)/2-1]   exp(th[n-1])]]
    I.e., the inverse covariance matrix is Cholesky-decomposed, exp(theta[0:n]) lie on the diagonal
    of the Cholesky matrix, and theta[n:n*(n+1)/2] lie in the lower triangular part of the Cholesky
    matrix.  This parameterization invertably maps all valid n x n covariance matrices to
    R^(n*(n+1)/2).  I.e., the range of each theta[i] is -inf...inf.
    :param  invLam:  Inverse covariance matrix of radial basis function.  Exactly one of invLam and
                     scale_length must be provided.
    :param  scale_length:  Axes-aligned scale lengths of the kernel.  len(scale_length) must be the
                     same as the dimensionality of the kernel, even if the scale length is the same
                     for each axis (i.e., even if the kernel is isotropic).  Exactly one of invLam
                     and scale_length must be provided.
    :param  bounds:  Optional keyword indicating fitting bounds on *theta*.  Can either be a
                     2-element iterable, which will be taken to be the min and max value for every
                     theta element, or an [ntheta, 2] array indicating bounds on each of ntheta
                     elements.
    """
    def __init__(self, invLam=None, scale_length=None, bounds=(-5,5)):
        if scale_length is not None:
            if invLam is not None:
                raise TypeError("Cannot set both invLam and scale_length in AnisotropicVonKarman.")
            invLam = np.diag(1./np.array(scale_length)**2)

        self.ndim = invLam.shape[0]
        self.ntheta = self.ndim*(self.ndim+1)//2
        self._d = np.diag_indices(self.ndim)
        self._t = np.tril_indices(self.ndim, -1)
        self.set_params(invLam)
        bounds = np.array(bounds)
        if bounds.ndim == 1:
            bounds = np.repeat(bounds[None, :], self.ntheta, axis=0)
        assert bounds.shape == (self.ntheta, 2)
        self._bounds = bounds

    def __call__(self, X, Y=None, eval_gradient=False):
        from scipy.spatial.distance import pdist, cdist, squareform
        from scipy import special
        X = np.atleast_2d(X)

        if Y is None:
            dists = pdist(X, metric='mahalanobis', VI=self.invLam)
            Filter = (dists != 0.)
            K = np.zeros_like(dists)
            K[Filter] = dists[Filter] **(5./6.) *  special.kv(5./6., 2*np.pi * dists[Filter])
            lim0 = special.gamma(5./6.) /(2 * ((np.pi)**(5./6.)) )
            K = squareform(K)
            np.fill_diagonal(K, lim0)
            K /= lim0
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can not be evaluated.")
            dists = cdist(X, Y, metric='mahalanobis', VI=self.invLam)
            Filter = (dists != 0.)
            K = np.zeros_like(dists)
            K[Filter] = dists[Filter] **(5./6.) *  special.kv(5./6., 2*np.pi * dists[Filter])
            lim0 = special.gamma(5./6.) /(2 * ((np.pi)**(5./6.)) )
            if np.sum(Filter) != len(K[0])*len(K[:,0]):
                K[~Filter] = lim0
            K /= lim0

        if eval_gradient:
            raise ValueError(
                "Gradient can not be evaluated.")
        else:
            return K

    @property
    def hyperparameter_cholesky_factor(self):
        return Hyperparameter("CholeskyFactor", "numeric", (1e-5, 1e5), int(self.ntheta))

    def get_params(self, deep=True):
        return {"invLam":self.invLam}

    def set_params(self, invLam=None):
        if invLam is not None:
            self.invLam = invLam
            self._L = np.linalg.cholesky(self.invLam)
            self._theta = np.hstack([np.log(self._L[self._d]), self._L[self._t]])

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        self._theta = theta
        self._L = np.zeros_like(self.invLam)
        self._L[np.diag_indices(self.ndim)] = np.exp(theta[:self.ndim])
        self._L[np.tril_indices(self.ndim, -1)] = theta[self.ndim:]
        self.invLam = np.dot(self._L, self._L.T)

    def __repr__(self):
        return "{0}(invLam={1!r})".format(self.__class__.__name__, self.invLam)

    @property
    def bounds(self):
        return self._bounds

class IndexingArray:
    """ Uses a numpy array, but allows us to specify that it's going to be used for indexing.
        Used to make sparse arrays of a very particular form where the last dimension is very sparse
        but the rest are not. We compress these by storing only the index of where the value is: a
        value of k in the IndexingArray at [i,j] means that the true array at [i,j,k] is its vals[i,j]. """
    
    def __init__(self, indices, vals=None, mask=None, max_depth=None):
        self.indices = indices
        if vals is None:
            self.vals = np.ones_like(self.indices)
        elif vals.shape != indices.shape:
            raise ValueError("Indices and values must match!")
        else:
            self.vals = vals
        if mask is None:
            self.mask = np.ones_like(indices,bool)
        elif mask.shape != indices.shape:
            raise ValueError("Mask and indices must match!")
        else:
            self.mask = mask
        if max_depth is None:
            self.max_depth = np.max(self.indices[self.mask])
        else:
            self.max_depth = max_depth
    
    def to_ndarray(self):
        output = np.zeros((self.shape))
        output[*np.meshgrid(*(range(self.indices.shape[i]) for i in range(self.indices.ndim))),self.indices[self.mask]] = self.vals[mask]
        return output
        
    def sum_last_axis(self):
        output = np.zeros((self.max_depth,))
        for i in range(self.max_depth):
            mask = np.logical_and(self.indices == i,self.mask)
            output[i] = np.sum(self.vals[mask])
    
    def multiply(self,other:np.ndarray,should_copy=False):
        if other.ndim != self.ndim:
            raise ValueError("Multiplication array must be broadcastable to full array!")
        if not np.all((self.shape[i] == other.shape[i] or other.shape[i] == 1 or self.shape[i] == 1 for i in range(self.ndim))):
            print(self.shape,other.shape)
            raise ValueError("Multiplication array must be broadcastable to full array!")
        if should_copy:
            output = copy.deepcopy(self)
        else:
            output = self
        if other.shape[-1] == 1:
            output.vals = output.vals * other[...,0]
        elif np.all(other.shape != 1):
            output.vals = output.vals.astype(other.dtype)
            for i in range(output.max_depth):
                mask = np.logical_and(self.indices == i,self.mask)
                output.vals[mask] *= other[...,i][mask]
        else:
            for i in range(output.max_depth):
                mask = np.logical_and(self.indices == i,self.mask)
                output.vals[mask] *= other[...,i][mask]
        return output
        
    def transpose(self,*axes):
        if axes[-1] not in (-1,self.ndim - 1):
            raise RuntimeError("Transposing with compressed axis currently not supported.")
        else:
            self.vals.transpose(*axes[:-1])
            self.indices.transpose(*axes[:-1])
            self.mask.transpose(*axes[:-1])
            
    def __getitem__(self,key):
        if not self.mask[key[:-1]]:
            return 0
        elif self.indices[key[:-1]] == key[-1]:
            return self.vals[key[:-1]]
        else:
            return 0
            
    def __setitem__(self,key,newval):
        if not self.mask[key[:-1]]:
            self.mask[key[:-1]] = True
            self.indices[key[:-1]] = key[-1]
            self.vals[key[:-1]] = newval
        elif self.indices[key[:-1]] == key[-1]:
            self.vals[key[:-1]] = newval
        else:
            raise KeyError("Cannot stack layers in the compressed axis, must use to_ndarray first or do an appropriate transpose.")
        
    @property
    def shape(self):
        return (*self.indices.shape,self.max_depth)
        
    @property
    def ndim(self):
        return self.indices.ndim + 1

class Binned2PCF(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """ Stores a generic (binned) two point correlation function as a kernel.
    :param  max_sep:  Farthest away distance to be considered. Past this, all correlation is treated
                      as 0.
    :param  n_bins:  Number of bins to be considered if isotropic, or square root of number of bins
                     if anisotropic,
    :param  anisotropic:  False if only distance matters, True if direction does too. Default: True
    :param  two_pcf:  Two-point correlation function. Defaults to all 0s.
    """
    def __init__(self, max_sep=None, n_bins=None, bins=None, anisotropic=True, two_pcf = None): # , fov_bounds
        
        if max_sep is None and bins is None:
            raise RuntimeError("Must somehow specify maximum separation.")
        if n_bins is None and bins is None and two_pcf is None:
            n_bins = 50
        if bins is None:
            if anisotropic:
                self.bins = np.linspace(-1*max_sep,max_sep,n_bins+1)
            else:
                self.bins = np.linspace(0,max_sep,n_bins+1)
        else:
            self.bins = bins
        self.anisotropic = anisotropic
        if two_pcf is None:
            if anisotropic:
                self.two_pcf = np.zeros((n_bins,n_bins))
            else:
                self.two_pcf = np.zeros((n_bins))
        else:
            if anisotropic:
                assert two_pcf.shape[0] == two_pcf.shape[1]
            self.two_pcf = two_pcf
            
        self.total_bins = np.prod(self.two_pcf.shape)

    def __call__(self, X, Y=None, eval_gradient=False):
        from scipy.spatial.distance import pdist, cdist, squareform
        X = np.atleast_2d(X)
        
        if not self.anisotropic:
            if Y is None:
                dists = pdist(X)
            else:
                dists = cdist(X, Y)
            dists = squareform(dists)
            indices = np.searchsorted(self.bins,dists)
            in_range_mask = indices < self.two_pcf.shape[0]
            K = np.zeros_like(indices)
            K[in_range_mask] = self.two_pcf[indices[in_range_mask].flatten()]
            if eval_gradient:
#                grad = np.zeros((*K.shape,self.total_bins))
#                # Set all grad[i,j,indices[i,j]] to 1
#                xs,ys = np.meshgrid(range(K.shape[0]),range(K.shape[1]))
#                xs = xs[in_range_mask].flatten
#                ys = ys[in_range_mask].flatten
#                # numpy indexing is (y,x)
#                grad[ys,xs,indices[in_range_mask].flatten()] = 1
                return K, IndexingArray(indices,mask=in_range_mask,max_depth=self.total_bins)
            else:
                return K
        else:
            if Y is None:
                dists = np.dstack((np.subtract.outer(X[:,0],X[:,0]),np.subtract.outer(X[:,1],X[:,1])))
            else:
                dists = np.dstack((np.subtract.outer(X[:,0],Y[:,0]),np.subtract.outer(X[:,1],Y[:,1])))
            indices = np.searchsorted(self.bins,dists)
            in_range_mask = np.logical_and(np.logical_and(indices[:,:,0] > 0, indices[:,:,1] > 0), np.logical_and(indices[:,:,0] < self.two_pcf.shape[0], indices[:,:,1] < self.two_pcf.shape[1]))
            K = np.zeros((indices.shape[0],indices.shape[1]))
            # numpy indexing is (y,x)
            K[in_range_mask] = self.two_pcf[indices[:,:,1][in_range_mask],indices[:,:,0][in_range_mask]]
            if eval_gradient:
#                grad = np.zeros((*K.shape,self.total_bins))
                # Get the indices that these correspond with in the flattened xi
                flattened_indices = indices @ np.array([self.two_pcf.shape[1],1])
#                # Set all grad[i,j,indices[i,j]] to 1
#                xs,ys = np.meshgrid(range(K.shape[0]),range(K.shape[1]))
#                # numpy indexing is (y,x)
#                grad[ys[in_range_mask],xs[in_range_mask],flattened_indices[in_range_mask]] = 1
                return K, IndexingArray(flattened_indices,mask=in_range_mask,max_depth = self.total_bins)
            else:
                return K
    
    def get_params(self, deep=True):
        return {"two_pcf":self.two_pcf}

    @property
    def theta(self):
        return self.two_pcf.flatten()

    @theta.setter
    def theta(self, theta):
        self.two_pcf = theta.reshape(self.two_pcf.shape)

    def __repr__(self):
        return "{0}(two_pcf={1!r})".format(self.__class__.__name__, self.two_pcf)
        
    def clone_with_theta(self, theta):
        cloned = copy.deepcopy(self)
        cloned.theta = theta
        return cloned

class FourierKernel(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """ Stores the Fourier transform of a generic two point correlation function as a kernel.
    :param ks: Wavenumbers of measured values, shape (n,n,2).
    :param values: The Fourier transform at ks, shape (n,n). values[i,j] should be the Fourier series
                    coefficient for wavevector k[i,j,:].
    :param scale_factor: A scale factor that all input distances should be multiplied by before doing
                        the inverse Fourier transform, that was used when calculating values. Default: 1.
    """
    def __init__(self, ks, values = None, scale_factor = 1.):
        
        self.ks = ks
        if values is None:
            self.values = np.zeros(ks.shape[:-1])
        elif ks.shape[:-1] != values.shape:
            raise ValueError("Input values must correspond to a k vector.")
        elif not np.all(values >= 0):
            raise ValueError("Input values must be nonnegative and real in order to produce PSD kernel matrices.")
        else:
            self.values = values
        self.scale_factor = scale_factor
        self.theta_ndim = np.prod(self.values.shape)

    def __call__(self, X, Y=None, eval_gradient=False, **kwargs):
        from finufft import nufft2d3
        X = np.atleast_2d(X)
        if Y is None:
            dists = np.dstack((np.subtract.outer(X[:,0],X[:,0]),np.subtract.outer(X[:,1],X[:,1])))
        else:
            dists = np.dstack((np.subtract.outer(X[:,0],Y[:,0]),np.subtract.outer(X[:,1],Y[:,1])))
        dists *= self.scale_factor
        K = nufft2d3(self.ks[...,0].flatten(),self.ks[...,1].flatten(),self.values.flatten(),dists[...,0].flatten(),dists[...,1].flatten(),**kwargs).reshape(dists.shape[:-1])
        if eval_gradient:
            grad = np.empty((dists.shape[:-1],self.theta_ndim))
            for i in range(self.theta_ndim):
                cur_indices = np.unravel_index(i,self.values.shape)
                grad[...,i] = nufft2d3(self.ks[cur_indices+tuple([0])].flatten(),self.ks[cur_indices+tuple([1])].flatten(),self.values[cur_indices],dists[...,0].flatten(),dists[...,1].flatten(),**kwargs).reshape(dists.shape[:-1])
                
            return np.real(K), np.real(grad)
        else:
            return np.real(K)
    
    def get_params(self, deep=True):
        return {"wavenumbers":self.ks,"values":self.values}

    @property
    def theta(self):
        return self.values.flatten()

    @theta.setter
    def theta(self, theta):
        self.values = theta.reshape(self.values.shape)

    def __repr__(self):
        return "{0}(wavenumbers={1!r},values={2!r})".format(self.__class__.__name__, self.ks, self.values)
        
    def clone_with_theta(self, theta):
        cloned = copy.deepcopy(self)
        cloned.theta = theta
        return cloned
