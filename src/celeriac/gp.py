# -*- coding: utf-8 -*-

__all__ = ["GaussianProcess"]

from functools import wraps
from typing import Callable, Optional, Union

import jax.numpy as jnp
import numpy as np
from jax import lax

from . import ops, terms
from .types import Array

Mean = Callable[[Array], Array]


class ConstantMean:
    def __init__(self, value: Array = 0.0):
        self.value = value

    def __call__(self, x: Array) -> Array:
        return self.value


def _handle_vector(
    func: Callable[["GaussianProcess", Array], Array]
) -> Callable[["GaussianProcess", Array], Array]:
    @wraps(func)
    def wrapped(self: "GaussianProcess", y: Array) -> None:
        y = jnp.asarray(y)
        if y.ndim == 1:
            return func(self, y[:, None])[:, 0]
        return func(self, y)

    return wrapped


class GaussianProcess:
    """The main interface to the celeriac Gaussian Process (GP) solver

    Args:
        kernel: An instance of a subclass of :class:`terms.Term`.
        mean (optional): The mean function of the process. This can either
            be a callable (it will be evaluated with a single argument, a
            vector of ``x`` values) or a scalar. (default: ``0.0``)
        **kwargs: Other arguments will be passed directly to
            :func:`GaussianProcess.compute` if the argument ``t`` is specified.
    """

    def __init__(
        self,
        kernel: terms.Term,
        t: Array,
        *,
        mean: Union[Mean, Array] = 0.0,
        yerr: Optional[Array] = None,
        diag: Optional[Array] = None,
        proj: Optional[Array] = None,
    ):
        self.kernel = kernel
        if callable(mean):
            self.mean = mean
        else:
            self.mean: Mean = ConstantMean(mean)

        # Check the input coordinates
        t = jnp.asarray(t)
        if t.ndim != 1:
            raise ValueError("The input coordinates must be one dimensional")

        # Save the diagonal
        self._t = t
        self._size = t.shape[0]
        self._mean_value = self._mean(t)

        if yerr is None and diag is None:
            self._diag = jnp.zeros_like(t)

        elif yerr is not None:
            if diag is not None:
                raise ValueError(
                    "only one of 'diag' and 'yerr' can be provided"
                )
            self._diag = jnp.broadcast_to(jnp.square(yerr), t.shape)

        else:
            self._diag = jnp.broadcast_to(diag, t.shape)

        # Fill the celerite matrices
        (
            self._a,
            self._U,
            self._V,
            self._P,
        ) = self.kernel.get_celerite_matrices(self._t, self._diag)

        if proj is not None:
            proj = jnp.asarray(proj)
            self._U *= proj
            self._V *= proj
            self._a = self._diag + jnp.sum(self._U * self._V, axis=1)

        self._d, self._W = ops.factor(self._a, self._U, self._V, self._P)

        self._is_finite = jnp.all(self._d > 0.0)
        self._log_det = jnp.sum(jnp.log(self._d))
        self._norm = -0.5 * (self._log_det + self._size * np.log(2 * np.pi))

    @property
    def mean(self) -> Mean:
        return self._mean

    @mean.setter
    def mean(self, mean: Union[Array, Mean]) -> None:
        if callable(mean):
            self._mean = mean
        else:
            self._mean = ConstantMean(mean)

    @property
    def mean_value(self) -> Array:
        return self._mean_value

    def _process_input(
        self, y: Array, *, require_vector: bool = False
    ) -> Array:
        y = jnp.asarray(y)
        if self._t.shape[0] != y.shape[0]:
            raise ValueError("Dimension mismatch")
        if require_vector and y.ndim != 1:
            raise ValueError("'y' must be one dimensional")
        return y

    @_handle_vector
    def _apply_inverse(self, y: Array) -> Array:
        z = ops.solve_lower(self._U, self._W, self._P, y)
        z /= self._d[:, None]
        return ops.solve_upper(self._U, self._W, self._P, z)

    def apply_inverse(self, y: Array) -> Array:
        """Apply the inverse of the covariance matrix to a vector or matrix

        Solve ``K.x = y`` for ``x`` where ``K`` is the covariance matrix of
        the GP.

        .. note:: The mean function is not applied in this method.

        Args:
            y (shape[N] or shape[N, M]): The vector or matrix ``y`` described
                above.

        Raises:
            RuntimeError: If :func:`GaussianProcess.compute` is not called
                first.
            ValueError: When the inputs are not valid (shape, number, etc.).
        """
        y = self._process_input(y)
        return self._apply_inverse(y)

    @_handle_vector
    def _dot_tril(self, y: Array) -> Array:
        z = y * jnp.sqrt(self._d)[:, None]
        return ops.matmul_lower(self._U, self._V, self._P, z)

    def dot_tril(self, y: Array) -> Array:
        """Dot the Cholesky factor of the GP system into a vector or matrix

        Compute ``x = L.y`` where ``K = L.L^T`` and ``K`` is the covariance
        matrix of the GP.

        .. note:: The mean function is not applied in this method.

        Args:
            y (shape[N] or shape[N, M]): The vector or matrix ``y`` described
                above.

        Raises:
            RuntimeError: If :func:`GaussianProcess.compute` is not called
                first.
            ValueError: When the inputs are not valid (shape, number, etc.).
        """
        y = self._process_input(y)
        return self._dot_tril(y)

    def condition(self, y: Array) -> Array:
        """Condition on data and return the marginalized likelihood

        Args:
            y (shape[N]): The observations at coordinates ``t``

        Raises:
            ValueError: When the inputs are not valid (shape, number, etc.).
        """
        self._y = self._process_input(y, require_vector=True)
        alpha = (self._y - self._mean_value)[:, None]
        self._alpha = ops.solve_lower(self._U, self._W, self._P, alpha)[:, 0]

        def compute_loglike(alpha: Array) -> Array:
            return self._norm - 0.5 * jnp.sum(jnp.square(alpha) / self._d)

        return lax.cond(
            self._is_finite,
            compute_loglike,
            lambda _: -jnp.inf,
            operand=self._alpha,
        )

    # def predict(self, x_test: Optional[Array] = None):
    #     pass


#     def predict(
#         self,
#         y,
#         t=None,
#         *,
#         return_cov=False,
#         return_var=False,
#         include_mean=True,
#         kernel=None,
#     ):
#         """Compute the conditional distribution

#         The factorized matrix from the previous call to
#         :func:`GaussianProcess.compute` is used so that method must be called
#         first.

#         Args:
#             y (shape[N]): The observations at coordinates ``t`` as defined by
#                 :func:`GaussianProcess.compute`.
#             t (shape[M], optional): The independent coordinates where the
#                 prediction should be evaluated. If not provided, this will be
#                 evaluated at the observations ``t`` from
#                 :func:`GaussianProcess.compute`.
#             return_var (bool, optional): Return the variance of the conditional
#                 distribution.
#             return_cov (bool, optional): Return the full covariance matrix of
#                 the conditional distribution.
#             include_mean (bool, optional): Include the mean function in the
#                 prediction.
#             kernel (optional): If provided, compute the conditional
#                 distribution using a different kernel. This is generally used
#                 to separate the contributions from different model components.
#                 Note that the computational cost and scaling will be worse
#                 when using this parameter.

#         Raises:
#             RuntimeError: If :func:`GaussianProcess.compute` is not called
#                 first.
#             ValueError: When the inputs are not valid (shape, number, etc.).
#         """
#         y = self._process_input(y, inplace=True, require_vector=True)
#         return ConditionalDistribution(
#             self, y, t=t, include_mean=include_mean, kernel=kernel
#         )

#     def sample(self, *, size=None, include_mean=True):
#         """Generate random samples from the prior implied by the GP system

#         The factorized matrix from the previous call to
#         :func:`GaussianProcess.compute` is used so that method must be called
#         first.

#         Args:
#             size (int, optional): The number of samples to generate. If not
#                 provided, only one sample will be produced.
#             include_mean (bool, optional): Include the mean function in the
#                 prediction.

#         Raises:
#             RuntimeError: If :func:`GaussianProcess.compute` is not called
#                 first.
#             ValueError: When the inputs are not valid (shape, number, etc.).
#         """

#         if self._t is None:
#             raise RuntimeError("you must call 'compute' first")
#         if size is None:
#             n = np.random.randn(self._size)
#         else:
#             n = np.random.randn(self._size, size)
#         result = self.dot_tril(n, inplace=True).T
#         if include_mean:
#             result += self._mean_value
#         return result

#     def sample_conditional(
#         self,
#         y,
#         t=None,
#         *,
#         size=None,
#         regularize=None,
#         include_mean=True,
#         kernel=None,
#     ):
#         """Sample from the conditional (predictive) distribution

#         .. note:: this method scales as ``O(M^3)`` for large ``M``, where
#             ``M == len(t)``.

#         Args:
#             y (shape[N]): The observations at coordinates ``x`` from
#                 :func:`GausianProcess.compute`.
#             t (shape[M], optional): The independent coordinates where the
#                 prediction should be made. If this is omitted the coordinates
#                 will be assumed to be ``x`` from
#                 :func:`GaussianProcess.compute` and an efficient method will
#                 be used to compute the mean prediction.
#             size (int, optional): The number of samples to generate. If not
#                 provided, only one sample will be produced.
#             regularize (float, optional): For poorly conditioned systems, you
#                 can provide a small number here to regularize the predictive
#                 covariance. This number will be added to the diagonal.
#             include_mean (bool, optional): Include the mean function in the
#                 prediction.
#             kernel (optional): If provided, compute the conditional
#                 distribution using a different kernel. This is generally used
#                 to separate the contributions from different model components.
#                 Note that the computational cost and scaling will be worse
#                 when using this parameter.
#         """
#         mu, cov = self.predict(
#             y, t, return_cov=True, include_mean=include_mean, kernel=kernel
#         )
#         if regularize is not None:
#             cov[np.diag_indices_from(cov)] += regularize
#         return np.random.multivariate_normal(mu, cov, size=size)


# class ConditionalDistribution:
#     def __init__(
#         self,
#         gp,
#         y,
#         t=None,
#         *,
#         include_mean=True,
#         kernel=None,
#     ):
#         self.gp = gp
#         self.y = y
#         self.t = t
#         self.include_mean = include_mean
#         self.kernel = kernel

#         self._c2 = None
#         self._U1 = None
#         self._V1 = None
#         self._U2 = None
#         self._V2 = None

#         self._KxsT = None
#         self._Kinv_KxsT = None

#         if self.t is None:
#             self._xs = self.gp._t
#         else:
#             self._xs = np.ascontiguousarray(self.t, dtype=np.float64)
#             if self._xs.ndim != 1:
#                 raise ValueError("'t' must be one-dimensional")

#     @property
#     def KxsT(self):
#         if self._KxsT is None:
#             tau = self.gp._t[:, None] - self._xs[None, :]
#             if self.kernel is None:
#                 self._KxsT = self.gp.kernel.get_value(tau)
#             else:
#                 self._KxsT = self.kernel.get_value(tau)
#         return self._KxsT

#     @property
#     def Kinv_KxsT(self):
#         if self._Kinv_KxsT is None:
#             self._Kinv_KxsT = self.gp.apply_inverse(self.KxsT, inplace=False)
#         return self._Kinv_KxsT

#     def _do_dot(self, inp, target):
#         if self.kernel is None:
#             kernel = self.gp.kernel
#             U1 = self.gp._U
#             V1 = self.gp._V
#         else:
#             kernel = self.kernel
#             if self._U1 is None or self._V1 is None:
#                 _, _, self._U1, self._V1 = kernel.get_celerite_matrices(
#                     self.gp._t,
#                     np.zeros_like(self.gp._t),
#                     U=self._U1,
#                     V=self._V1,
#                 )
#             U1 = self._U1
#             V1 = self._V1

#         if self._c2 is None or self._U2 is None or self._V2 is None:
#             self._c2, _, self._U2, self._V2 = kernel.get_celerite_matrices(
#                 self._xs,
#                 np.zeros_like(self._xs),
#                 c=self._c2,
#                 U=self._U2,
#                 V=self._V2,
#             )
#         c = self._c2
#         U2 = self._U2
#         V2 = self._V2

#         is_vector = False
#         if inp.ndim == 1:
#             is_vector = True
#             inp = inp[:, None]
#             target = target[:, None]

#         target = driver.general_matmul_lower(
#             self._xs, self.gp._t, c, U2, V1, inp, target
#         )
#         target = driver.general_matmul_upper(
#             self._xs, self.gp._t, c, V2, U1, inp, target
#         )

#         if is_vector:
#             return target[:, 0]
#         return target

#     @property
#     def mean(self):
#         alpha = self.gp._apply_inverse(self.y - self.gp._mean_value)

#         if self.t is None and self.kernel is None:
#             mu = self.y - self.gp._diag * alpha
#             if not self.include_mean:
#                 mu -= self.gp._mean_value
#             return mu

#         mu = np.zeros_like(self._xs)
#         mu = self._do_dot(alpha, mu)

#         if self.include_mean:
#             mu += self.gp._mean(self._xs)
#         return mu

#     @property
#     def variance(self):
#         if self.kernel is None:
#             kernel = self.gp.kernel
#         else:
#             kernel = self.kernel
#         return kernel.get_value(0.0) - np.einsum(
#             "ij,ij->j", self.KxsT, self.Kinv_KxsT
#         )

#     @property
#     def covariance(self):
#         if self.kernel is None:
#             kernel = self.gp.kernel
#         else:
#             kernel = self.kernel
#         neg_cov = -kernel.get_value(self._xs[:, None] - self._xs[None, :])
#         neg_cov = self._do_dot(self.Kinv_KxsT, neg_cov)
#         return -neg_cov

#     def sample(self, *, size=None, regularize=None):
#         """Sample from the conditional (predictive) distribution

#         .. note:: this method scales as ``O(M^3)`` for large ``M``, where
#             ``M == len(t)``.

#         Args:
#             y (shape[N]): The observations at coordinates ``x`` from
#                 :func:`GausianProcess.compute`.
#             t (shape[M], optional): The independent coordinates where the
#                 prediction should be made. If this is omitted the coordinates
#                 will be assumed to be ``x`` from
#                 :func:`GaussianProcess.compute` and an efficient method will
#                 be used to compute the mean prediction.
#             size (int, optional): The number of samples to generate. If not
#                 provided, only one sample will be produced.
#             regularize (float, optional): For poorly conditioned systems, you
#                 can provide a small number here to regularize the predictive
#                 covariance. This number will be added to the diagonal.
#             include_mean (bool, optional): Include the mean function in the
#                 prediction.
#             kernel (optional): If provided, compute the conditional
#                 distribution using a different kernel. This is generally used
#                 to separate the contributions from different model components.
#                 Note that the computational cost and scaling will be worse
#                 when using this parameter.
#         """
#         mu = self.mean
#         cov = self.covariance
#         if regularize is not None:
#             cov[np.diag_indices_from(cov)] += regularize
#         return np.random.multivariate_normal(mu, cov, size=size)
