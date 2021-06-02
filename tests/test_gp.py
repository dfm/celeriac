# -*- coding: utf-8 -*-

from typing import Tuple

import jax.numpy as jnp
import numpy as np
import pytest
from jax.config import config

import celeriac
from celeriac import terms
from celeriac.types import Array

config.update("jax_enable_x64", True)


@pytest.fixture
def data() -> Tuple[Array, ...]:
    # Generate fake data
    random = np.random.default_rng(40582)
    x = np.sort(random.uniform(0, 10, 50))
    t = np.sort(random.uniform(-1, 12, 100))
    diag = random.uniform(0.1, 0.3, len(x))
    y = np.sin(x)
    return x, diag, y, t


test_terms = [
    terms.SHOTerm(sigma=1.2, rho=0.3, Q=3.4),
    terms.SHOTerm(sigma=np.array([1.2, 0.8]), rho=0.3, Q=np.array([3.4, 0.1])),
    terms.SHOTerm(sigma=1.2, rho=0.3, Q=3.4)
    + terms.SHOTerm(sigma=0.5, rho=0.3, Q=0.4),
    terms.RotationTerm(sigma=1.5, period=3.2, Q0=3.1, dQ=0.5, f=0.7),
]


@pytest.mark.parametrize("term", test_terms)
@pytest.mark.parametrize("mean", [0.0, 10.5])
def test_log_likelihood(
    term: terms.Term, mean: celeriac.gp.Mean, data: Tuple[Array, ...]
) -> None:
    x, diag, y, t = data
    gp = celeriac.GaussianProcess(term, t=x, diag=diag, mean=mean)

    # Check apply inverse
    K = term.get_value(x[:, None] - x[None, :]) + jnp.diag(diag)
    np.testing.assert_allclose(np.linalg.solve(K, y), gp.apply_inverse(y))

    # Check the log likelihood calculation
    resid = y - mean
    alpha = np.linalg.solve(K, resid)
    _, logdet = np.linalg.slogdet(K)
    np.testing.assert_allclose(
        -0.5 * (np.sum(resid * alpha) + logdet + len(x) * np.log(2 * np.pi)),
        gp.condition(y),
    )

    # # Setup the original GP
    # original_gp = original_celerite.GP(oterm, mean=mean)
    # original_gp.compute(x, np.sqrt(diag))

    # # Setup the new GP
    # term = terms.OriginalCeleriteTerm(oterm)
    # gp = celerite2.GaussianProcess(term, mean=mean)
    # gp.compute(x, diag=diag)

    # # "log_likelihood" method
    # assert np.allclose(original_gp.log_likelihood(y), gp.log_likelihood(y))

    # # Apply inverse
    # assert np.allclose(
    #     np.squeeze(original_gp.apply_inverse(y)), gp.apply_inverse(y)
    # )

    # conditional_t = gp.condition(y, t=t)
    # mu, cov = original_gp.predict(y, t=t, return_cov=True)
    # assert np.allclose(conditional_t.mean, mu)
    # assert np.allclose(conditional_t.variance, np.diag(cov))
    # assert np.allclose(conditional_t.covariance, cov)

    # conditional = gp.condition(y)
    # mu, cov = original_gp.predict(y, return_cov=True)
    # assert np.allclose(conditional.mean, mu)
    # assert np.allclose(conditional.variance, np.diag(cov))
    # assert np.allclose(conditional.covariance, cov)

    # # "sample" method
    # seed = 5938
    # np.random.seed(seed)
    # a = original_gp.sample()
    # np.random.seed(seed)
    # b = gp.sample()
    # assert np.allclose(a, b)

    # np.random.seed(seed)
    # a = original_gp.sample(size=10)
    # np.random.seed(seed)
    # b = gp.sample(size=10)
    # assert np.allclose(a, b)

    # # "sample_conditional" method, numerics make this one a little unstable;
    # # just check the shape
    # a = original_gp.sample_conditional(y, t=t)
    # b = conditional_t.sample()
    # assert a.shape == b.shape

    # a = original_gp.sample_conditional(y, size=10)
    # b = conditional.sample(size=10)
    # assert a.shape == b.shape


# def test_diag(data):
#     x, diag, y, t = data

#     term = terms.SHOTerm(S0=1.0, w0=0.5, Q=3.0)
#     gp1 = celerite2.GaussianProcess(term, t=x, diag=diag)
#     gp2 = celerite2.GaussianProcess(term, t=x, yerr=np.sqrt(diag))
#     assert np.allclose(gp1.log_likelihood(y), gp2.log_likelihood(y))

#     gp1 = celerite2.GaussianProcess(term, t=x, diag=np.zeros_like(x))
#     gp2 = celerite2.GaussianProcess(term, t=x)
#     assert np.allclose(gp1.log_likelihood(y), gp2.log_likelihood(y))


# def test_mean(data):
#     x, diag, y, t = data

#     term = terms.SHOTerm(S0=1.0, w0=0.5, Q=3.0)
#     gp = celerite2.GaussianProcess(term, t=x, diag=diag, mean=lambda x: 2 * x)

#     cond1 = gp.condition(y, include_mean=True)
#     cond2 = gp.condition(y, x, include_mean=True)
#     assert np.allclose(cond1.mean, cond2.mean)

#     cond1 = gp.condition(y, include_mean=False)
#     cond2 = gp.condition(y, x, include_mean=False)
#     assert np.allclose(cond1.mean, cond2.mean)

#     cond_mean = gp.condition(y, include_mean=True)
#     cond_no_mean = gp.condition(y, include_mean=False)
#     assert np.allclose(cond_mean.mean, cond_no_mean.mean + 2 * x)

#     cond_mean = gp.condition(y, t, include_mean=True)
#     cond_no_mean = gp.condition(y, t, include_mean=False)
#     assert np.allclose(cond_mean.mean, cond_no_mean.mean + 2 * t)

#     np.random.seed(42)
#     s1 = gp.sample(size=5, include_mean=True)
#     np.random.seed(42)
#     s2 = gp.sample(size=5, include_mean=False)
#     assert np.allclose(s1, s2 + 2 * x)


# def test_predict_kernel(data):
#     x, diag, y, t = data

#     term1 = terms.SHOTerm(S0=1.0, w0=0.5, Q=3.0)
#     term2 = terms.SHOTerm(S0=0.3, w0=0.1, Q=0.1)
#     term = term1 + term2
#     gp = celerite2.GaussianProcess(term, t=x, diag=diag)

#     cond0 = gp.condition(y)
#     cond1 = gp.condition(y, kernel=term)
#     assert np.allclose(cond0.mean, cond1.mean)

#     cond0 = gp.condition(y, t)
#     cond1 = gp.condition(y, t, kernel=term)
#     assert np.allclose(cond0.mean, cond1.mean)

#     cond0 = gp.condition(y, t)
#     cond1 = gp.condition(y, t, kernel=term1)
#     cond2 = gp.condition(y, t, kernel=term2)
#     assert np.allclose(cond0.mean, cond1.mean + cond2.mean)

#     cond1 = gp.condition(y, kernel=term1)
#     mu2 = term1.dot(x, np.zeros_like(x), gp.apply_inverse(y))
#     assert np.allclose(cond1.mean, mu2)


# def test_errors():
#     # Generate fake data
#     np.random.seed(40582)
#     x = np.sort(np.random.uniform(0, 10, 50))
#     t = np.sort(np.random.uniform(-1, 12, 100))
#     diag = np.random.uniform(0.1, 0.3, len(x))
#     y = np.sin(x)

#     term = terms.SHOTerm(S0=1.0, w0=0.5, Q=3.0)
#     gp = celerite2.GaussianProcess(term)

#     # Need to call compute first
#     with pytest.raises(RuntimeError):
#         gp.recompute()

#     with pytest.raises(RuntimeError):
#         gp.log_likelihood(y)

#     with pytest.raises(RuntimeError):
#         gp.sample()

#     # Sorted
#     with pytest.raises(ValueError):
#         gp.compute(x[::-1], diag=diag)

#     # 1D
#     with pytest.raises(ValueError):
#         gp.compute(np.tile(x[:, None], (1, 5)), diag=diag)

#     # Only one of diag and yerr
#     with pytest.raises(ValueError):
#         gp.compute(x, diag=diag, yerr=np.sqrt(diag))

#     # Not positive definite
#     # with pytest.raises(celerite2.driver.LinAlgError):
#     with pytest.raises(Exception):
#         gp.compute(x, diag=-10 * diag)

#     # Not positive definite with `quiet`
#     gp.compute(x, diag=-10 * diag, quiet=True)
#     assert np.isinf(gp._log_det)
#     assert gp._log_det < 0

#     # Compute correctly
#     gp.compute(x, diag=diag)
#     gp.log_likelihood(y)

#     # Dimension mismatch
#     with pytest.raises(ValueError):
#         gp.log_likelihood(y[:-1])

#     with pytest.raises(ValueError):
#         gp.log_likelihood(np.tile(y[:, None], (1, 5)))

#     with pytest.raises(ValueError):
#         gp.predict(y, t=np.tile(t[:, None], (1, 5)))
