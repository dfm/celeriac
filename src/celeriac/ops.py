# -*- coding: utf-8 -*-

__all__ = [
    "factor",
    "solve_lower",
    "solve_upper",
    "matmul_lower",
    "matmul_upper",
]

from typing import Tuple

import jax.numpy as jnp
from jax import lax

from .types import Array


def factor(a: Array, U: Array, V: Array, P: Array) -> Tuple[Array, Array]:
    J = U.shape[1]
    Si = jnp.zeros((J, J), dtype=a.dtype)
    di = jnp.zeros_like(a[0])
    Wi = jnp.zeros_like(V[0])
    return lax.scan(_factor_impl, (Si, di, Wi), (a, U, V, P))[1]


def solve_lower(U: Array, W: Array, P: Array, Y: Array) -> Array:
    J = U.shape[1]
    M = Y.shape[1]
    Fi = jnp.zeros((J, M), dtype=W.dtype)
    Wi = jnp.zeros_like(W[0])
    Zi = jnp.zeros_like(Y[0])
    return lax.scan(_solve_impl, (Fi, Wi, Zi), (U, W, P, Y))[1]


def solve_upper(U: Array, W: Array, P: Array, Y: Array) -> Array:
    J = U.shape[1]
    M = Y.shape[1]
    Fi = jnp.zeros((J, M), dtype=W.dtype)
    Wi = jnp.zeros_like(W[0])
    Zi = jnp.zeros_like(Y[0])
    return lax.scan(
        _solve_impl,
        (Fi, Wi, Zi),
        (W, U, jnp.roll(P, -1, axis=0), Y),
        reverse=True,
    )[1]


def matmul_lower(U: Array, V: Array, P: Array, Y: Array) -> Array:
    return jnp.einsum("nj,njk->nk", U, _get_matmul_lower_f(V, P, Y))


def matmul_upper(U: Array, V: Array, P: Array, Y: Array) -> Array:
    return jnp.einsum("nj,njk->nk", V, _get_matmul_upper_f(U, P, Y))


def general_matmul_lower(
    inds: Array, Up: Array, V: Array, P: Array, Pp: Array, Y: Array
) -> Array:
    F = _get_matmul_lower_f(V, P, Y)
    return jnp.einsum("nj,njk->nk", Pp * Up, F[inds])


def general_matmul_upper(
    inds: Array, U: Array, Vp: Array, P: Array, Pp: Array, Y: Array
) -> Array:
    F = _get_matmul_upper_f(U, P, Y)
    return jnp.einsum("nj,njk->nk", Pp * Vp, F[inds])


#
# Below are the inner update functions that are used in the scan
# implementations of each op above
#


def _pdot(P: Array, other: Array, transpose: bool = False) -> Array:
    if P.ndim == 1:
        if transpose:
            return other * P[None, :]
        return P[:, None] * other
    if transpose:
        return other @ P
    return P.T @ other


Carry = Tuple[Array, Array, Array]
Data = Tuple[Array, Array, Array, Array]
MatmulData = Tuple[Array, Array, Array]


def _factor_impl(
    state: Carry, data: Data
) -> Tuple[Carry, Tuple[Array, Array]]:
    Sp, dp, Wp = state
    an, Un, Vn, Pn = data
    Sn = _pdot(Pn, _pdot(Pn, Sp + dp * jnp.outer(Wp, Wp), transpose=True))
    tmp = Sn @ Un
    dn = an - tmp @ Un
    Wn = (Vn - tmp) / dn
    return (Sn, dn, Wn), (dn, Wn)


def _solve_impl(state: Carry, data: Data) -> Tuple[Carry, Array]:
    Fp, Wp, Zp = state
    Un, Wn, Pn, Yn = data
    Fn = _pdot(Pn, Fp + jnp.outer(Wp, Zp))
    Zn = Yn - Un @ Fn
    return (Fn, Wn, Zn), Zn


def _matmul_impl(state: Carry, data: MatmulData) -> Tuple[Carry, Array]:
    (Fp, Vp, Yp) = state
    Vn, Pn, Yn = data
    Fn = _pdot(Pn, Fp + jnp.outer(Vp, Yp))
    return (Fn, Vn, Yn), Fn


def _get_matmul_lower_f(V: Array, P: Array, Y: Array) -> Array:
    J = V.shape[1]
    M = Y.shape[1]
    Fi = jnp.zeros((J, M), dtype=V.dtype)
    Vi = jnp.zeros_like(V[0])
    Yi = jnp.zeros_like(Y[0])
    return lax.scan(_matmul_impl, (Fi, Vi, Yi), (V, P, Y))[1]


def _get_matmul_upper_f(U: Array, P: Array, Y: Array) -> Array:
    J = U.shape[1]
    M = Y.shape[1]
    Fi = jnp.zeros((J, M), dtype=U.dtype)
    Ui = jnp.zeros_like(U[0])
    Yi = jnp.zeros_like(Y[0])
    return lax.scan(
        _matmul_impl,
        (Fi, Ui, Yi),
        (U, jnp.roll(P, -1, axis=0), Y),
        reverse=True,
    )[1]
