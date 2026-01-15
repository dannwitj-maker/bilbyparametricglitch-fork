"""Methods which are useful for jax but not implemented for whatever reason"""

from jax._src.numpy.lax_numpy import (
    util,
    partial,
    jit,
    ArrayLike,
    Array,
    asarray,
    diff,
    moveaxis,
)
import jax.numpy as jnp
import jax.typing as jnpt
import jax

jax.config.update("jax_enable_x64", True)


@partial(jit, static_argnames=("axis", "initial"))
def cumulative_trapezoid(
    y: ArrayLike,
    x: ArrayLike | None = None,
    dx: ArrayLike = 1.0,
    axis: int = -1,
    initial=None,
) -> Array:
    """Almost exact, built in combination from jax cumsum and scipy's
    cumulative_trapezoid.
    """
    dx_array: Array
    if x is None:
        util.check_arraylike("trapezoid", y)
        (y_arr,) = util.promote_dtypes_inexact(y)
        dx_array = asarray(dx)
    else:
        util.check_arraylike("trapezoid", y, x)
        y_arr, x_arr = util.promote_dtypes_inexact(y, x)
        if x_arr.ndim == 1:
            dx_array = diff(x_arr)
        else:
            dx_array = moveaxis(diff(x_arr, axis=axis), axis, -1)
    y_arr = moveaxis(y_arr, axis, -1)
    res = 0.5 * (dx_array * (y_arr[..., 1:] + y_arr[..., :-1])).cumsum(-1)
    if initial is None:
        return res
    else:
        return jnp.concat([jnp.array([initial]), res], axis=0)


@partial(jax.jit, static_argnames=("M", "alpha"))
def tukey(M, alpha=0.5):
    """Return a Tukey window, also known as a tapered cosine window.
    This snippet is copied from symjax
    https://symjax.readthedocs.io/en/latest/_modules/symjax/tensor/signal.html

    Parameters
    ==========
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    alpha : float, optional
        Shape parameter of the Tukey window, representing the fraction of the
        window inside the cosine tapered region.
        If zero, the Tukey window is equivalent to a rectangular window.
        If one, the Tukey window is equivalent to a Hann window.

    Returns
    =======
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    References
    ==========
    .. [1] Harris, Fredric J. (Jan 1978). "On the use of Windows for Harmonic
           Analysis with the Discrete Fourier Transform". Proceedings of the
           IEEE 66 (1): 51-83. :doi:`10.1109/PROC.1978.10837`
    .. [2] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function#Tukey_window
    """
    n = jnp.arange(0, M)
    width = (jnp.floor(alpha * (M - 1) / 2.0)).astype(int)

    # n1 = n[0 : width + 1]
    # n2 = n[width + 1 : M - width - 1]
    # n3 = n[M - width - 1 :]

    w = jnp.ones_like(n)
    w = jnp.where(
        n <= width, 0.5 * (1 + jnp.cos(jnp.pi * (-1 + 2.0 * n / alpha / (M - 1)))), w
    )
    w = jnp.where(
        n >= M - width,
        0.5 * (1 + jnp.cos(jnp.pi * (-2.0 / alpha + 1 + 2.0 * n / alpha / (M - 1)))),
        w,
    )

    return w


def index_of_nearest_value(value: float, array: jnpt.ArrayLike) -> int:
    """Gets the index of the array element nearest to the value.

    Parameters
    ==========
        value : float
            The value to which the array is compared
        array : jnpt.ArrayLike
            The array being compared

    Returns
    =======
        int
            The index of the element in the array nearest to the value
    """
    return jnp.abs(array - value).argmin()
