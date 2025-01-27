"""Methods which are useful for jax but not implemented for whatever reason"""

from jax._src.numpy import (
    util,
    np,
    partial,
    jit,
    ArrayLike,
    Array,
    asarray,
    diff,
    moveaxis,
)


@util.implements(getattr(np, "trapezoid", getattr(np, "trapz", None)))
@partial(jit, static_argnames=("axis",))
def cumulative_trapezoid(
    y: ArrayLike, x: ArrayLike | None = None, dx: ArrayLike = 1.0, axis: int = -1
) -> Array:
    """Almost exact"""
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
    return 0.5 * (dx_array * (y_arr[..., 1:] + y_arr[..., :-1])).cumsum(-1)
