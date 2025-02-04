from importlib import import_module

__backend__ = ""
SUPPORTED_BACKENDS = ["numpy", "jax"]
_np_module = dict(numpy="numpy", jax="jax.numpy")
_scipy_module = dict(numpy="scipy", jax="jax.scipy")

# from GWPopulation


def modules_to_update():
    """
    Return all modules that need to be updated with the backend.

    Returns
    -------
    all_with_xp: list
        List of all modules that need to be updated with the backend's :code:`xp`
        (:code:`numpy`).
    all_with_scs: list
        List of all modules that need to be updated with the backend's :code:`scs`
        (:code:`scipy.special`).
    others: dict
        Dictionary of all modules that need to be updated with arbitrary functions
        from :code:`scipy`.
    """
    import sys

    if sys.version_info < (3, 10):
        from importlib_metadata import entry_points
    else:
        from importlib.metadata import entry_points
    all_with_xp = [module.value for module in entry_points(group="bilby_glitch.xp")]
    all_with_scs = [module.value for module in entry_points(group="bilby_glitch.scs")]
    other_entries = [
        module.value.split(":") for module in entry_points(group="bilby_glitch.other")
    ]
    others = {key: value for key, value in other_entries}
    return all_with_xp, all_with_scs, others


def _configure_jax(xp, scs):
    """
    Configuration requirements for :code:`jax`

    - use 64-bit floats.
    - update :code:`xp.trapz` to :code:`jax.scipy.integrate.trapezoid`
    """
    from jax import config
    from jax.scipy.integrate import trapezoid
    from .jax_utils import tukey, cumulative_trapezoid

    config.update("jax_enable_x64", True)
    xp.trapz = trapezoid
    scs.cumulative_trapezoid = cumulative_trapezoid
    scs.tukey = tukey


def _configure_numpy_and_scipy(xp, scs):
    scs.tukey = scs.signal.windows.tukey
    scs.cumulative_trapezoid = scs.integrate.cumulative_trapezoid


def _load_numpy_and_scipy(backend):
    try:
        xp = import_module(_np_module[backend])
        scs = import_module(_scipy_module[backend])
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"{backend} not installed for gwpopulation")
    except ImportError:
        raise ImportError(f"{backend} installed but not importable for gwpopulation")

    if backend == "jax":
        _configure_jax(xp, scs)
    if backend == "numpy":
        _configure_numpy_and_scipy(xp, scs)

    return xp, scs


def set_backend(backend="numpy"):
    """
    Set the backend for :code:`GWPopulation` and plugins.

    This will automatically update all modules that have been registered to use
    the :code:`GWPopulation` automatic backend tracking.

    .. warning::

        This will not update existing instances of classes.

    Parameters
    ----------
    backend: str
        The backend to use, one of the :code:`SUPPORTED_BACKENDS`.

    Raises
    ------
    ValueError
        If the backend is not in :code:`SUPPORTED_BACKENDS`.
    """
    global __backend__
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Backend {backend} not supported, should be in {', '.join(SUPPORTED_BACKENDS)}"
        )
    elif backend == __backend__:
        return

    xp, scs = _load_numpy_and_scipy(backend)

    __backend__ = backend
    all_with_xp, all_with_scs, others = modules_to_update()
    for module in all_with_xp:
        setattr(import_module(module), "xp", xp)
    for module in all_with_scs:
        setattr(import_module(module), "scs", scs)

    if backend == "jax":
        from ..models.parametric_glitch_models import (
            set_jit_functions_with_jax_backend,
            xp,
        )

        set_jit_functions_with_jax_backend()
    # for module, func in others.items():
    #     setattr(
    #         import_module(module),
    #         func.split(".")[-1],
    #         _load_arbitrary(func, backend),
    #     )
