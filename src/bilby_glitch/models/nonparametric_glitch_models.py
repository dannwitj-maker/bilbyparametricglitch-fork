import numpy as np
import jax.typing as jnpt
import jax.numpy as jnp
import jax
import gwpy
from scipy.signal import savgol_filter


def auxiliary_f_of_t(
    auxiliary_timeseries: "gwpy.timeseries.TimeSeries", scale: float
) -> np.array:
    """Returns the frequency of the base fringe given an auxiliary timeseries
    and a scale coefficient.
    Based on gwdetchar.scattering.core.get_fringe_frequency L88-114

    Parameters
    ==========
    auxiliary_timeseries : gwpy.timeseries.TimeSeries
        A gwpy timeseries for the auxiliary data to be extrapolated from
    scale : float
        A scale coefficient to multiply by f(t)

    Returns
    =======
    np.array
        The f(t) array predicted by the auxiliary
    """
    velocity = savgol_filter(auxiliary_timeseries.value, 5, 2, deriv=1)
    return np.abs(scale * 2 / 1.064 * velocity * auxiliary_timeseries.sample_rate.value)
