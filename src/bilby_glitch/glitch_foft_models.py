import numpy as np
import jax.typing as jnpt
import jax.numpy as jnp
import gwpy
from scipy.signal import savgol_filter

from .series_transforms import taper_off, taper_on

################################################################################
################################################################################
###############################   f(t) Models    ###############################
################################################################################
################################################################################


def slow_scattering_single_arch_f_of_t(
    t: np.array,
    central_time: float,
    modulation_frequency: float,
    harmonic_frequency: float,
):
    """Returns arch frequency as a function of time for a parameterized slow scattering model

    Parameters
    ==========
    t : np.array
        The time array over which to evaluate the frequency
    central_time : float
        The central time (time of peak frequency) for the arch
    modulation_frequency : float
        The driving (microseism) frequency
    harmonic_frequency : float
        The peak frequency of the arch

    Returns
    =======
    np.array
        The frequency of the arch at sampled times
    """
    return harmonic_frequency * np.cos(
        2 * np.pi * modulation_frequency * (t - central_time)
    )


def jax_slow_scattering_f_of_t_all_arches(
    t: jnpt.ArrayLike,
    scattering_parameters: jnpt.ArrayLike,
) -> jnpt.ArrayLike:
    """Produces the frequency as a function
    of time for each arch in a stack.
    This will always produce a 10xN array,
    with N being the length of the timeseries.
    10 is assumed as the nominal maximum number
    of arches.

    Parameters
    ==========
        t : jnpt.ArrayLike
            The times for which to evaluate the
            frequencies, of shape (N,)
        scattering_parameters : jnpt.ArrayLike
            [0] is the modulation_frequency
            which sets the width of the arches
            [1] is the base_harmonic_frequency
            which sets the peak frequency of the
            bottom arch
            [2] is the harmonic_frequency_delta,
            which sets the space between arches
            [3] is the central time, which sets
            the time at which the arches peak
            [4:14] are the respective phases for
            each arch

    Returns
    =======
        jnpt.ArrayLike
            An array of shape (10, N) corresponding
            to the frequency as a function of time
            for 10 stacked arches, where N is the
            length of the timeseries.
    """
    f_harm_k = jnp.arange(10) * scattering_parameters[2] + scattering_parameters[1]
    modulation_of_t = jnp.sin(
        2 * jnp.pi * scattering_parameters[0] * (t - scattering_parameters[3])
    )
    f_of_t_all_arches = jnp.outer(f_harm_k, modulation_of_t)
    f_of_t_all_arches += scattering_parameters[4:14][:, jnp.newaxis]
    return f_of_t_all_arches


def slow_scattering_f_of_t_all_arches(
    t: jnpt.ArrayLike,
    modulation_frequency: float = 0.5,
    base_harmonic_frequency: float = 30,
    harmonic_frequency_delta: float = 25,
    central_time: float = 0,
    phi_0: float = 0,
    phi_1: float = 0,
    phi_2: float = 0,
    phi_3: float = 0,
    phi_4: float = 0,
    phi_5: float = 0,
    phi_6: float = 0,
    phi_7: float = 0,
    phi_8: float = 0,
    phi_9: float = 0,
) -> jnpt.ArrayLike:
    """A wrapper around jax_slow_scattering_f_of_t_all_arches
    For use with APIs which assume parameters are passed as keyword arguments.

    Parameters
    ==========
        t : jnpt.ArrayLike
            The times for which to evaluate the frequency of the arches
        modulation_frequency : float
            The modulation frequency of the scattering arches
        base_harmonic_frequency : float
            The harmonic frequency (i.e. maximum frequency reached) for the 0th arch
        harmonic_frequency_delta : float
            The expected difference in harmonic frequency between the ith and i+1th arch
        central_time : float
            The geocenter time to center the 0th arch at, and optionally all other arches as well
        phi_{i} : float
            {i} runs over 0-9, the phase of the ith arch
    Returns
    =======

    """
    scattering_arguments = jnp.array(
        [
            modulation_frequency,
            base_harmonic_frequency,
            harmonic_frequency_delta,
            central_time,
            phi_0,
            phi_1,
            phi_2,
            phi_3,
            phi_4,
            phi_5,
            phi_6,
            phi_7,
            phi_8,
            phi_9,
        ]
    )
    return jax_slow_scattering_f_of_t_all_arches(t, scattering_arguments)


def slow_scattering_shifting_modulation_f_of_t(
    t: np.array,
    central_time: float,
    modulation_frequency: float,
    modulation_frequency_shift: float,
    harmonic_frequency: float,
    taper_central_time: float,
    taper_time_scale: float,
):
    """Gets the slow scattering frequency with a smooth shift from initial to final modulation frequency

    Parameters
    ==========
    t : np.array
        The time array over which to evaluate the frequency
    central_time : float
        The central time (time of peak frequency) for the arch
    modulation_frequency : float
        The initial driving (microseism) frequency
    modulation_frequency_shift : float
        The ratio between initial and final modulation frequency
    harmonic_frequency : float
        The peak frequency of the arch
    taper_central_time : float
        The midpoint of the logistic function
    taper_time_scale : float
        The characteristic time scale of the logistic function

    Returns
    =======
    np.array
        The sum of the two frequency arches, with a smooth handoff
    """
    initial_function = slow_scattering_single_arch_f_of_t(
        t=t,
        central_time=central_time,
        modulation_frequency=modulation_frequency,
        harmonic_frequency=harmonic_frequency,
    ) * taper_off(
        t=t, taper_central_time=taper_central_time, taper_time_scale=taper_time_scale
    )
    final_function = slow_scattering_single_arch_f_of_t(
        t=t,
        central_time=central_time,
        modulation_frequency=modulation_frequency * modulation_frequency_shift,
        harmonic_frequency=harmonic_frequency * modulation_frequency_shift,
    ) * taper_on(
        t=t, taper_central_time=taper_central_time, taper_time_scale=taper_time_scale
    )
    return initial_function + final_function


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
