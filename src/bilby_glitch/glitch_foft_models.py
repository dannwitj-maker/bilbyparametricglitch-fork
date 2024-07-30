import numpy as np
import gwpy
from scipy.signal import savgol_filter

from .series_transforms import taper_off, taper_on

################################################################################
################################################################################
###############################   f(t) Models    ###############################
################################################################################
################################################################################


def slow_scattering_f_of_t(
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
    initial_function = slow_scattering_f_of_t(
        t=t,
        central_time=central_time,
        modulation_frequency=modulation_frequency,
        harmonic_frequency=harmonic_frequency,
    ) * taper_off(
        t=t, taper_central_time=taper_central_time, taper_time_scale=taper_time_scale
    )
    final_function = slow_scattering_f_of_t(
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
