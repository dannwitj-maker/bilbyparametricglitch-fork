import numpy as np


def taper_on(t: np.array, taper_central_time: float, taper_time_scale: float):
    """Returns a logistic function, going from off (0) to on (1) smoothly

    Parameters
    ==========
    t : np.array
        The sample time array
    taper_central_time : float
        The midpoint of the logistic function
    taper_time_scale : float
        The characteristic time scale of the logistic function

    Returns
    =======
    np.array
        The smooth logistic function turning on
    """
    return 1 / (1 + np.exp((taper_central_time - t) / taper_time_scale))


def taper_off(t: np.array, taper_central_time: float, taper_time_scale: float):
    """Returns a logistic function, going from on (1) to off (0) smoothly

    Parameters
    ==========
    t : np.array
        The sample time array
    taper_central_time : float
        The midpoint of the logistic function
    taper_time_scale : float
        The characteristic time scale of the logistic function

    Returns
    =======
    np.array
        The smooth logistic function turning off
    """
    return 1 / (1 + np.exp((t - taper_central_time) / taper_time_scale))
