import logging
import numpy as np
import copy
from scipy.signal.windows import tukey
from scipy.signal import savgol_filter
from scipy.integrate import cumtrapz
from bilby.core.utils import nfft

from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import gwpy

################################################################################
################################################################################
####################   Utility Functions and Definitions    ####################
################################################################################
################################################################################


SLOW_SCATTERING_ORIGINAL_PARAMETERS = [
    "geocent_time",
    "f_diff",
    "f_harm0",
    "delta_f_harm1",
    "delta_f_harm2",
    "delta_f_harm3",
    "delta_f_harm4",
    "delta_f_harm5",
    "delta_f_harm6",
    "delta_f_harm7",
    "delta_f_harm8",
    "delta_f_harm9",
    "f_mod0",
    "delta_f_mod1",
    "delta_f_mod2",
    "delta_f_mod3",
    "delta_f_mod4",
    "delta_f_mod5",
    "delta_f_mod6",
    "delta_f_mod7",
    "delta_f_mod8",
    "delta_f_mod9",
    "phi0",
    "phi1",
    "phi2",
    "phi3",
    "phi4",
    "phi5",
    "phi6",
    "phi7",
    "phi8",
    "phi9",
    "amp0",
    "amp1",
    "amp2",
    "amp3",
    "amp4",
    "amp5",
    "amp6",
    "amp7",
    "amp8",
    "amp9",
    "delta_gc_t1",
    "delta_gc_t2",
    "delta_gc_t3",
    "delta_gc_t4",
    "delta_gc_t5",
    "delta_gc_t6",
    "delta_gc_t7",
    "delta_gc_t8",
    "delta_gc_t9",
]

SLOW_SCATTERING_PARAMETERS = [
    "geocent_time",
    "harmonic_frequency_delta",
    "base_harmonic_frequency",
    "modulation_frequency",
    "phi_0",
    "phi_1",
    "phi_2",
    "phi_3",
    "phi_4",
    "phi_5",
    "phi_6",
    "phi_7",
    "phi_8",
    "phi_9",
    "amplitude_0",
    "amplitude_1",
    "amplitude_2",
    "amplitude_3",
    "amplitude_4",
    "amplitude_5",
    "amplitude_6",
    "amplitude_7",
    "amplitude_8",
    "amplitude_9",
]

SLOW_SCATTERING_SHIFTING_PARAMETERS = SLOW_SCATTERING_PARAMETERS + [
    "modulation_frequency_shift",
    "taper_central_time",
    "taper_time_scale",
]

FAST_SCATTERING_PARAMETERS = [
    "amplitude",
    "phi",
    "microseism_frequency",
    "microseism_coefficient",
    "anthropogenic_frequency",
    "anthropogenic_coefficient",
    "geocent_time",
]


def populate_unused_parameters(
    original_dict: dict,
    full_parameter_list: list = SLOW_SCATTERING_PARAMETERS,
    add_sky_params=False,
) -> dict:
    """Fill out a dictionary (either injection parameters or a priordict)
    with the parameters needed for the model, but unused.
    These are all set as 0.

    Parameters
    ==========
    original_dict : dict
        The dictionary to modify
    full_parameter_list : list, optional
        The list of parameters to fill out, defaults to those in the slow scattering SHO model
    add_sky_params : bool
        Whether to also add unused sky params needed by Bilby - should be turned off if a cbc model is also present

    Returns
    =======
    dict
        The updated dictionary with unused but necessary parameters set as 0
    """
    temp_dict = copy.deepcopy(original_dict)
    for key in full_parameter_list:
        if key not in original_dict:
            temp_dict[key] = 0
    if add_sky_params:
        temp_dict["ra"] = 0
        temp_dict["dec"] = 0
        temp_dict["psi"] = 0
    return temp_dict


def _blank_frequency_domain_source_model(f, geocent_time, **kwargs):
    """A placeholder for when no CBC is passed to a joint analysis"""
    return {"plus": np.zeros(f.shape), "cross": np.zeros(f.shape)}


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


################################################################################
################################################################################
##########################    Parameterized Models    ##########################
################################################################################
################################################################################


def slow_scattering(
    f,
    geocent_time=0,
    harmonic_frequency_delta=25,
    base_harmonic_frequency=30,
    modulation_frequency=0.5,
    phi_0=0,
    phi_1=0,
    phi_2=0,
    phi_3=0,
    phi_4=0,
    phi_5=0,
    phi_6=0,
    phi_7=0,
    phi_8=0,
    phi_9=0,
    amplitude_0=1e-21,
    amplitude_1=0,
    amplitude_2=0,
    amplitude_3=0,
    amplitude_4=0,
    amplitude_5=0,
    amplitude_6=0,
    amplitude_7=0,
    amplitude_8=0,
    amplitude_9=0,
    **kwargs,
):
    """
    Returns frequency domain strain associated with a set of slow scattering arches

    Parameters
    -----------
    f :
        The frequencies array for the data, should satisfy associated conditions with duration and sampling rate of data
    geocent_time : float
        The geocenter time to center the 0th arch at, and optionally all other arches as well
    base_harmonic_frequency : float
        The harmonic frequency (i.e. maximum frequency reached) for the 0th arch
    harmonic_frequency_delta : float
        The expected difference in harmonic frequency between the ith and i+1th arch
    modulation_frequency : float
        The modulation frequency of the scattering arches
    phi_{i} : float
        {i} runs over 0-9, the phase of the ith arch
    amplitude_{i} : float
        {i} runs over 0-9, the amplitude of the ith arch
    **kwargs
        number_harmonics : int
            Number of arches to loop over - default 1 implies only 0th arch
        duration : int
            Duration of the segment in seconds, should be multiple of 2
        sampling_frequency : int
            The sampling rate of the data, should be a multiple of 2
        time_array :
            Precomputed time array corresponding to the data characteristics, passed for efficiency
        glitch_mode_name : str
            The name to assign the glitch mode - should take the form glitch_{ifo}_{name}

    Returns
    =======
    strain_dict : dict
        Contains 'glitch_mode_name' with the glitch strain, and 'plus' and 'cross' as 0s with the appropriate shape
    """
    # set number of arches to loop over
    if not kwargs.keys():
        n_harmonics = 1
    else:
        n_harmonics = kwargs["number_harmonics"]

    # get kwargs, setup ifo delay for geocent time injection purposes
    duration = kwargs["duration"]
    sampling_frequency = kwargs["sampling_frequency"]
    segment_start_time = kwargs["start_time"]

    time_to_rotate_base = -duration / 2
    # bilby assumes that the segment starts at start time and rotates accordingly
    # see https://git.ligo.org/lscsoft/bilby/-/blob/master/bilby/gw/detector/interferometer.py#L309 through L313
    # we generate in the middle of the segment, so rotate it back to the beginning then let bilby do this

    # if possible preload array
    if "time_array" in kwargs.keys():
        t = kwargs["time_array"]
        if t[0] != 0:
            t -= segment_start_time
    else:
        # make a time space
        t = np.linspace(0, duration, duration * sampling_frequency)

    # get hoft of 0th arch
    # note - generates initially in the center of the time array, to prevent weirdness with the window
    # this will get shifted appropriately later
    hoft_temp = amplitude_0 * np.sin(
        (base_harmonic_frequency / modulation_frequency)
        * np.sin(2 * np.pi * modulation_frequency * (t - duration / 2))
        + phi_0
    )

    for harmonic_idx in range(1, n_harmonics):
        # fetch variables by index
        temp_amp = locals()[f"amplitude_{harmonic_idx}"]
        temp_phi = locals()[f"phi_{harmonic_idx}"]

        temp_f_harm = base_harmonic_frequency + harmonic_idx * harmonic_frequency_delta

        # get hoft for this arch
        hoft_temp += temp_amp * np.sin(
            (temp_f_harm / modulation_frequency)
            * np.sin(2 * np.pi * modulation_frequency * (t - duration / 2))
            + temp_phi
        )

    # Determine and apply window, centered on duration / 2
    # This window is necessary because our equation describes an infinite set of arches
    # But naturally we only want to fit for one
    # Get the window - centered at the arches center, total width of 1 / 2 / fmod0
    window_low = duration / 2 - 0.25 / modulation_frequency
    window_high = duration / 2 + 0.25 / modulation_frequency
    window_idxs = np.argwhere((t > window_low) & (t < window_high))
    tukey_window = tukey(window_idxs.flatten().shape[0], alpha=0.2)
    hoft_temp[window_idxs.flatten()] *= tukey_window
    outside_window_mask = np.zeros(hoft_temp.shape, dtype=bool)
    outside_window_mask[window_idxs] = True
    hoft_temp[~outside_window_mask] = 0

    hoft = hoft_temp

    # FFT our TD waveform
    htilde_temp = nfft(hoft, sampling_frequency)
    # Apply a shift so that it's where bilby expects it to be rather than in the center of the segment
    htilde_shifted = htilde_temp[0] * np.exp(
        -2 * np.pi * 1j * htilde_temp[1] * (time_to_rotate_base)
    )
    htilde = htilde_shifted

    return {kwargs["glitch_mode_name"]: htilde, "plus": htilde * 0, "cross": htilde * 0}


def slow_scattering_original(
    f,
    geocent_time=0,
    f_harm0=30,
    f_mod0=0.5,
    phi0=0,
    amp0=1e-21,
    delta_gc_t1=0,
    delta_f_mod1=0,
    delta_f_harm1=0,
    phi1=0,
    amp1=0,
    delta_gc_t2=0,
    delta_f_mod2=0,
    delta_f_harm2=0,
    phi2=0,
    amp2=0,
    delta_gc_t3=0,
    delta_f_mod3=0,
    delta_f_harm3=0,
    phi3=0,
    amp3=0,
    delta_gc_t4=0,
    delta_f_mod4=0,
    delta_f_harm4=0,
    phi4=0,
    amp4=0,
    delta_gc_t5=0,
    delta_f_mod5=0,
    delta_f_harm5=0,
    phi5=0,
    amp5=0,
    delta_gc_t6=0,
    delta_f_mod6=0,
    delta_f_harm6=0,
    phi6=0,
    amp6=0,
    delta_gc_t7=0,
    delta_f_mod7=0,
    delta_f_harm7=0,
    phi7=0,
    amp7=0,
    delta_gc_t8=0,
    delta_f_mod8=0,
    delta_f_harm8=0,
    phi8=0,
    amp8=0,
    delta_gc_t9=0,
    delta_f_mod9=0,
    delta_f_harm9=0,
    phi9=0,
    amp9=0,
    f_diff=25,
    **kwargs,
):
    """
    Returns frequency domain strain associated with a set of slow scattering arches.
    Has greater flexibility than the base model - this is actually the original model from Udall + Davis 2022

    Parameters
    -----------
    f :
        The frequencies array for the data, should satisfy associated conditions with duration and sampling rate of data
    geocent_time : float
        The geocenter time to center the 0th arch at, and optionally all other arches as well
    f_harm0 : float
        The harmonic frequency (i.e. maximum frequency reached) for the 0th arch
    f_mod0 : float
        The modulation frequency of the 0th arch - 1 / (2 * arch width in seconds)
    phi0 : float
        The phase of the 0th arch
    delta_gc_t{i} : float
        {i} runs over 1-9, the deviation of the ith arch's central time from that of the 0th arch
    delta_f_mod{i} : float
        {i} runs over 1-9, the deviation of the ith arch's modulation frequency from that of the 0th arch
    delta_f_harm{i} : float
        {i} runs over 1-9, the deviation of the ith arch's harmonic frequency
        from that predicted by f_harm0 + i * f_diff
    phi{i} : float
        {i} runs over 1-9, the phase of the ith arch
    amp{i} : float
        {i} runs over 1-9, the amplitude of the ith arch
    f_diff : float
        The expected difference in harmonic frequency between the ith and i+1th arch
    **kwargs
        number_harmonics : int
            Number of arches to loop over - default 1 implies only 0th arch
        duration : int
            Duration of the segment in seconds, should be multiple of 2
        sampling_frequency : int
            The sampling rate of the data, should be a multiple of 2
        centered_arches : bool
            Whether all higher arches are centered at the same time - if True will compute more efficiently
        time_array :
            Precomputed time array corresponding to the data characteristics, passed for efficiency
        glitch_mode_name : str
            The name to assign the glitch mode - should take the form glitch_{ifo}_{name}

    Returns
    =======
    strain_dict : dict
        Contains 'plus' => the compute strain and 'cross' => 0s in the appropriate shame, as a dummy
    """
    # set number of arches to loop over
    if not kwargs.keys():
        n_harmonics = 1
    else:
        n_harmonics = kwargs["number_harmonics"]

    # get kwargs, setup ifo delay for geocent time injection purposes
    duration = kwargs["duration"]
    sampling_frequency = kwargs["sampling_frequency"]
    segment_start_time = kwargs["start_time"]

    time_to_rotate_base = -duration / 2
    # bilby assumes that the segment starts at start time and rotates accordingly
    # see https://git.ligo.org/lscsoft/bilby/-/blob/master/bilby/gw/detector/interferometer.py#L309 through L313
    # we generate in the middle of the segment, so rotate it back to the beginning then let bilby do this

    # default False for centered arches, for robustness
    if "centered_arches" not in kwargs.keys():
        kwargs["centered_arches"] = False

    # if possible preload array
    if "time_array" in kwargs.keys():
        t = kwargs["time_array"]
        if t[0] != 0:
            t -= segment_start_time
    else:
        # make a time space
        t = np.linspace(0, duration, duration * sampling_frequency)

    # get hoft of 0th arch
    # note - generates initially in the center of the time array, to prevent weirdness with the window
    # this will get shifted appropriately later
    hoft_temp = amp0 * np.sin(
        (f_harm0 / f_mod0) * np.sin(2 * np.pi * f_mod0 * (t - duration / 2)) + phi0
    )

    # determine and apply window, centered on duration / 2
    # This window is necessary because our equation describes an infinite set of arches
    # But naturally we only want to fit for one
    # Get the window - centered at the arches center, total width of 1 / 2 / fmod0
    window_low = duration / 2 - 0.25 / f_mod0
    window_high = duration / 2 + 0.25 / f_mod0
    window_idxs = np.argwhere((t > window_low) & (t < window_high))
    tukey_window = tukey(window_idxs.flatten().shape[0], alpha=0.2)
    hoft_temp[window_idxs.flatten()] *= tukey_window
    outside_window_mask = np.zeros(hoft_temp.shape, dtype=bool)
    outside_window_mask[window_idxs] = True
    hoft_temp[~outside_window_mask] = 0

    # FFT and timeshift only if arches aren't centered, else do it at the end
    if kwargs["centered_arches"]:
        hoft = hoft_temp
    else:
        # FFT
        htilde_temp = nfft(hoft_temp, sampling_frequency)

        # A time shift - first, to the beginning / end of the data segment, then backwards to the ifo_time
        htilde_shifted = htilde_temp[0] * np.exp(
            -2 * np.pi * 1j * htilde_temp[1] * (time_to_rotate_base)
        )
        htilde = htilde_shifted

    for harmonic_idx in range(1, n_harmonics):
        # fetch variables by index
        temp_amp = locals()[f"amp{harmonic_idx}"]
        temp_phi = locals()[f"phi{harmonic_idx}"]
        temp_delta_gc_t = locals()[f"delta_gc_t{harmonic_idx}"]
        temp_delta_f_mod = locals()[f"delta_f_mod{harmonic_idx}"]
        temp_delta_f_harm = locals()[f"delta_f_harm{harmonic_idx}"]

        # make temporary variables
        temp_time_to_rotate = time_to_rotate_base + temp_delta_gc_t
        temp_f_mod = temp_delta_f_mod + f_mod0
        temp_f_harm = f_harm0 + harmonic_idx * f_diff + temp_delta_f_harm

        # get hoft for this arch
        hoft_temp = temp_amp * np.sin(
            (temp_f_harm / temp_f_mod)
            * np.sin(2 * np.pi * temp_f_mod * (t - duration / 2))
            + temp_phi
        )

        # make and apply the window for this arch
        window_low = duration / 2 - 0.25 / temp_f_mod
        window_high = duration / 2 + 0.25 / temp_f_mod
        window_idxs = np.argwhere((t > window_low) & (t < window_high))
        tukey_window = tukey(window_idxs.flatten().shape[0], alpha=0.2)
        hoft_temp[window_idxs.flatten()] *= tukey_window
        outside_window_mask = np.zeros(hoft_temp.shape, dtype=bool)
        outside_window_mask[window_idxs] = True
        hoft_temp[~outside_window_mask] = 0

        # same FFT procedure
        if kwargs["centered_arches"]:
            hoft += hoft_temp
        else:
            htilde_temp = nfft(hoft_temp, sampling_frequency)
            htilde_shifted = htilde_temp[0] * np.exp(
                -2 * np.pi * 1j * htilde_temp[1] * (temp_time_to_rotate)
            )
            htilde += htilde_shifted

    # same FFT procedure, in this case applied to all arches at once
    if kwargs["centered_arches"]:
        htilde_temp = nfft(hoft, sampling_frequency)
        htilde_shifted = htilde_temp[0] * np.exp(
            -2 * np.pi * 1j * htilde_temp[1] * (time_to_rotate_base)
        )
        htilde = htilde_shifted

    return {kwargs["glitch_mode_name"]: htilde, "plus": htilde * 0, "cross": htilde * 0}


def slow_scattering_shifting_frequency(
    f,
    geocent_time=0,
    harmonic_frequency_delta=25,
    base_harmonic_frequency=30,
    modulation_frequency=0.5,
    modulation_frequency_shift=1,
    taper_central_time=0,
    taper_time_scale=0.1,
    phi_0=0,
    phi_1=0,
    phi_2=0,
    phi_3=0,
    phi_4=0,
    phi_5=0,
    phi_6=0,
    phi_7=0,
    phi_8=0,
    phi_9=0,
    amplitude_0=1e-21,
    amplitude_1=0,
    amplitude_2=0,
    amplitude_3=0,
    amplitude_4=0,
    amplitude_5=0,
    amplitude_6=0,
    amplitude_7=0,
    amplitude_8=0,
    amplitude_9=0,
    **kwargs,
):
    """
    Returns frequency domain strain associated with a set of slow scattering arches

    Parameters
    -----------
    f : np.array
        The frequencies array for the data, should satisfy associated conditions with duration and sampling rate of data
    geocent_time : float
        The geocenter time to center the 0th arch at, and optionally all other arches as well
    base_harmonic_frequency : float
        The harmonic frequency (i.e. maximum frequency reached) for the 0th arch
    harmonic_frequency_delta : float
        The expected difference in harmonic frequency between the ith and i+1th arch
    modulation_frequency : float
        The modulation frequency of the scattering arches
    modulation_frequency_shift : float
        The ratio of final modulation frequency to initial modulation frequency
    taper_central_time : float
        The central time of the logistic tapering function
    taper_time_scale : float
        The characteristic time scale of the logistic tapering function
    phi_{i} : float
        {i} runs over 0-9, the phase of the ith arch
    amplitude_{i} : float
        {i} runs over 0-9, the amplitude of the ith arch
    **kwargs
        number_harmonics : int
            Number of arches to loop over - default 1 implies only 0th arch
        duration : int
            Duration of the segment in seconds, should be multiple of 2
        sampling_frequency : int
            The sampling rate of the data, should be a multiple of 2
        time_array :
            Precomputed time array corresponding to the data characteristics, passed for efficiency
        glitch_mode_name : str
            The name to assign the glitch mode - should take the form glitch_{ifo}_{name}

    Returns
    =======
    strain_dict : dict
        Contains 'glitch_mode_name' with the glitch strain, and 'plus' and 'cross' as 0s with the appropriate shape
    """
    # set number of arches to loop over
    if not kwargs.keys():
        n_harmonics = 1
    else:
        n_harmonics = kwargs["number_harmonics"]

    # get kwargs, setup ifo delay for geocent time injection purposes
    duration = kwargs["duration"]
    sampling_frequency = kwargs["sampling_frequency"]
    segment_start_time = kwargs["start_time"]

    time_to_rotate_base = -duration / 2
    # bilby assumes that the segment starts at start time and rotates accordingly
    # see https://git.ligo.org/lscsoft/bilby/-/blob/master/bilby/gw/detector/interferometer.py#L309 through L313
    # we generate in the middle of the segment, so rotate it back to the beginning then let bilby do this

    # if possible preload array
    if "time_array" in kwargs.keys():
        t = kwargs["time_array"]
        if t[0] != 0:
            t -= segment_start_time
    else:
        # make a time space
        t = np.linspace(0, duration, duration * sampling_frequency)

    f_of_t = slow_scattering_shifting_modulation_f_of_t(
        t=t,
        central_time=duration / 2,
        modulation_frequency=modulation_frequency,
        modulation_frequency_shift=modulation_frequency_shift,
        harmonic_frequency=base_harmonic_frequency,
        taper_central_time=taper_central_time,
        taper_time_scale=taper_time_scale,
    )

    base_phi_of_t = 2 * np.pi * cumtrapz(f_of_t, t)
    base_phi_of_t = np.concatenate([np.zeros(1), base_phi_of_t])

    # get hoft of 0th arch
    # note - generates initially in the center of the time array, to prevent weirdness with the window
    # this will get shifted appropriately later
    hoft_temp = amplitude_0 * np.sin(base_phi_of_t + phi_0)

    for harmonic_idx in range(1, n_harmonics):
        # fetch variables by index
        temp_amp = locals()[f"amplitude_{harmonic_idx}"]
        temp_phi = locals()[f"phi_{harmonic_idx}"]

        temp_f_harm = base_harmonic_frequency + harmonic_idx * harmonic_frequency_delta

        # get hoft for this arch
        hoft_temp += temp_amp * np.sin(
            (temp_f_harm / base_harmonic_frequency) * base_phi_of_t + temp_phi
        )

    # Determine and apply window, centered on duration / 2
    # This window is necessary because our equation describes an infinite set of arches
    # But naturally we only want to fit for one
    # Get the window - centered at the arches center, total width of 1 / 2 / fmod0
    window_low = duration / 2 - 0.25 / modulation_frequency
    window_high = duration / 2 + 0.25 / modulation_frequency
    window_idxs = np.argwhere((t > window_low) & (t < window_high))
    tukey_window = tukey(window_idxs.flatten().shape[0], alpha=0.2)
    hoft_temp[window_idxs.flatten()] *= tukey_window
    outside_window_mask = np.zeros(hoft_temp.shape, dtype=bool)
    outside_window_mask[window_idxs] = True
    hoft_temp[~outside_window_mask] = 0

    hoft = hoft_temp

    # FFT our TD waveform
    htilde_temp = nfft(hoft, sampling_frequency)
    # Apply a shift so that it's where bilby expects it to be rather than in the center of the segment
    htilde_shifted = htilde_temp[0] * np.exp(
        -2 * np.pi * 1j * htilde_temp[1] * (time_to_rotate_base)
    )
    htilde = htilde_shifted

    return {kwargs["glitch_mode_name"]: htilde, "plus": htilde * 0, "cross": htilde * 0}


def fast_scattering(
    f,
    amplitude=1e-21,
    microseism_coefficient=50,
    microseism_frequency=0.3,
    anthropogenic_coefficient=7,
    anthropogenic_frequency=5,
    geocent_time=0,
    phi=0,
    **kwargs,
):
    # get kwargs, setup ifo delay for geocent time injection purposes
    duration = kwargs["duration"]
    sampling_frequency = kwargs["sampling_frequency"]
    segment_start_time = kwargs["start_time"]

    time_to_rotate_base = -duration / 2
    # bilby assumes that the segment starts at start time and rotates accordingly
    # see https://git.ligo.org/lscsoft/bilby/-/blob/master/bilby/gw/detector/interferometer.py#L309 through L313
    # we generate in the middle of the segment, so rotate it back to the beginning then let bilby do this

    # if possible preload array
    if "time_array" in kwargs.keys():
        t = kwargs["time_array"]
        if t[0] != 0:
            t -= segment_start_time
    else:
        # make a time space
        t = np.linspace(0, duration, duration * sampling_frequency)

    xoft = microseism_coefficient / microseism_frequency * np.sin(
        2 * np.pi * microseism_frequency * (t - duration / 2)
    ) + anthropogenic_coefficient / anthropogenic_frequency * np.sin(
        2 * np.pi * anthropogenic_frequency * (t - duration / 2)
    )
    hoft = amplitude * np.sin(xoft + phi)

    # determine and apply window, centered on duration / 2
    # This window is necessary because our equation describes an infinite set of arches
    # But naturally we only want to fit for one
    # Get the window - centered at the arches center, total width of 1 / 2 / fmod0
    fwindow = kwargs.get("window_frequency", microseism_frequency)
    window_low = duration / 2 - 0.25 / fwindow
    window_high = duration / 2 + 0.25 / fwindow
    window_idxs = np.argwhere((t > window_low) & (t < window_high))
    tukey_window = tukey(window_idxs.flatten().shape[0], alpha=0.2)
    hoft[window_idxs.flatten()] *= tukey_window
    outside_window_mask = np.zeros(hoft.shape, dtype=bool)
    outside_window_mask[window_idxs] = True
    hoft[~outside_window_mask] = 0

    # FFT
    htilde_temp = nfft(hoft, sampling_frequency)

    # A time shift - first, to the beginning / end of the data segment, then backwards to the ifo_time
    htilde_shifted = htilde_temp[0] * np.exp(
        -2 * np.pi * 1j * htilde_temp[1] * (time_to_rotate_base)
    )
    htilde = htilde_shifted

    return {kwargs["glitch_mode_name"]: htilde, "plus": htilde * 0, "cross": htilde * 0}


################################################################################
################################################################################
##########################    Ingested Data Models    ##########################
################################################################################
################################################################################


def aux_informed(
    f: np.ndarray,
    amp0: float = 0,
    amp1: float = 0,
    amp2: float = 0,
    amp3: float = 0,
    amp4: float = 0,
    amp5: float = 0,
    amp6: float = 0,
    amp7: float = 0,
    amp8: float = 0,
    amp9: float = 0,
    phi0: float = 0,
    phi1: float = 0,
    phi2: float = 0,
    phi3: float = 0,
    phi4: float = 0,
    phi5: float = 0,
    phi6: float = 0,
    phi7: float = 0,
    phi8: float = 0,
    phi9: float = 0,
    scale: float = 1,
    geocent_time: float = 0,
    **kwargs,
):
    """Get the strain data for a scattering arch, as predicted by a witness channel w/ arbitrary phase and amplitude coupling

    Parameters
    ==========
    f : np.ndarray
        The frequencies for which the strain should be computed
    amp{i} : float
        Runs over i=0,1,...9, the amplitude for the ith arch
    phi{i} : float
        Runs over i=0,1,...9, the phase for the ith arch
    scale : float
        The scale factor controlling the frequency distance between arches
    kwargs:
        number_harmonics : int
            Number of arches to loop over - default 1 implies only 0th arch
        duration : int
            Duration of the segment in seconds, should be multiple of 2
        sampling_frequency : int
            The sampling rate of the data, should be a multiple of 2
        time_array :
            Precomputed time array corresponding to the data characteristics, passed for efficiency
        glitch_mode_name : str
            The name to assign the glitch mode - should take the form glitch_{ifo}_{name}
        aux_time_series : `gwpy.timeseries.TimeSeries`
            The time series from the auxiliary data channel
            When called by GlitchWaveformGenerator, pass the channel name as aux_channel

    Returns
    =======
    strain_dict : dict
        Contains 'plus' => the compute strain and 'cross' => 0s in the appropriate shame, as a dummy
    """

    # set number of arches to loop over
    if not kwargs.keys():
        n_harmonics = 1
    else:
        n_harmonics = kwargs["number_harmonics"]

    # get kwargs, setup ifo delay for geocent time injection purposes
    duration = kwargs["duration"]
    sampling_frequency = kwargs["sampling_frequency"]
    segment_start_time = kwargs["start_time"]
    aux_time_series = kwargs["aux_time_series"]
    post_trigger_duration = kwargs["post_trigger_duration"]

    # if possible preload array
    if "time_array" in kwargs.keys():
        t = kwargs["time_array"]
        if t[0] != 0:
            t -= segment_start_time
    else:
        # make a time space
        t = np.linspace(0, duration, duration * sampling_frequency)

    # The duration of the aux channel needs to match
    assert (
        aux_time_series.duration.value == duration
    ), f"Witness data duration is {aux_time_series.duration.value}, but analysis duration is {duration}"
    # Aux channels are probably sampled at less than our rate, so resample
    aux_time_series_resampled = aux_time_series.copy().resample(sampling_frequency)
    # Make goft
    goft = np.zeros(t.shape)
    f_of_t = auxiliary_f_of_t(aux_time_series_resampled, scale)
    # Add up each arch
    base_phi_of_t = 2 * np.pi * cumtrapz(f_of_t, t)
    base_phi_of_t = np.concatenate([np.zeros(1), base_phi_of_t])

    for harmonic_idx in range(1, 1 + n_harmonics):
        # fetch variables by index
        temp_amp = locals()[f"amp{harmonic_idx - 1}"]
        temp_phi = locals()[f"phi{harmonic_idx - 1}"]
        goft += temp_amp * np.sin(harmonic_idx * base_phi_of_t + temp_phi)

    goft *= tukey(goft.shape[0], alpha=0.2)

    gtilde, frequencies = nfft(goft, sampling_frequency=sampling_frequency)
    gtilde *= np.exp(-2 * np.pi * 1j * frequencies * post_trigger_duration)

    return {kwargs["glitch_mode_name"]: gtilde, "plus": gtilde * 0, "cross": gtilde * 0}


def aux_informed_amplitude_splines(
    f: np.ndarray,
    amp0: float = 0,
    amp1: float = 0,
    amp2: float = 0,
    amp3: float = 0,
    amp4: float = 0,
    amp5: float = 0,
    amp6: float = 0,
    amp7: float = 0,
    amp8: float = 0,
    amp9: float = 0,
    phi0: float = 0,
    phi1: float = 0,
    phi2: float = 0,
    phi3: float = 0,
    phi4: float = 0,
    phi5: float = 0,
    phi6: float = 0,
    phi7: float = 0,
    phi8: float = 0,
    phi9: float = 0,
    spline0: float = 1,
    spline1: float = 1,
    spline2: float = 1,
    spline3: float = 1,
    spline4: float = 1,
    spline5: float = 1,
    spline6: float = 1,
    spline7: float = 1,
    spline8: float = 1,
    spline9: float = 1,
    scale: float = 1,
    geocent_time: float = 0,
    **kwargs,
):
    """Get the strain data for a scattering arch, as predicted by a witness channel w/ arbitrary phase and amplitude coupling

    Parameters
    ==========
    f : np.ndarray
        The frequencies for which the strain should be computed
    amp{i} : float
        Runs over i=0,1,...9, the amplitude for the ith arch
    phi{i} : float
        Runs over i=0,1,...9, the phase for the ith arch
    scale : float
        The scale factor controlling the frequency distance between arches
    kwargs:
        number_harmonics : int
            Number of arches to loop over - default 1 implies only 0th arch
        duration : int
            Duration of the segment in seconds, should be multiple of 2
        sampling_frequency : int
            The sampling rate of the data, should be a multiple of 2
        time_array :
            Precomputed time array corresponding to the data characteristics, passed for efficiency
        glitch_mode_name : str
            The name to assign the glitch mode - should take the form glitch_{ifo}_{name}
        aux_time_series : `gwpy.timeseries.TimeSeries`
            The time series from the auxiliary data channel
            When called by GlitchWaveformGenerator, pass the channel name as aux_channel
        number_splines : int
            The number of amplitude splines to place, between 2 and 10
        spline_interpolation_times : np.ndarray
            The array of times to place the splines at - they should be evenly spaced over the duration.
            If this isn't passed then this function will do compute it
            *which is EXTREMELY inefficient*!

    Returns
    =======
    strain_dict : dict
        Contains 'plus' => the compute strain and 'cross' => 0s in the appropriate shame, as a dummy
    """

    # set number of arches to loop over
    if not kwargs.keys():
        n_harmonics = 1
    else:
        n_harmonics = kwargs["number_harmonics"]

    number_splines = kwargs["number_splines"]

    # get kwargs, setup ifo delay for geocent time injection purposes
    duration = kwargs["duration"]
    sampling_frequency = kwargs["sampling_frequency"]
    segment_start_time = kwargs["start_time"]
    aux_time_series = kwargs["aux_time_series"]

    # if possible preload array
    if "time_array" in kwargs.keys():
        t = kwargs["time_array"]
        if t[0] != 0:
            t -= segment_start_time
    else:
        # make a time space
        t = np.linspace(0, duration, duration * sampling_frequency)

    if "spline_interpolation_times" in kwargs.keys():
        interp_times = kwargs["spline_interpolation_times"]
    else:
        interp_times = np.linspace(0, duration, kwargs["number_splines"])

    # The duration of the aux channel needs to match
    assert (
        aux_time_series.duration.value == duration
    ), f"Witness data duration is {aux_time_series.duration.value}, but analysis duration is {duration}"
    # Aux channels are probably sampled at less than our rate, so resample
    aux_time_series_resampled = aux_time_series.copy().resample(sampling_frequency)
    # Make goft
    goft = np.zeros(t.shape)
    # Add up each arch
    f_of_t = auxiliary_f_of_t(aux_time_series_resampled, scale)
    # Add up each arch
    base_phi_of_t = 2 * np.pi * cumtrapz(f_of_t, t)
    base_phi_of_t = np.concatenate([np.zeros(1), base_phi_of_t])

    for harmonic_idx in range(1, 1 + n_harmonics):
        # fetch variables by index
        temp_amp = locals()[f"amp{harmonic_idx - 1}"]
        temp_phi = locals()[f"phi{harmonic_idx - 1}"]
        goft += temp_amp * np.sin(harmonic_idx * base_phi_of_t + temp_phi)

    amplitude_spline_values = np.zeros((number_splines,))
    for ii in range(number_splines):
        # This is so stupid I wish I knew a better way
        # TODO figure out a better way
        amplitude_spline_values[ii] = locals()[f"spline{ii}"]
    amplitude_spline_interp = np.interp(t, interp_times, amplitude_spline_values)

    logger.info(amplitude_spline_interp)

    goft *= amplitude_spline_interp

    goft *= tukey(goft.shape[0], alpha=0.2)

    gtilde, frequencies = nfft(goft, sampling_frequency=sampling_frequency)
    gtilde *= np.exp(-2 * np.pi * 1j * frequencies * (duration / 2))

    return {kwargs["glitch_mode_name"]: gtilde, "plus": gtilde * 0, "cross": gtilde * 0}


def posterior_waveforms(f: np.ndarray, draw: int, **kwargs):
    """Draw a realization from a posterior of glitch realizations, computed in some previous Bayesian inference process.

    Parameters
    ==========
    f : np.ndarray
        The frequency array for the data
    draw : int
        The index of the draw to use
    **kwargs:
        duration : int
            Duration of the segment in seconds, should be multiple of 2
        sampling_frequency : int
            The sampling rate of the data, should be a multiple of 2
        time_array : np.ndarray
            Precomputed time array corresponding to the data characteristics, passed for efficiency
        bayeswave_realizations : np.ndarray
            Precomputed glitch realizations to draw from
        glitch_mode_name : str
            The name to assign the glitch mode - should take the form glitch_{ifo}_{name}

    Returns
    =======
    strain_dict : dict
        Contains 'glitch_mode_name' with the glitch strain, and 'plus' and 'cross' as 0s with the appropriate shape
    """

    # bayeswave_realizations = kwargs["bayeswave_realizations"]
    # Get a specific FD realization from the pre-loaded set of all realizations
    # glitch_realization = bayeswave_realizations[draw, :]

    # Do some time/frequency manipulation garbage to appease bilby, maybe

    # return {kwargs["glitch_mode_name"]: glitch_realization,
    # "plus": glitch_realization * 0, "cross": glitch_realization * 0}
    pass
