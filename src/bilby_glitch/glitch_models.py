import logging
import numpy as np
import copy
from scipy.signal.windows import tukey
from bilby.core.utils import nfft

logger = logging.getLogger(__name__)

SLOW_SCATTERING_PARAMETERS = [
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


def populate_unused_parameters(
    original_dict: dict,
    full_parameter_list: list = SLOW_SCATTERING_PARAMETERS,
    add_sky_params=True,
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


def slow_scattering(
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
    Returns frequency domain strain associated with a set of slow scattering arches

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


def _blank_frequency_domain_source_model(f, geocent_time, **kwargs):
    """A placeholder for when no CBC is passed to a joint analysis"""
    return {"plus": np.zeros(f.shape), "cross": np.zeros(f.shape)}


def fast_scattering(
    f, A=1e-21, C1=50, f1=0.3, C2=7, f2=5, geocent_time=0, phi=0, **kwargs
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

    xoft = C1 * np.sin(2 * np.pi * f1 * (t - duration / 2)) + C2 * np.sin(
        2 * np.pi * f2 * (t - duration / 2)
    )
    hoft = A * np.sin(xoft + phi)

    # determine and apply window, centered on duration / 2
    # This window is necessary because our equation describes an infinite set of arches
    # But naturally we only want to fit for one
    # Get the window - centered at the arches center, total width of 1 / 2 / fmod0
    fwindow = min(f1, f2)
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
    # Add up each arch

    for harmonic_idx in range(1, 1 + n_harmonics):
        # fetch variables by index
        temp_amp = locals()[f"amp{harmonic_idx - 1}"]
        temp_phi = locals()[f"phi{harmonic_idx - 1}"]
        goft += temp_amp * np.sin(
            4 * np.pi / 1.064 * scale * harmonic_idx * aux_time_series_resampled.value
            + temp_phi
        )

    goft *= tukey(goft.shape[0], alpha=0.2)

    gtilde, frequencies = nfft(goft, sampling_frequency=sampling_frequency)
    gtilde *= np.exp(-2 * np.pi * 1j * frequencies * (duration / 2))

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

    for harmonic_idx in range(1, 1 + n_harmonics):
        # fetch variables by index
        temp_amp = locals()[f"amp{harmonic_idx - 1}"]
        temp_phi = locals()[f"phi{harmonic_idx - 1}"]
        goft += temp_amp * np.sin(
            4 * np.pi / 1.064 * scale * harmonic_idx * aux_time_series_resampled.value
            + temp_phi
        )

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


def anti_glitch(
    f: np.ndarray, A: float, phi: float, f0: float, gamma: float, t0: float, **kwargs
):
    """Implement the antiglitch model for short duration glitches
    arXiv:2309.06594 [astro-ph, physics:gr-qc]

    Parameters
    ==========
    f : np.ndarray,
        The array of frequencies to evaluate the waveform at
    A : float
        The amplitude of the glitch
        Note that we don't include the normalization factor used in Bondarescu+
        Instead this just has range ~10^-21
    phi : float
        The overall phase of the glitch
    f0 : float
        The peak frequency of the glitch
    gamma : float
        The damping constant of the damping part of the glitch
    t0 : float
        The central time of the glitch

    Returns
    =======
    """

    # get kwargs, setup ifo delay for geocent time injection purposes
    duration = kwargs["duration"]

    time_to_rotate_base = -duration / 2
    # bilby assumes that the segment starts at start time and rotates accordingly
    # see https://git.ligo.org/lscsoft/bilby/-/blob/master/bilby/gw/detector/interferometer.py#L309 through L313
    # we generate in the middle of the segment, so rotate it back to the beginning then let bilby do this

    htilde_temp = (
        A
        * np.exp(-gamma * (np.log(f) - np.log(f0)) ** 2 / 2)
        * np.exp(1j * phi - 2 * np.pi * 1j * f * t0)
    )

    # A time shift - first, to the beginning / end of the data segment, then backwards to the ifo_time
    htilde = htilde_temp * np.exp(-2 * np.pi * 1j * f * (time_to_rotate_base))

    return {kwargs["glitch_mode_name"]: htilde, "plus": htilde * 0, "cross": htilde * 0}
