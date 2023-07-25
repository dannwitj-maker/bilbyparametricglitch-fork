import logging
import numpy as np
from scipy.signal.windows import tukey
from bilby.core.utils import nfft

logger = logging.getLogger(__name__)


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
            The name to assign the glitch mode - should take the form
    Returns
    ------------
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
