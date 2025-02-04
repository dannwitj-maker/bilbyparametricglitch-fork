import jax.typing as jnpt
from functools import partial

from .series_transforms import taper_off, taper_on

from ..backend_utilities.backend import _configure_numpy_and_scipy

import numpy as np
import scipy as scs

xp = np
_configure_numpy_and_scipy(xp, scs)

################################################################################
################################################################################
##########################    JAX Strain Functions    ##########################
################################################################################
################################################################################


def slow_scattering_phi_of_t_array(
    t: jnpt.ArrayLike,
    sampling_parameters: jnpt.ArrayLike,
    groups_left_of_center: int = 0,
    groups_right_of_center: int = 0,
) -> jnpt.ArrayLike:
    """Produces the frequency as a function
    of time for each arch in a single group.
    This will always produce a 10xN array,
    with N being the length of the timeseries.
    10 is assumed as the nominal maximum number
    of arches.

    Parameters
    ==========
        t : jnpt.ArrayLike
            The times for which to evaluate the
            frequencies, of shape (N,)
        sampling_parameters : jnpt.ArrayLike
            [0] is the central time, which sets
            the time at which the arches peak
            [1] is the modulation_frequency
            which sets the width of the arches
            [2] is the base_harmonic_frequency
            which sets the peak frequency of the
            bottom arch
            [3] is the harmonic_frequency_delta,
            which sets the space between arches
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

    # Get all peak harmonic frequencies
    f_harm_k = xp.arange(10) * sampling_parameters[3] + sampling_parameters[2]
    # Get baseline phi(t) behavior i.e. the appropriate sinusoid for all arches
    modulation_of_t = xp.sin(
        2 * xp.pi * sampling_parameters[1] * (t - sampling_parameters[0])
    )
    # Get correctly scaled phi(t) i.e. the indefinite integral of d phi / d t
    # For each arch
    phi_of_t = xp.outer(f_harm_k, modulation_of_t) / sampling_parameters[1]
    # Add respective phases for each (i.e. setting boundary conditions on integral)
    phi_of_t += sampling_parameters[4:14][:, xp.newaxis]

    # Get an array which is just the minimum of each row, with shape of phi_of_t
    minimums_array = xp.repeat(
        xp.min(phi_of_t, axis=1)[:, xp.newaxis], phi_of_t.shape[1], axis=1
    )
    # Get the corresponding maximums array
    maximums_array = xp.repeat(
        xp.max(phi_of_t, axis=1)[:, xp.newaxis], phi_of_t.shape[1], axis=1
    )
    # Get the condition where we cut out all earlier arches
    # This is the left side of the central arch (1/4 of modulation frequency)
    # Then the full width of the other arches (each 1/2 of modulation frequency)
    lower_window_condition_array = xp.repeat(
        (
            t
            < sampling_parameters[0]
            - (1 / 4 + groups_left_of_center / 2) / sampling_parameters[1]
        )[xp.newaxis, :],
        10,
        axis=0,
    )
    # Get the corresponding upper condition
    upper_window_condition_array = xp.repeat(
        (
            t
            > sampling_parameters[0]
            + (1 / 4 + groups_right_of_center / 2) / sampling_parameters[1]
        )[xp.newaxis, :],
        10,
        axis=0,
    )

    # Decide what to fill the left window with.
    # If an even number of arches to left, it ends on a minimum, else on a maximum
    lower_window_fill = xp.where(
        xp.mod(groups_left_of_center, 2), maximums_array, minimums_array
    )
    # As above but inverted
    upper_window_fill = xp.where(
        xp.mod(groups_right_of_center, 2), minimums_array, maximums_array
    )

    # Actually set the windowed values
    windowed_phi_of_t = xp.where(
        lower_window_condition_array, lower_window_fill, phi_of_t
    )
    windowed_phi_of_t = xp.where(
        upper_window_condition_array, upper_window_fill, windowed_phi_of_t
    )
    return windowed_phi_of_t


def slow_scattering_g_of_t_array(
    t: jnpt.ArrayLike,
    sampling_parameters: jnpt.ArrayLike,
    number_of_arches: int = 1,
    groups_left_of_center: int = 0,
    groups_right_of_center: int = 0,
    **kwargs,
):
    """A model for slow scattering g(t) in the detector
    when the glitch contains N arches,
    where N is set by the

    Parameters
    =========
        t : jnpt.ArrayLike
            The times for which g(t) is being
            computed.
        sampling_parameters : jnpt.ArrayLike
            [0] is the central time, which sets
            the time at which the arches peak
            [1] is the modulation_frequency
            which sets the width of the arches
            [2] is the base_harmonic_frequency
            which sets the peak frequency of the
            bottom arch
            [3] is the harmonic_frequency_delta,
            which sets the space between arches
            [4:14] are the respective phases for
            each arch
            [14:24] are the respective amplitudes
            for each arch
    """

    arches_mask = xp.arange(number_of_arches)
    arches_phi_of_t = slow_scattering_phi_of_t_array(
        t,
        sampling_parameters[0:14],
        groups_left_of_center=groups_left_of_center,
        groups_right_of_center=groups_right_of_center,
    )

    arches_amplitudes = sampling_parameters[14:][:, xp.newaxis]
    arches_g_of_t_unmasked = xp.sin(arches_phi_of_t) * arches_amplitudes
    arches_g_of_t_masked = arches_g_of_t_unmasked[arches_mask]
    g_of_t = xp.sum(arches_g_of_t_masked, axis=0)

    return g_of_t


def slow_scattering_g_of_f_array(
    f: jnpt.ArrayLike,
    sampling_parameters: jnpt.ArrayLike,
    sampling_frequency: int,
    duration: float,
    start_time: float,
    number_of_arches: int = 1,
    groups_left_of_center: int = 0,
    groups_right_of_center: int = 0,
):
    """A model for slow scattering g(t) in the detector
    when the glitch contains N arches,
    where N is set by the

    Parameters
    =========
        f : jnpt.ArrayLike
            The frequencies for which g(f) is being
            computed.
        sampling_parameters : jnpt.ArrayLike
            [0] is the central time, which sets
            the time at which the arches peak
            [1] is the modulation_frequency
            which sets the width of the arches
            [2] is the base_harmonic_frequency
            which sets the peak frequency of the
            bottom arch
            [3] is the harmonic_frequency_delta,
            which sets the space between arches
            [4:14] are the respective phases for
            each arch
            [14:24] are the respective amplitudes
            for each arch
    """
    t = xp.linspace(start_time, duration + start_time, duration * sampling_frequency)
    time_domain_parametric_glitch = slow_scattering_g_of_t_array(
        t,
        sampling_parameters,
        number_of_arches=number_of_arches,
        groups_left_of_center=groups_left_of_center,
        groups_right_of_center=groups_right_of_center,
    )
    amplitude_tukey = scs.tukey(duration * sampling_frequency, alpha=0.2)
    return (
        xp.fft.rfft(time_domain_parametric_glitch * amplitude_tukey)
        / sampling_frequency
    )


def wavelet_g_of_f_array(
    f,
    sampling_parameters,
    start_time: float = 0,
):
    """
    Parameters
    ==========
    parameters : xp.array
        index 0 -> central_frequency
        index 1 -> central_time
        index 2 -> quality_factor
        index 3 -> amplitude
        index 4 -> reference_phase
    """
    time = sampling_parameters[1] - start_time
    tau = sampling_parameters[2] / (2 * xp.pi * sampling_parameters[0])
    A = sampling_parameters[3]

    const = A * tau * xp.sqrt(xp.pi) / 2
    decay_term = xp.exp(-(tau**2) * xp.pi**2 * (f - sampling_parameters[0]) ** 2)

    Q_decay_term = xp.exp(-sampling_parameters[2] ** 2 * f / sampling_parameters[0])
    plus_term = xp.exp(-1j * (2 * xp.pi * time * f - sampling_parameters[4]))
    minus_term = xp.exp(-1j * (2 * xp.pi * time * f + sampling_parameters[4]))

    return const * decay_term * (plus_term + minus_term * Q_decay_term)


def fast_scattering_phi_of_t_array(
    t: jnpt.ArrayLike,
    sampling_parameters: jnpt.ArrayLike,
    scattering_lower_window: float,
    scattering_upper_window: float,
) -> jnpt.ArrayLike:
    """Generates $\\phi(t)$ for a fast scattering model,
    with windowing applied on the frequency evolution.
    In analytic terms this would be
    $\\phi(t) = a_1 sin(2 \\pi f_1 (t - t_1))
    + a_2 sin(2 \\pi f_2 (t - t_2)) + \\phi_0$
    However, for windowing purposes this is actually generated as
    $\\phi(t) = (2 \\pi) (W * \\int f(t) dt) + \\phi_0$
    For the window function W

    Parameters
    ==========
        t : jnpt.ArrayLike
            The times for which to evaluate the
            frequencies, of shape (N,)
        sampling_parameters : jnpt.ArrayLike
            The parameters to generate the frequency evolution
            of the fast scattering glitch.
            [0] is t_1, the central time associated with the first
            frequency, which to establish convention will be
            the lower (microseism) frequency. This will also    ```
            be treated as the geocent_time for purposes of
            bilby compatibility
            [1] is t_2, the central time associated with the second
            frequency, which to establish convention will be
            the higher (anthropogenic) frequency
            [2] is f_1, the first frequency, which we'll assume
            for sake of convention is the lower
            (microseism) frequency
            [3] is f_2, the second frequency, which we'll assume
            for the sake of convention is the higher
            (anthropogenic) frequency
            [4] is c_1, the first frequency coefficient, associated
            to the lower frequency f_1
            [5] is c_2, the second frequency coefficient, associated
            to the higher frequency f_2
            [6] is phi, the phase offset of the integrated frequency evolution.
        scattering_lower_window : float
            The start time of the windowing function.
            All times outside this window but in the data segment will be set
            to 0.
        scattering_upper_window : float
            The end time of the windowing function

    Returns
    =======
        jnpt.ArrayLike
            The phi(t) associated with the fast scattering
    """
    f_of_t = fast_scattering_f_of_t_array(t, sampling_parameters[:-1])

    # Here begins something contrived
    # The goal is to
    below_window_and_zero_crossing = (xp.isclose(f_of_t, 0, atol=0.5)) & (
        t <= scattering_lower_window
    )
    indices_below_window_and_zero_crossing = xp.where(
        below_window_and_zero_crossing,
        xp.indices(t.shape).flatten(),
        xp.zeros_like(t),
    )
    above_window_and_zero_crossing = (xp.isclose(f_of_t, 0, atol=0.5)) & (
        t >= scattering_upper_window
    )
    indices_above_window_and_zero_crossing = xp.where(
        above_window_and_zero_crossing,
        xp.indices(t.shape).flatten(),
        xp.ones_like(t) * (t.shape[0] + 1),
    )

    index_last_crossing_before_window = xp.argmax(
        indices_below_window_and_zero_crossing
    )
    index_first_crossing_after_window = xp.argmin(
        indices_above_window_and_zero_crossing
    )

    windowed_fast_scattering_f_of_t = xp.where(
        (xp.indices(t.shape) <= index_last_crossing_before_window)
        | (xp.indices(t.shape) >= index_first_crossing_after_window),
        xp.zeros_like(t),
        f_of_t,
    )

    phi_of_t = (
        scs.cumulative_trapezoid(
            windowed_fast_scattering_f_of_t.flatten(), t, initial=0
        )
        * 2
        * xp.pi
        + sampling_parameters[6]
    )
    return phi_of_t


def fast_scattering_g_of_t_array(
    t: jnpt.ArrayLike,
    sampling_parameters: jnpt.ArrayLike,
    scattering_lower_window: float,
    scattering_upper_window: float,
) -> jnpt.ArrayLike:
    """

    Parameters
    ==========
        t : jnpt.ArrayLike
            The times for which to evaluate the
            frequencies, of shape (N,)
        sampling_parameters : jnpt.ArrayLike
            The parameters to generate the frequency evolution
            of the fast scattering glitch.
            [0] is t_1, the central time associated with the first
            frequency, which to establish convention will be
            the lower (microseism) frequency. This will also    ```
            be treated as the geocent_time for purposes of
            bilby compatibility
            [1] is t_2, the central time associated with the second
            frequency, which to establish convention will be
            the higher (anthropogenic) frequency
            [2] is f_1, the first frequency, which we'll assume
            for sake of convention is the lower
            (microseism) frequency
            [3] is f_2, the second frequency, which we'll assume
            for the sake of convention is the higher
            (anthropogenic) frequency
            [4] is c_1, the first frequency coefficient, associated
            to the lower frequency f_1
            [5] is c_2, the second frequency coefficient, associated
            to the higher frequency f_2
            [6] is phi, the phase offset of the integrated frequency evolution.
            [7] is A, the amplitude of the scattering signal
        scattering_lower_window : float
            The start time of the windowing function.
            All times outside this window but in the data segment will be set
            to 0.
        scattering_upper_window : float
            The end time of the windowing function

    Returns
    =======
    jnpt.ArrayLike
        The glitch signal for this fast scattering glitch
    """
    phi_of_t = fast_scattering_phi_of_t_array(
        t,
        sampling_parameters[:-1],
        scattering_lower_window=scattering_lower_window,
        scattering_upper_window=scattering_upper_window,
    )
    g_of_t = sampling_parameters[7] * xp.sin(phi_of_t)

    return g_of_t


def fast_scattering_g_of_f_array(
    f: jnpt.ArrayLike,
    sampling_parameters: jnpt.ArrayLike,
    scattering_lower_window: float,
    scattering_upper_window: float,
    sampling_frequency: int,
    duration: float,
    start_time: float,
) -> jnpt.ArrayLike:
    t = xp.linspace(start_time, duration + start_time, duration * sampling_frequency)
    time_domain_parametric_glitch = fast_scattering_g_of_t_array(
        t,
        sampling_parameters=sampling_parameters,
        scattering_lower_window=scattering_lower_window,
        scattering_upper_window=scattering_upper_window,
    )
    amplitude_tukey = scs.tukey(duration * sampling_frequency, alpha=0.2)
    return (
        xp.fft.rfft(time_domain_parametric_glitch * amplitude_tukey)
        / sampling_frequency
    )


################################################################################
################################################################################
###########################    JAX f(t) Functions    ###########################
################################################################################
################################################################################


def slow_scattering_f_of_t_all_arches_array(
    t: jnpt.ArrayLike, sampling_parameters: jnpt.ArrayLike, number_of_arches: int = 10
) -> jnpt.ArrayLike:
    """Produces the frequency as a function
    of time for each arch in a stack.
    This will always produce a MxN array,
    with M being the number of arches and
    N being the length of the timeseries.

    Parameters
    ==========
        t : jnpt.ArrayLike
            The times for which to evaluate the
            frequencies, of shape (N,)
        sampling_parameters : jnpt.ArrayLike
            [0] is the central time, which sets
            the time at which the arches peak
            [1] is the modulation_frequency
            which sets the width of the arches
            [2] is the base_harmonic_frequency
            which sets the peak frequency of the
            bottom arch
            [3] is the harmonic_frequency_delta,
            which sets the space between arches
        number_of_arches : int
            The number of arches to return.

    Returns
    =======
        jnpt.ArrayLike
            An array of shape (number_of_arches, N) corresponding
            to the frequency as a function of time
            for 10 stacked arches, where N is the
            length of the timeseries.
    """
    f_harm_k = (
        xp.arange(number_of_arches) * sampling_parameters[3] + sampling_parameters[2]
    )
    modulation_of_t = xp.cos(
        2 * xp.pi * sampling_parameters[1] * (t - sampling_parameters[0])
    )
    f_of_t_all_arches = xp.outer(f_harm_k, modulation_of_t)
    return f_of_t_all_arches


def fast_scattering_f_of_t_array(
    t: jnpt.ArrayLike, sampling_parameters: jnpt.ArrayLike
) -> jnpt.ArrayLike:
    """Gives the frequency as a function of time for idealized
    fast scattering. Used for smooth windowing when integrated
    to phi(t) for generating strain.
    The frequency will thus take the form:
    $f(t) = c_1 f_1 cos(2 \\pi f_1 (t - t_1)) +
    c_2 f_2 cos(2 \\pi f_2 (t - t_2))$

    Parameters
    ==========
        t : jnpt.ArrayLike
            The times for which to evaluate the
            frequencies, of shape (N,)
        sampling_parameters : jnpt.ArrayLike
            The parameters to generate the frequency evolution
            of the fast scattering glitch.
            [0] is t_1, the central time associated with the first
            frequency, which to establish convention will be
            the lower (microseism) frequency. This will also    ```
            be treated as the geocent_time for purposes of
            bilby compatibility
            [1] is second_scatterer_offset, the offset from the geocent time for the second
            frequency, which to establish convention will be
            the higher (anthropogenic) frequency
            [2] is f_1, the first frequency, which we'll assume
            for sake of convention is the lower
            (microseism) frequency
            [3] is f_2, the second frequency, which we'll assume
            for the sake of convention is the higher
            (anthropogenic) frequency
            [4] is c_1, the first frequency coefficient, associated
            to the lower frequency f_1
            [5] is c_2, the second frequency coefficient, associated
            to the higher frequency f_2

    Returns
    =======
        jnpt.ArrayLike
            The frequency of the fast scattering as a function of the
            time array input.
    """
    first_frequency_term = (
        sampling_parameters[4]
        * sampling_parameters[2]
        * xp.cos(2 * xp.pi * sampling_parameters[2] * (t - sampling_parameters[0]))
    )
    second_frequency_term = (
        sampling_parameters[5]
        * sampling_parameters[3]
        * xp.cos(
            2
            * xp.pi
            * sampling_parameters[3]
            * (t - sampling_parameters[1] - sampling_parameters[0])
        )
    )
    return first_frequency_term + second_frequency_term


################################################################################
################################################################################
#####################   Keyword Argument f(t) Functions    #####################
################################################################################
################################################################################


def slow_scattering_f_of_t_single_arch_with_keyword(
    t: xp.array,
    central_time: float,
    modulation_frequency: float,
    harmonic_frequency: float,
):
    """Returns arch frequency as a function of time for a parameterized slow scattering model

    Parameters
    ==========
    t : xp.array
        The time array over which to evaluate the frequency
    central_time : float
        The central time (time of peak frequency) for the arch
    modulation_frequency : float
        The driving (microseism) frequency
    harmonic_frequency : float
        The peak frequency of the arch

    Returns
    =======
    xp.array
        The frequency of the arch at sampled times
    """
    return harmonic_frequency * xp.cos(
        2 * xp.pi * modulation_frequency * (t - central_time)
    )


def slow_scattering_f_of_t_all_arches_with_keyword(
    t: jnpt.ArrayLike,
    modulation_frequency: float = 0.5,
    base_harmonic_frequency: float = 30,
    harmonic_frequency_delta: float = 25,
    central_time: float = 0,
    number_of_arches: int = 10,
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
        number_of_arches : int
            The number of arches to return
    Returns
    =======
        jnpt.ArrayLike
            An array of shape (number_of_arches, N) corresponding
            to the frequency as a function of time
            for 10 stacked arches, where N is the
            length of the timeseries.
    """
    input_parameters = [
        central_time,
        modulation_frequency,
        base_harmonic_frequency,
        harmonic_frequency_delta,
    ]
    return slow_scattering_f_of_t_all_arches_array(
        t, input_parameters, number_of_arches=number_of_arches
    )


def slow_scattering_shifting_modulation_f_of_t(
    t: xp.array,
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
        t : xp.array
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
        xp.array
            The sum of the two frequency arches, with a smooth handoff
    """
    initial_function = slow_scattering_f_of_t_single_arch_with_keyword(
        t=t,
        central_time=central_time,
        modulation_frequency=modulation_frequency,
        harmonic_frequency=harmonic_frequency,
    ) * taper_off(
        t=t, taper_central_time=taper_central_time, taper_time_scale=taper_time_scale
    )
    final_function = slow_scattering_f_of_t_single_arch_with_keyword(
        t=t,
        central_time=central_time,
        modulation_frequency=modulation_frequency * modulation_frequency_shift,
        harmonic_frequency=harmonic_frequency * modulation_frequency_shift,
    ) * taper_on(
        t=t, taper_central_time=taper_central_time, taper_time_scale=taper_time_scale
    )
    return initial_function + final_function


################################################################################
################################################################################
########################    JITting when appropriate    ########################
################################################################################
################################################################################


def set_jit_functions_with_jax_backend():
    if "jax" not in xp.__name__:
        print("Cannot jit functions when jax is not the backend")
        return
    import jax
    from importlib import import_module

    this_module = import_module("bilby_glitch.models.parametric_glitch_models")
    fast_scattering_f_of_t_array = jax.jit(this_module.fast_scattering_f_of_t_array)
    slow_scattering_f_of_t_all_arches_array = jax.jit(
        this_module.slow_scattering_f_of_t_all_arches_array,
        static_argnames=("number_of_arches",),
    )
    fast_scattering_g_of_f_array = jax.jit(
        this_module.fast_scattering_g_of_f_array,
        static_argnames=(
            "sampling_frequency",
            "duration",
            "start_time",
            "scattering_lower_window",
            "scattering_upper_window",
        ),
    )
    fast_scattering_g_of_t_array = jax.jit(
        this_module.fast_scattering_g_of_t_array,
        static_argnames=("scattering_lower_window", "scattering_upper_window"),
    )
    fast_scattering_phi_of_t_array = jax.jit(
        this_module.fast_scattering_phi_of_t_array,
        static_argnames=("scattering_lower_window", "scattering_upper_window"),
    )
    wavelet_g_of_f_array = jax.jit(
        this_module.wavelet_g_of_f_array, static_argnames=("start_time")
    )
    slow_scattering_g_of_f_array = jax.jit(
        this_module.slow_scattering_g_of_f_array,
        static_argnames=(
            "sampling_frequency",
            "duration",
            "start_time",
            "number_of_arches",
            "groups_left_of_center",
            "groups_right_of_center",
        ),
    )
    slow_scattering_g_of_t_array = jax.jit(
        this_module.slow_scattering_g_of_t_array,
        static_argnames=(
            "number_of_arches",
            "groups_left_of_center",
            "groups_right_of_center",
        ),
    )
    slow_scattering_phi_of_t_array = jax.jit(
        this_module.slow_scattering_phi_of_t_array,
        static_argnames=("groups_left_of_center", "groups_right_of_center"),
    )
    setattr(
        this_module,
        "slow_scattering_f_of_t_all_arches_array",
        slow_scattering_f_of_t_all_arches_array,
    )
    setattr(
        this_module, "slow_scattering_phi_of_t_array", slow_scattering_phi_of_t_array
    )
    setattr(this_module, "slow_scattering_g_of_t_array", slow_scattering_g_of_t_array)
    setattr(this_module, "slow_scattering_g_of_f_array", slow_scattering_g_of_f_array)
    setattr(this_module, "fast_scattering_f_of_t_array", fast_scattering_f_of_t_array)
    setattr(
        this_module, "fast_scattering_phi_of_t_array", fast_scattering_phi_of_t_array
    )
    setattr(this_module, "fast_scattering_g_of_t_array", fast_scattering_g_of_t_array)
    setattr(this_module, "fast_scattering_g_of_f_array", fast_scattering_g_of_f_array)
    setattr(this_module, "wavelet_g_of_f_array", wavelet_g_of_f_array)
