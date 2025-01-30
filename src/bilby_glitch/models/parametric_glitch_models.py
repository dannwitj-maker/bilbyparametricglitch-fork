import jax
import jax.numpy as jnp
import jax.typing as jnpt
from functools import partial

import numpy as np

from .series_transforms import taper_off, taper_on
from .jax_utils import tukey, index_of_nearest_value, cumulative_trapezoid

from typing import Tuple

jax.config.update("jax_enable_x64", True)


################################################################################
################################################################################
##########################    JAX Strain Functions    ##########################
################################################################################
################################################################################


@partial(jax.jit, static_argnames=("groups_left_of_center", "groups_right_of_center"))
def slow_scattering_phi_of_t_jax(
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
    f_harm_k = jnp.arange(10) * sampling_parameters[3] + sampling_parameters[2]
    # Get baseline phi(t) behavior i.e. the appropriate sinusoid for all arches
    modulation_of_t = jnp.sin(
        2 * jnp.pi * sampling_parameters[1] * (t - sampling_parameters[0])
    )
    # Get correctly scaled phi(t) i.e. the indefinite integral of d phi / d t
    # For each arch
    phi_of_t = jnp.outer(f_harm_k, modulation_of_t) / sampling_parameters[1]
    # Add respective phases for each (i.e. setting boundary conditions on integral)
    phi_of_t += sampling_parameters[4:14][:, jnp.newaxis]

    # Get an array which is just the minimum of each row, with shape of phi_of_t
    minimums_array = jnp.repeat(
        jnp.min(phi_of_t, axis=1)[:, jnp.newaxis], phi_of_t.shape[1], axis=1
    )
    # Get the corresponding maximums array
    maximums_array = jnp.repeat(
        jnp.max(phi_of_t, axis=1)[:, jnp.newaxis], phi_of_t.shape[1], axis=1
    )
    # Get the condition where we cut out all earlier arches
    # This is the left side of the central arch (1/4 of modulation frequency)
    # Then the full width of the other arches (each 1/2 of modulation frequency)
    lower_window_condition_array = jnp.repeat(
        (
            t
            < sampling_parameters[0]
            - (1 / 4 + groups_left_of_center / 2) / sampling_parameters[1]
        )[jnp.newaxis, :],
        10,
        axis=0,
    )
    # Get the corresponding upper condition
    upper_window_condition_array = jnp.repeat(
        (
            t
            > sampling_parameters[0]
            + (1 / 4 + groups_right_of_center / 2) / sampling_parameters[1]
        )[jnp.newaxis, :],
        10,
        axis=0,
    )

    # Decide what to fill the left window with.
    # If an even number of arches to left, it ends on a minimum, else on a maximum
    lower_window_fill = jnp.where(
        jnp.mod(groups_left_of_center, 2), maximums_array, minimums_array
    )
    # As above but inverted
    upper_window_fill = jnp.where(
        jnp.mod(groups_right_of_center, 2), minimums_array, maximums_array
    )

    # Actually set the windowed values
    windowed_phi_of_t = jnp.where(
        lower_window_condition_array, lower_window_fill, phi_of_t
    )
    windowed_phi_of_t = jnp.where(
        upper_window_condition_array, upper_window_fill, windowed_phi_of_t
    )
    return windowed_phi_of_t


# @partial(
#     jax.jit,
#     static_argnames=(
#         "number_of_arches",
#         "groups_left_of_center",
#         "groups_right_of_center",
#     ),
# )
def slow_scattering_g_of_t_jax(
    t: jnpt.ArrayLike,
    sampling_parameters: jnpt.ArrayLike,
    number_of_arches: int = 1,
    groups_left_of_center: int = 0,
    groups_right_of_center: int = 0,
    **kwargs
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

    arches_mask = jnp.arange(number_of_arches)
    arches_phi_of_t = slow_scattering_phi_of_t_jax(
        t,
        sampling_parameters[0:14],
        groups_left_of_center=groups_left_of_center,
        groups_right_of_center=groups_right_of_center,
    )

    arches_amplitudes = sampling_parameters[14:][:, jnp.newaxis]
    print(arches_amplitudes)
    arches_g_of_t_unmasked = jnp.sin(arches_phi_of_t) * arches_amplitudes
    arches_g_of_t_masked = arches_g_of_t_unmasked[arches_mask]
    print(arches_g_of_t_unmasked)
    print(arches_mask)
    g_of_t = jnp.sum(arches_g_of_t_masked, axis=0)

    return g_of_t


@partial(
    jax.jit,
    static_argnames=(
        "sampling_frequency",
        "duration",
        "start_time",
        "number_of_arches",
        "groups_left_of_center",
        "groups_right_of_center",
    ),
)
def slow_scattering_g_of_f_jax(
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
    t = jnp.linspace(start_time, duration + start_time, duration * sampling_frequency)
    time_domain_parametric_glitch = slow_scattering_g_of_t_jax(
        t,
        sampling_parameters,
        number_of_arches=number_of_arches,
        groups_left_of_center=groups_left_of_center,
        groups_right_of_center=groups_right_of_center,
    )
    amplitude_tukey = tukey(duration * sampling_frequency, alpha=0.2)
    return (
        jnp.fft.rfft(time_domain_parametric_glitch * amplitude_tukey)
        / sampling_frequency
    )


@partial(jax.jit, static_argnames=("start_time"))
def wavelet_g_of_f_jax(
    f,
    sampling_parameters,
    start_time: float = 0,
):
    """
    Parameters
    ==========
    parameters : jnp.array
        index 0 -> central_frequency
        index 1 -> central_time
        index 2 -> quality_factor
        index 3 -> amplitude
        index 4 -> reference_phase
    """
    time = sampling_parameters[1] - start_time
    tau = sampling_parameters[2] / (2 * jnp.pi * sampling_parameters[0])
    A = sampling_parameters[3]

    const = A * tau * jnp.sqrt(jnp.pi) / 2
    decay_term = jnp.exp(-(tau**2) * jnp.pi**2 * (f - sampling_parameters[0]) ** 2)

    Q_decay_term = jnp.exp(-sampling_parameters[2] ** 2 * f / sampling_parameters[0])
    plus_term = jnp.exp(-1j * (2 * jnp.pi * time * f - sampling_parameters[4]))
    minus_term = jnp.exp(-1j * (2 * jnp.pi * time * f + sampling_parameters[4]))

    return const * decay_term * (plus_term + minus_term * Q_decay_term)


@partial(jax.jit, static_argnames=("scattering_window"))
def fast_scattering_phi_of_t_jax(
    t: jnpt.ArrayLike,
    sampling_parameters: jnpt.ArrayLike,
    scattering_window: Tuple[float, float],
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
        scattering_window : Tuple[float, float]
            The start and end times of the windowing function.
            All times outside this window but in the data segment will be set
            to 0.

    Returns
    =======
        jnpt.ArrayLike
            The phi(t) associated with the fast scattering
    """
    f_of_t = fast_scattering_f_of_t_jax(t, sampling_parameters[:-1])

    # Here begins something contrived
    # The goal is to
    below_window_and_zero_crossing = (jnp.isclose(f_of_t, 0, atol=0.5)) & (
        t <= scattering_window[0]
    )
    indices_below_window_and_zero_crossing = jnp.where(
        below_window_and_zero_crossing,
        jnp.indices(t.shape).flatten(),
        jnp.zeros_like(t),
    )
    above_window_and_zero_crossing = (jnp.isclose(f_of_t, 0, atol=0.5)) & (
        t >= scattering_window[1]
    )
    indices_above_window_and_zero_crossing = jnp.where(
        above_window_and_zero_crossing,
        jnp.indices(t.shape).flatten(),
        jnp.ones_like(t) * (t.shape[0] + 1),
    )

    index_last_crossing_before_window = jnp.argmax(
        indices_below_window_and_zero_crossing
    )
    index_first_crossing_after_window = jnp.argmin(
        indices_above_window_and_zero_crossing
    )

    windowed_fast_scattering_f_of_t = jnp.where(
        (jnp.indices(t.shape) <= index_last_crossing_before_window)
        | (jnp.indices(t.shape) >= index_first_crossing_after_window),
        jnp.zeros_like(t),
        f_of_t,
    )

    phi_of_t = (
        cumulative_trapezoid(windowed_fast_scattering_f_of_t.flatten(), t, initial=0)
        * 2
        * jnp.pi
        + sampling_parameters[6]
    )
    return phi_of_t


@partial(jax.jit, static_argnames=("scattering_window"))
def fast_scattering_g_of_t_jax(
    t: jnpt.ArrayLike,
    sampling_parameters: jnpt.ArrayLike,
    scattering_window: Tuple[float, float],
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
        scattering_window : Tuple[float, float]
            The start and end times of the windowing function.
            All times outside this window but in the data segment will be set
            to 0.

    Returns
    =======
    jnpt.ArrayLike
        The glitch signal for this fast scattering glitch
    """
    phi_of_t = fast_scattering_phi_of_t_jax(
        t, sampling_parameters[:-1], scattering_window=scattering_window
    )
    g_of_t = sampling_parameters[7] * jnp.sin(phi_of_t)

    return g_of_t


@partial(
    jax.jit,
    static_argnames=(
        "sampling_frequency",
        "duration",
        "start_time",
        "scattering_window",
    ),
)
def fast_scattering_g_of_f_jax(
    f: jnpt.ArrayLike,
    sampling_parameters: jnpt.ArrayLike,
    scattering_window: Tuple[float, float],
    sampling_frequency: int,
    duration: float,
    start_time: float,
) -> jnpt.ArrayLike:
    t = jnp.linspace(start_time, duration + start_time, duration * sampling_frequency)
    time_domain_parametric_glitch = fast_scattering_g_of_t_jax(
        t, sampling_parameters=sampling_parameters, scattering_window=scattering_window
    )
    amplitude_tukey = tukey(duration * sampling_frequency, alpha=0.2)
    return (
        jnp.fft.rfft(time_domain_parametric_glitch * amplitude_tukey)
        / sampling_frequency
    )


################################################################################
################################################################################
###########################    JAX f(t) Functions    ###########################
################################################################################
################################################################################


@partial(jax.jit, static_argnames=("number_of_arches"))
def slow_scattering_f_of_t_all_arches_jax(
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
        jnp.arange(number_of_arches) * sampling_parameters[3] + sampling_parameters[2]
    )
    modulation_of_t = jnp.cos(
        2 * jnp.pi * sampling_parameters[1] * (t - sampling_parameters[0])
    )
    f_of_t_all_arches = jnp.outer(f_harm_k, modulation_of_t)
    return f_of_t_all_arches


@partial(jax.jit)
def fast_scattering_f_of_t_jax(
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

    Returns
    =======
        jnpt.ArrayLike
            The frequency of the fast scattering as a function of the
            time array input.
    """
    first_frequency_term = (
        sampling_parameters[4]
        * sampling_parameters[2]
        * jnp.cos(2 * jnp.pi * sampling_parameters[2] * (t - sampling_parameters[0]))
    )
    second_frequency_term = (
        sampling_parameters[5]
        * sampling_parameters[3]
        * jnp.cos(2 * jnp.pi * sampling_parameters[3] * (t - sampling_parameters[1]))
    )
    return first_frequency_term + second_frequency_term


################################################################################
################################################################################
#####################   Keyword Argument f(t) Functions    #####################
################################################################################
################################################################################


def slow_scattering_f_of_t_single_arch_with_keyword(
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
    return slow_scattering_f_of_t_all_arches_jax(
        t, input_parameters, number_of_arches=number_of_arches
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
