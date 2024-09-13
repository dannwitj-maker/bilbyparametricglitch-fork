import jax
import jax.numpy as jnp
import jax.typing as jnpt
from functools import partial

from .glitch_foft_models import jax_slow_scattering_f_of_t_all_arches

from typing import Callable

@partial(jax.jit, static_argnames=("groups_left_of_center", "groups_right_of_center"))
def jax_slow_scattering_phi_of_t(
    t: jnpt.ArrayLike,
    scattering_parameters: jnpt.ArrayLike,
    groups_left_of_center : int = 0,
    groups_right_of_center : int = 0,
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
        scattering_parameters : jnpt.ArrayLike
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
    f_harm_k = jnp.arange(10) * scattering_parameters[3] + scattering_parameters[2]
    # Get baseline phi(t) behavior i.e. the appropriate sinusoid for all arches
    modulation_of_t = jnp.sin(
        2 * jnp.pi * scattering_parameters[1] * (t - scattering_parameters[0])
    )
    # Get correctly scaled phi(t) i.e. the indefinite integral of d phi / d t
    # For each arch
    phi_of_t = jnp.outer(f_harm_k, modulation_of_t) / scattering_parameters[1]
    # Add respective phases for each (i.e. setting boundary conditions on integral)
    phi_of_t += scattering_parameters[4:14][:, jnp.newaxis]

    # Get an array which is just the minimum of each row, with shape of phi_of_t  
    minimums_array = jnp.repeat(jnp.min(phi_of_t, axis=1)[:, jnp.newaxis], phi_of_t.shape[1], axis = 1)
    # Get the corresponding maximums array
    maximums_array = jnp.repeat(jnp.max(phi_of_t, axis=1)[:, jnp.newaxis], phi_of_t.shape[1], axis = 1)
    # Get the condition where we cut out all earlier arches
    # This is the left side of the central arch (1/4 of modulation frequency)
    # Then the full width of the other arches (each 1/2 of modulation frequency)
    lower_window_condition_array = jnp.repeat((t < scattering_parameters[0] - (1 / 4 + groups_left_of_center / 2) / scattering_parameters[1])[jnp.newaxis, :], 10, axis=0)
    # Get the corresponding upper condition
    upper_window_condition_array = jnp.repeat((t > scattering_parameters[0] + (1 / 4 + groups_right_of_center / 2)/ scattering_parameters[1])[jnp.newaxis, :], 10, axis=0)
    
    # Decide what to fill the left window with.
    # If an even number of arches to left, it ends on a minimum, else on a maximum
    lower_window_fill = jnp.where(jnp.mod(groups_left_of_center, 2), maximums_array, minimums_array)
    # As above but inverted
    upper_window_fill = jnp.where(jnp.mod(groups_right_of_center, 2), minimums_array, maximums_array)

    # Actually set the windowed values
    windowed_phi_of_t = jnp.where(lower_window_condition_array, lower_window_fill, phi_of_t)
    windowed_phi_of_t = jnp.where(upper_window_condition_array, upper_window_fill, windowed_phi_of_t)
    return windowed_phi_of_t

@partial(jax.jit, static_argnames=('number_of_arches', 'groups_left_of_center', 'groups_right_of_center'))
def jax_slow_scattering_g_of_t(
    t: jnpt.ArrayLike,
    scattering_parameters: jnpt.ArrayLike,
    number_of_arches : int = 1,
    groups_left_of_center : int = 0,
    groups_right_of_center : int = 0
):
    """A model for slow scattering g(t) in the detector
    when the glitch contains N arches,
    where N is set by the

    Parameters
    =========
        t : jnpt.ArrayLike
            The times for which g(t) is being
            computed.
        scattering_parameters : jnpt.ArrayLike
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
    arches_phi_of_t = jax_slow_scattering_phi_of_t(
        t, scattering_parameters[0:14], groups_left_of_center=groups_left_of_center, groups_right_of_center=groups_right_of_center
    )

    arches_amplitudes = scattering_parameters[14:][:, jnp.newaxis]
    arches_g_of_t_unmasked = jnp.sin(arches_phi_of_t) * arches_amplitudes
    arches_g_of_t_masked = arches_g_of_t_unmasked[arches_mask]
    g_of_t = jnp.sum(arches_g_of_t_masked, axis=0)

    return g_of_t