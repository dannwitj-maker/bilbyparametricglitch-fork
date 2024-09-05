import jax
import jax.numpy as jnp
import jax.typing as jnpt

from .glitch_foft_models import jax_slow_scattering_f_of_t_all_arches

from typing import Callable


def jax_slow_scattering_factory(number_arches: int) -> Callable:
    """Produces a jax-ified function for slow scattering g(t) in
    the detector, with N arches.

    Parameters
    ==========
        number_arches : int
            The number of arches
    """

    arches_mask = jnp.arange(number_arches)

    @jax.jit
    def jax_slow_scattering_g_of_t(
        t: jnpt.ArrayLike,
        scattering_parameters: jnpt.ArrayLike,
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
        arches_f_of_t = jax_slow_scattering_f_of_t_all_arches(
            t, scattering_parameters[0:14]
        )
        arches_amplitudes = scattering_parameters[14:][:, jnp.newaxis]
        arches_g_of_t_unmasked = jnp.sin(arches_f_of_t) * arches_amplitudes
        arches_g_of_t_masked = arches_g_of_t_unmasked[arches_mask]
        g_of_t = jnp.sum(arches_g_of_t_masked, axis=0)

        # window_low = scattering_parameters[0] - 0.25 / scattering_parameters[1]
        # window_high = scattering_parameters[0] + 0.25 / scattering_parameters[1]
        # window_idxs = jnp.argwhere((t > window_low) & (t < window_high))
        # tukey_window = tukey(window_idxs.flatten().shape[0], alpha=0.2)
        # hoft_temp[window_idxs.flatten()] *= tukey_window
        # outside_window_mask = np.zeros(hoft_temp.shape, dtype=bool)
        # outside_window_mask[window_idxs] = True
        # hoft_temp[~outside_window_mask] = 0

        return g_of_t

    return jax_slow_scattering_g_of_t
