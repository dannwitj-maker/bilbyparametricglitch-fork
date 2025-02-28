import bilby
import copy
import numpy as np
from pandas import DataFrame

from ..models.parametric_glitch_models import (
    slow_scattering_f_of_t_single_arch_with_keyword,
)

from typing import Optional, List


def posterior_of_arch_tracks(
    posterior,
    interferometer: bilby.gw.detector.interferometer.Interferometer,
    relative_time_of_track_start: Optional[float] = None,
    relative_time_of_track_end: Optional[float] = None,
    number_of_samples=500,
    seed=1234,
    arch_index=0,
    prefix="glitch_Slow_Arches",
) -> List[np.array]:
    if number_of_samples is None:
        samples = copy.deepcopy(posterior)
    elif number_of_samples <= len(posterior):
        samples = posterior.sample(number_of_samples, random_state=seed)
    else:
        samples = copy.deepcopy(posterior)

    foft_posterior = []
    for _, sample in samples.iterrows():
        harmonic_frequency = (
            sample[f"{prefix}_base_harmonic_frequency"]
            + sample[f"{prefix}_harmonic_frequency_delta"] * arch_index
        )
        modulation_frequency = sample[f"{prefix}_modulation_frequency"]
        geocent_time = sample[f"{prefix}_geocent_time"]
        central_time = geocent_time + interferometer.time_delay_from_geocenter(
            0, 0, geocent_time
        )
        sample_foft = slow_scattering_f_of_t_single_arch_with_keyword(
            interferometer.time_array,
            central_time,
            modulation_frequency,
            harmonic_frequency,
        )
        foft_posterior.append(sample_foft)
    foft_posterior = np.array(foft_posterior)

    return foft_posterior


def clean_glitch_posterior(posterior: DataFrame, glitch_name: str) -> DataFrame:
    glitch_posterior = copy.deepcopy(posterior)
    glitch_key = f"glitch_{glitch_name}"
    for k in ["phi_1", "phi_2"]:
        # annoying special case
        if k in glitch_posterior.keys():
            glitch_posterior.pop(k)

    relevant_keys = [k for k in glitch_posterior if glitch_key in k]
    cleanup_map = {k: k.replace(glitch_key, "") for k in relevant_keys}
    glitch_posterior = glitch_posterior[relevant_keys]
    glitch_posterior = glitch_posterior.rename(columns=cleanup_map)
    for k in ["ra", "dec", "psi"]:
        glitch_posterior[k] = np.zeros(glitch_posterior.shape[0])
    return glitch_posterior
