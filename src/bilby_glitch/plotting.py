import bilby
import copy
import numpy as np

from .glitch_foft_models import slow_scattering_f_of_t

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
        sample_foft = slow_scattering_f_of_t(
            interferometer.time_array,
            central_time,
            modulation_frequency,
            harmonic_frequency,
        )
        foft_posterior.append(sample_foft)
    foft_posterior = np.array(foft_posterior)

    return foft_posterior
