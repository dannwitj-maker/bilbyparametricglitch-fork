from bilby.core.utils import logger
from bilby.gw.conversion import fill_from_fixed_priors, compute_per_detector_log_likelihoods, generate_posterior_samples_from_marginalized_likelihood, MarginalizedLikelihoodReconstructionError, compute_snrs

def identity_map_conversion(parameters):
    """An identity map on the parameters, doing what bilby says it does already"""
    return parameters, []


def identity_map_generation(sample, likelihood=None, priors=None, npool=1):
    """An identity map on the parameters that handles marginalizations, SNRs, etc. correctly

    Parameters
    ==========
    sample: dict or pandas.DataFrame
        Samples to fill in with extra parameters, this may be either an
        injection or posterior samples.
    likelihood: bilby.gw.likelihood.GravitationalWaveTransient, optional
        GravitationalWaveTransient used for sampling, used for waveform and
        likelihood.interferometers.
    priors: dict, optional
        Dictionary of prior objects, used to fill in non-sampled parameters.

    Returns
    =======

    """
    output_sample = sample.copy()

    output_sample = fill_from_fixed_priors(output_sample, priors)

    if likelihood is not None:
        compute_per_detector_log_likelihoods(
            samples=output_sample, likelihood=likelihood, npool=npool)

        marginalized_parameters = getattr(likelihood, "_marginalized_parameters", list())
        if len(marginalized_parameters) > 0:
            try:
                generate_posterior_samples_from_marginalized_likelihood(
                    samples=output_sample, likelihood=likelihood, npool=npool)
            except MarginalizedLikelihoodReconstructionError as e:
                logger.warning(
                    "Marginalised parameter reconstruction failed with message "
                    "{}. Some parameters may not have the intended "
                    "interpretation.".format(e)
                )

        compute_snrs(output_sample, likelihood, npool=npool)

    return output_sample
