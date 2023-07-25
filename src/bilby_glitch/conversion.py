from bilby.gw.conversion import (
    fill_from_fixed_priors,
    compute_per_detector_log_likelihoods,
    generate_posterior_samples_from_marginalized_likelihood,
    MarginalizedLikelihoodReconstructionError,
    _compute_snrs,
    multiprocessing,
    logger,
    sys
)
import bilby.gw.conversion

def glitch_compute_snrs(sample, likelihood, npool=1):
    """
    Compute the optimal and matched filter snrs of all posterior samples
    and print it out.
    Adapted to also compute SNRs for glitch modes

    Parameters
    ==========
    sample: dict or array_like

    likelihood: bilby.gw.likelihood.GravitationalWaveTransient
        Likelihood function to be applied on the posterior

    """
    if likelihood is not None:
        if isinstance(sample, dict):
            likelihood.parameters.update(sample)
            signal_polarizations = likelihood.waveform_generator.frequency_domain_strain(likelihood.parameters.copy())
            for ifo in likelihood.interferometers:
                per_detector_snr = likelihood.calculate_snrs(signal_polarizations, ifo)
                sample['{}_matched_filter_snr'.format(ifo.name)] =\
                    per_detector_snr.complex_matched_filter_snr
                sample['{}_optimal_snr'.format(ifo.name)] = \
                    per_detector_snr.optimal_snr_squared.real ** 0.5
        else:
            from tqdm.auto import tqdm
            logger.info('Computing SNRs for every sample.')

            fill_args = [(ii, row) for ii, row in sample.iterrows()]
            if npool > 1:
                from bilby.core.sampler.base_sampler import _initialize_global_variables
                pool = multiprocessing.Pool(
                    processes=npool,
                    initializer=_initialize_global_variables,
                    initargs=(likelihood, None, None, False),
                )
                logger.info(
                    "Using a pool with size {} for nsamples={}".format(npool, len(sample))
                )
                new_samples = pool.map(_compute_snrs, tqdm(fill_args, file=sys.stdout))
                pool.close()
                pool.join()
            else:
                from bilby.core.sampler.base_sampler import _sampling_convenience_dump
                _sampling_convenience_dump.likelihood = likelihood
                new_samples = [_compute_snrs(xx) for xx in tqdm(fill_args, file=sys.stdout)]

            for ii, ifo in enumerate(likelihood.interferometers):
                if hasattr(likelihood._CalculatedSNRs, "optimal_glitch_snr_squared"):
                    # I would prefer to do an isinstance check, but this produces circular imports
                    # This should be an equivalent way to do so without explicitly referencing
                    # JointGlitchGravitationalWaveTransient
                    gw_matched_filter_snrs = list()
                    glitch_matched_filter_snrs = list()
                    gw_optimal_snrs = list()
                    glitch_optimal_snrs = list()
                    gw_mf_key = '{}_gw_matched_filter_snr'.format(ifo.name)
                    glitch_mf_key = '{}_glitch_matched_filter_snr'.format(ifo.name)
                    gw_optimal_key = '{}_gw_optimal_snr'.format(ifo.name)
                    glitch_optimal_key = '{}_glitch_optimal_snr'.format(ifo.name)
                    for new_sample in new_samples:
                        gw_matched_filter_snrs.append(new_sample[ii].complex_matched_filter_gw_snr)
                        gw_optimal_snrs.append(new_sample[ii].optimal_gw_snr_squared.real ** 0.5)
                        glitch_matched_filter_snrs.append(new_sample[ii].complex_matched_filter_glitch_snr)
                        glitch_optimal_snrs.append(new_sample[ii].optimal_glitch_snr_squared.real ** 0.5)
                    sample[gw_mf_key] = gw_matched_filter_snrs
                    sample[gw_optimal_key] = gw_optimal_snrs
                    sample[glitch_mf_key] = glitch_matched_filter_snrs
                    sample[glitch_optimal_key] = glitch_optimal_snrs

                else:
                    matched_filter_snrs = list()
                    optimal_snrs = list()
                    mf_key = '{}_matched_filter_snr'.format(ifo.name)
                    optimal_key = '{}_optimal_snr'.format(ifo.name)
                    for new_sample in new_samples:
                        matched_filter_snrs.append(new_sample[ii].complex_matched_filter_snr)
                        optimal_snrs.append(new_sample[ii].optimal_snr_squared.real ** 0.5)

                    sample[mf_key] = matched_filter_snrs
                    sample[optimal_key] = optimal_snrs

    else:
        logger.debug('Not computing SNRs.')


bilby.gw.conversion.compute_snrs = glitch_compute_snrs
