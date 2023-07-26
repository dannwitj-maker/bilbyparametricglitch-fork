import sys
import numpy as np
import os
import time
import signal

import bilby

from bilby_pipe.utils import (
    parse_args,
    logger,
    log_version_information,
    BilbyPipeError,
    DataDump,
    get_function_from_string_path,
    CHECKPOINT_EXIT_CODE,
)
from bilby_pipe.data_analysis import (
    create_analysis_parser,
    _get_items_in_group,
    sighandler,
)

from .pipe_input import GlitchInput


class GlitchDataAnalysisInput(GlitchInput):
    """Handles user-input for the data analysis script

    Parameters
    ----------
    parser: BilbyArgParser, optional
        The parser containing the command line / ini file inputs
    args_list: list, optional
        A list of the arguments to parse. Defaults to `sys.argv[1:]`

    """

    def __init__(self, args, unknown_args, test=False):
        super().__init__(args, unknown_args)

        # Generic initialisation
        self.meta_data = dict()
        self.result = None

        # Admin arguments
        self.ini = args.ini
        self.scheduler = args.scheduler
        self.periodic_restart_time = args.periodic_restart_time
        self.request_cpus = args.request_cpus
        self.run_local = args.local

        # Naming arguments
        self.outdir = args.outdir
        self.label = args.label
        self.result_format = args.result_format

        # Data dump file to run on
        self.data_dump_file = args.data_dump_file

        # Check whether to do joint conditionals
        self.likelihood_is_joint = args.likelihood_is_joint

        # Choices for running
        self.detectors = args.detectors
        self.sampler = args.sampler
        self.sampler_kwargs = args.sampler_kwargs
        self.sampling_seed = args.sampling_seed

        # Frequencies
        self.sampling_frequency = args.sampling_frequency
        self.minimum_frequency = args.minimum_frequency
        self.maximum_frequency = args.maximum_frequency
        self.reference_frequency = args.reference_frequency

        # If we are doing joint, initialize extra stuff
        if self.likelihood_is_joint:
            # For joint, we need duration and start time
            self.duration = args.duration
            self.trigger_time = args.trigger_time
            self.post_trigger_duration = args.post_trigger_duration

            self.joint_injection_files = args.joint_injection_files
            self.joint_prior_defaults = args.joint_prior_defaults
            self.joint_prior_files = args.joint_prior_files
            self.joint_waveform_generator_source_models = (
                args.joint_waveform_generator_source_models
            )
            self.joint_waveform_generator_kwargs = args.joint_waveform_generator_args
            self.joint_waveform_generator_model_kwargs = (
                args.joint_waveform_generator_model_args
            )
            self.joint_waveform_generator_classes = args.joint_waveform_generators

        # Waveform, source model and likelihood
        self.waveform_generator_class = args.waveform_generator
        self.waveform_approximant = args.waveform_approximant
        self.extra_waveform_generator_kwargs = args.extra_waveform_generator_kwargs
        self.catch_waveform_errors = args.catch_waveform_errors
        self.pn_spin_order = args.pn_spin_order
        self.pn_tidal_order = args.pn_tidal_order
        self.pn_phase_order = args.pn_phase_order
        self.pn_amplitude_order = args.pn_amplitude_order
        self.mode_array = args.mode_array
        self.waveform_arguments_dict = args.waveform_arguments_dict
        self.numerical_relativity_file = args.numerical_relativity_file
        self.frequency_domain_source_model = args.frequency_domain_source_model
        self.conversion_function = args.conversion_function
        self.generation_function = args.generation_function
        self.likelihood_type = args.likelihood_type
        self.reference_frame = args.reference_frame
        self.time_reference = args.time_reference
        self.extra_likelihood_kwargs = args.extra_likelihood_kwargs
        self.enforce_signal_duration = args.enforce_signal_duration

        # ROQ
        self.roq_folder = args.roq_folder
        self.roq_scale_factor = args.roq_scale_factor

        # Calibration
        self.calibration_model = args.calibration_model
        self.spline_calibration_envelope_dict = args.spline_calibration_envelope_dict
        self.spline_calibration_amplitude_uncertainty_dict = (
            args.spline_calibration_amplitude_uncertainty_dict
        )
        self.spline_calibration_phase_uncertainty_dict = (
            args.spline_calibration_phase_uncertainty_dict
        )
        self.spline_calibration_nodes = args.spline_calibration_nodes
        self.calibration_prior_boundary = args.calibration_prior_boundary

        # Marginalization
        self.distance_marginalization = args.distance_marginalization
        self.distance_marginalization_lookup_table = None
        self.phase_marginalization = args.phase_marginalization
        self.time_marginalization = args.time_marginalization
        self.jitter_time = args.jitter_time
        self.calibration_marginalization = args.calibration_marginalization
        self.calibration_lookup_table = args.calibration_lookup_table
        self.number_of_response_curves = args.number_of_response_curves

        if test is False:
            self._load_data_dump()

            # Reweighting
            self.reweighting_configuration = self.meta_data.get(
                "reweighting_configuration", None
            )
            self.reweight_nested_samples = args.reweight_nested_samples

            # heterodyning - this relies on the data dump
            self.fiducial_parameters = self.meta_data.get("fiducial_parameters", None)
            self.update_fiducial_parameters = False
            self.epsilon = args.epsilon

    @property
    def sampling_seed(self):
        return self._sampling_seed

    @sampling_seed.setter
    def sampling_seed(self, sampling_seed):
        if sampling_seed is None:
            sampling_seed = np.random.randint(1, 1e6)
        self._sampling_seed = sampling_seed
        np.random.seed(sampling_seed)
        logger.info(f"Sampling seed set to {sampling_seed}")

        if not any(
            [
                k in self.sampler_kwargs
                for k in bilby.core.sampler.Sampler.sampling_seed_equiv_kwargs
            ]
        ):
            self.sampler_kwargs["sampling_seed"] = self._sampling_seed

    @property
    def interferometers(self):
        try:
            return self._interferometers
        except AttributeError:
            ifos = self.data_dump.interferometers
            names = [ifo.name for ifo in ifos]
            logger.info(f"Found data for detectors = {names}")
            ifos_to_use = [ifo for ifo in ifos if ifo.name in self.detectors]
            names_to_use = [ifo.name for ifo in ifos_to_use]
            logger.info(f"Using data for detectors = {names_to_use}")
            self._interferometers = bilby.gw.detector.InterferometerList(ifos_to_use)
            self.print_detector_information(self._interferometers)
            return self._interferometers

    @staticmethod
    def print_detector_information(interferometers):
        for ifo in interferometers:
            logger.info(
                "{}: sampling-frequency={}, segment-start-time={}, duration={}".format(
                    ifo.name,
                    ifo.strain_data.sampling_frequency,
                    ifo.strain_data.start_time,
                    ifo.strain_data.duration,
                )
            )

    @property
    def data_dump(self):
        if hasattr(self, "_data_dump"):
            return self._data_dump
        else:
            raise BilbyPipeError("Data dump not loaded")

    def _load_data_dump(self):
        filename = self.data_dump_file
        self.meta_data["data_dump"] = filename

        logger.debug("Data dump not previously loaded")

        if os.path.isfile(filename):
            pass
        elif os.path.isfile(os.path.basename(filename)):
            filename = os.path.basename(filename)
        else:
            raise FileNotFoundError(
                "No dump data {} file found. Most likely the generation "
                "step failed".format(filename)
            )

        self._data_dump = DataDump.from_pickle(filename)
        self.meta_data.update(self._data_dump.meta_data)
        return self._data_dump

    @property
    def result_class(self):
        """The bilby result class to store results in"""
        try:
            return bilby.gw.result.CompactBinaryCoalescenceResult
        except AttributeError:
            logger.debug("Unable to use CBC specific result class")
            return None

    @property
    def result_directory(self):
        result_dir = os.path.join(self.outdir, "result")
        return os.path.relpath(result_dir)

    def get_likelihood_and_priors(self):
        """Read in the likelihood and prior from the data dump

        This reads in the data dump values and reconstructs the likelihood and
        priors. Note, care must be taken to use the "search_priors" which differ
        from the true prior when using marginalization

        Returns
        -------
        likelihood, priors
            The bilby likelihood and priors
        """

        priors = self.data_dump.priors_class(self.data_dump.priors_dict)
        self.priors = priors

        self.likelihood_lookup_table = self.data_dump.likelihood_lookup_table
        self.likelihood_roq_weights = self.data_dump.likelihood_roq_weights
        self.likelihood_roq_params = self.data_dump.likelihood_roq_params
        self.likelihood_multiband_weights = self.data_dump.likelihood_multiband_weights

        likelihood = self.likelihood
        priors = self.search_priors
        return likelihood, priors

    def run_sampler(self):
        if self.scheduler.lower() == "condor" and not self.run_local:
            signal.signal(signal.SIGALRM, handler=sighandler)
            signal.alarm(self.periodic_restart_time)

        likelihood, priors = self.get_likelihood_and_priors()

        if self.reweighting_configuration is not None:
            conversion_function = None
        else:
            conversion_function = self.parameter_generation

        if (
            self.sampler == "bilby_mcmc"
            and "proposal_cycle" in self._sampler_kwargs.keys()
        ):
            if "." in self._sampler_kwargs["proposal_cycle"]:
                self._sampler_kwargs["proposal_cycle"] = get_function_from_string_path(
                    self._sampler_kwargs["proposal_cycle"]
                )

        self.result = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            sampler=self.sampler,
            label=self.label,
            outdir=self.result_directory,
            conversion_function=conversion_function,
            injection_parameters=self.meta_data["injection_parameters"],
            meta_data=self.meta_data,
            result_class=self.result_class,
            exit_code=CHECKPOINT_EXIT_CODE,
            save=self.result_format,
            **self.sampler_kwargs,
        )

    def get_likelihood_and_priors_for_reweighting(self):
        data = self.reweighting_configuration
        need_likelihood = False
        likelihood = None
        priors = None
        self.search_priors = self.priors.copy()

        target_arguments = _get_items_in_group("likelihood")
        target_arguments += _get_items_in_group("waveform")

        if "prior-file" in data:
            priors = self.priors.copy()
            priors.update(bilby.core.prior.PriorDict(data["prior-file"]))
            self.search_priors = priors
        for key, value in data.items():
            key = key.replace("-", "_")
            if key in target_arguments:
                logger.info(f"Setting {key} to {value} for reweighting")
                setattr(self, key, value)
                need_likelihood = True
        if need_likelihood:
            likelihood = self.likelihood
            likelihood.parameters.update(self.search_priors.sample())
            likelihood.outdir = self.outdir
            likelihood.label = self.label
        return likelihood, priors

    def reweight_result(self):
        old_priors = self.result.priors
        self.search_priors = old_priors
        reweight_nest = self.reweight_nested_samples
        if reweight_nest and self.sampler in ["dynesty", "nessai"]:
            self.result.nested_samples["log_prior"] = old_priors.ln_prob(
                {
                    key: self.result.nested_samples[key].values
                    for key in self.result.search_parameter_keys
                },
                axis=0,
            )
            for key in old_priors:
                if isinstance(old_priors[key], bilby.core.prior.DeltaFunction):
                    self.result.nested_samples[key] = old_priors[key].sample()
        else:
            reweight_nest = False
        old_likelihood = self.likelihood
        likelihood, priors = self.get_likelihood_and_priors_for_reweighting()
        if likelihood is None:
            old_likelihood = None
        if priors is None:
            old_priors = None
        target_keys = list(self.search_priors.keys()) + ["log_prior", "log_likelihood"]
        for key in list(self.result.posterior):
            if key not in target_keys:
                del self.result.posterior[key]

        if likelihood is not None:
            n_evaluations = min(100, len(self.result.posterior))
            t_start = time.time()
            for i in range(n_evaluations):
                likelihood.parameters = {
                    key: self.result.posterior.iloc[i][key]
                    for key in self.result.posterior
                }
                likelihood.log_likelihood()
            time_per_likelihood = (time.time() - t_start) / n_evaluations
            logger.debug(f"{time_per_likelihood:.2f} s per likelihood evaluation")
            n_checkpoint = int(300 / time_per_likelihood)
        else:
            n_checkpoint = 3000
        logger.debug(f"Checkpointing every {n_checkpoint} samples")

        reweighted = bilby.core.result.reweight(
            result=self.result,
            label=self.label,
            new_likelihood=likelihood,
            new_prior=priors,
            old_likelihood=old_likelihood,
            old_prior=old_priors,
            conversion_function=self.parameter_generation,
            npool=self.request_cpus,
            verbose_output=False,
            resume_file=f"{self.result_directory}/{self.label}_reweight_resume.pkl",
            n_checkpoint=n_checkpoint,
            use_nested_samples=reweight_nest,
        )
        reweighted.save_to_file(
            extension=self.result_format, overwrite=True, outdir=self.result_directory
        )


def main():
    """Data analysis main logic"""
    args, unknown_args = parse_args(sys.argv[1:], create_analysis_parser())
    log_version_information()
    analysis = GlitchDataAnalysisInput(args, unknown_args)
    analysis.run_sampler()
    if analysis.reweighting_configuration is not None:
        analysis.reweight_result()
    logger.info("Run completed")
