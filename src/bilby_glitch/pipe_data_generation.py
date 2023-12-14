import sys
import os
import numpy as np
import gwpy
import lal
import glob

import bilby

from bilby.gw.detector import PowerSpectralDensity, calibration

from bilby_pipe.plotting_utils import plot_whitened_data, strain_spectrogram_plot
from bilby_pipe.utils import (
    logger,
    parse_args,
    log_version_information,
    get_version_information,
    convert_string_to_dict,
    BilbyPipeError,
    is_a_power_of_2,
    get_geocent_time_with_uncertainty,
    DataDump,
)
from bilby_pipe.data_generation import DataGenerationInput

from .pipe_input import GlitchInput
from .pipe_parser import glitch_create_parser
from .glitch_models import _blank_frequency_domain_source_model


class GlitchDataGenerationInput(DataGenerationInput, GlitchInput):
    """Handles user-input for the data generation script

    Parameters
    ----------
    parser: configargparse.ArgParser, optional
        The parser containing the command line / ini file inputs
    args_list: list, optional
        A list of the arguments to parse. Defaults to `sys.argv[1:]`
    create_data: bool
        If false, no data is generated (used for testing)

    """

    def __init__(self, args, unknown_args, create_data=True):
        DataGenerationInput.__init__(self, args, unknown_args, create_data=False)

        self.extra_waveform_generator_kwargs = args.extra_waveform_generator_kwargs

        # Check whether to do joint conditionals
        self.likelihood_is_joint = args.likelihood_is_joint
        self.injection_is_joint = args.injection_is_joint

        # If we are doing joint, initialize extra stuff
        if self.likelihood_is_joint:
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

        if self.injection_is_joint:
            self.joint_injection_files = args.joint_injection_files

            if (
                args.joint_injection_waveform_generator_source_models is None
                and self.likelihood_is_joint
            ):
                self.joint_injection_waveform_generator_source_models = (
                    args.joint_waveform_generator_source_models
                )
            else:
                self.joint_injection_waveform_generator_source_models = (
                    args.joint_injection_waveform_generator_source_models
                )

            if (
                args.joint_injection_waveform_generators is None
                and self.likelihood_is_joint
            ):
                self.joint_injection_waveform_generator_classes = (
                    args.joint_waveform_generators
                )
            else:
                self.joint_injection_waveform_generator_classes = (
                    args.joint_injection_waveform_generators
                )

            if (
                args.joint_injection_waveform_generator_args is None
                and self.likelihood_is_joint
            ):
                self.joint_injection_waveform_generator_kwargs = (
                    args.joint_waveform_generator_args
                )
            else:
                self.joint_injection_waveform_generator_kwargs = (
                    args.joint_injection_waveform_generator_args
                )

            if (
                args.joint_injection_waveform_generator_model_args is None
                and self.likelihood_is_joint
            ):
                self.joint_injection_waveform_generator_model_kwargs = (
                    args.joint_waveform_generator_model_args
                )
            else:
                self.joint_injection_waveform_generator_model_kwargs = (
                    args.joint_injection_waveform_generator_model_args
                )

        if args.injection_frequency_domain_source_model is None:
            self.injection_frequency_domain_source_model = (
                args.frequency_domain_source_model
            )
        else:
            self.injection_frequency_domain_source_model = (
                args.injection_frequency_domain_source_model
            )

        if self.frequency_domain_source_model == _blank_frequency_domain_source_model:
            self.distance_marginalization = False

        if create_data:
            self.create_data(args)

    def _set_interferometers_from_injection_in_gaussian_noise(self):
        """Method to generate the interferometers data from an injection in Gaussian noise"""

        self.injection_parameters = self.injection_df.iloc[self.idx].to_dict()
        if self.injection_is_joint:
            for index_name, injection_file in self.joint_injection_files.items():
                read_joint_injection_parameters = self.read_injection_file(
                    injection_file
                )
                logger.info(read_joint_injection_parameters)
                read_joint_injection_parameters = read_joint_injection_parameters.iloc[
                    self.idx
                ].to_dict()
                joint_injection_parameters = dict()
                for key, val in read_joint_injection_parameters.items():
                    # Distinguish by index name
                    joint_injection_parameters[f"glitch_{index_name}_{key}"] = val
                self.injection_parameters.update(joint_injection_parameters)

        logger.info("Injecting waveform with ")
        for prop in [
            "minimum_frequency",
            "maximum_frequency",
            "trigger_time",
            "start_time",
            "duration",
        ]:
            logger.info(f"{prop} = {getattr(self, prop)}")

        self._set_interferometers_from_gaussian_noise()

        waveform_arguments = self.get_injection_waveform_arguments()
        logger.info(f"Using waveform arguments: {waveform_arguments}")
        logger.info(f"Parameter conversion function is: {self.parameter_conversion}")
        waveform_generator = self.waveform_generator_class(
            duration=self.duration,
            start_time=self.start_time,
            sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=self.injection_bilby_frequency_domain_source_model,
            parameter_conversion=self.parameter_conversion,
            waveform_arguments=waveform_arguments,
            **self.extra_waveform_generator_kwargs,
        )

        self.interferometers.inject_signal(
            waveform_generator=waveform_generator,
            parameters=self.injection_parameters,
            raise_error=self.enforce_signal_duration,
        )

        logger.info(f"All injection parameters: {self.injection_parameters}")

        if self.injection_is_joint:
            for (
                index_name,
                waveform_generator,
            ) in self.joint_injection_waveform_generators.items():
                logger.info(
                    f"Using waveform generator {waveform_generator} for glitch {index_name}"
                )

                glitch_injection_parameters = dict()
                # Deobfuscate injection parameters for this case:
                for key, val in self.injection_parameters.items():
                    if index_name in key:
                        # The case that this is a param corresponding to this glitch
                        new_key = key.split(index_name)[-1].strip("_")
                        if "time" in new_key and "geocent" not in new_key:
                            if "geocent_time" in self.injection_parameters.keys():
                                raise KeyError(
                                    "geocent_time and detector time are being set simultaneously: something's wrong!"
                                )
                            ifo = new_key.split("_")[0]
                            inteferometer = bilby.gw.detector.get_empty_interferometer(
                                ifo
                            )
                            # We are converting from IFO time to geocenter time, so a -1 should be applied
                            # Since the geocent prior will be small
                            time_shift = -1 * inteferometer.time_delay_from_geocenter(
                                0, 0, val
                            )
                            glitch_injection_parameters["geocent_time"] = (
                                val + time_shift
                            )
                        else:
                            glitch_injection_parameters[new_key] = val
                glitch_injection_parameters["ra"] = 0
                glitch_injection_parameters["dec"] = 0
                glitch_injection_parameters["psi"] = 0

                logger.info(
                    f"Glitch injection parameters: {glitch_injection_parameters}"
                )

                self.interferometers.inject_signal(
                    waveform_generator=waveform_generator,
                    parameters=glitch_injection_parameters,
                    raise_error=False,
                )


def main():
    """Data generation main logic"""
    args, unknown_args = parse_args(sys.argv[1:], glitch_create_parser(top_level=False))
    print(args)
    print(unknown_args)
    log_version_information()
    data = GlitchDataGenerationInput(args, unknown_args)
    if args.likelihood_type == "ROQGravitationalWaveTransient":
        data.save_roq_weights()
    if data.is_likelihood_multiband:
        data.save_multiband_weights()
    data.save_data_dump()
    logger.info("Completed data generation")
