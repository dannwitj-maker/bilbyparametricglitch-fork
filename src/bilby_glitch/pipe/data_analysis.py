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
    DataAnalysisInput,
)

from .input import GlitchInput
from .parser import glitch_create_parser
from ..result import JointCBCGlitchResult
from ..models.bilby_interface import _blank_frequency_domain_source_model


class GlitchDataAnalysisInput(DataAnalysisInput, GlitchInput):
    """Handles user-input for the data analysis script

    Parameters
    ----------
    parser: BilbyArgParser, optional
        The parser containing the command line / ini file inputs
    args_list: list, optional
        A list of the arguments to parse. Defaults to `sys.argv[1:]`

    """

    def __init__(self, args, unknown_args, test=False):
        DataAnalysisInput.__init__(self, args, unknown_args)

        self.extra_waveform_generator_kwargs = args.extra_waveform_generator_kwargs

        self.likelihood_is_joint = args.likelihood_is_joint

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

        if self.frequency_domain_source_model == _blank_frequency_domain_source_model:
            self.distance_marginalization = False

    @property
    def result_class(self):
        """The bilby result class to store results in"""
        if self.likelihood_is_joint:
            try:
                return JointCBCGlitchResult
            except AttributeError:
                logger.debug("Unable to use Joint CBC-Glitch specific result class")
                return None
        else:
            try:
                return bilby.gw.result.CBCResult
            except AttributeError:
                logger.debug("Unable to use CBC result class")
                return None


def main():
    """Data analysis main logic"""
    args, unknown_args = parse_args(sys.argv[1:], glitch_create_parser(top_level=False))
    log_version_information()
    analysis = GlitchDataAnalysisInput(args, unknown_args)
    analysis.run_sampler()
    if analysis.reweighting_configuration is not None:
        analysis.reweight_result()
    logger.info("Run completed")
