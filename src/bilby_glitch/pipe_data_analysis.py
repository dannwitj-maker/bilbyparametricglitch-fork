import sys

import bilby_glitch.conversion
import bilby_glitch.interferometer

import bilby
import bilby_pipe

from bilby_pipe.utils import parse_args, logger, log_version_information
from bilby_pipe.data_analysis import create_analysis_parser


class GlitchDataAnalysisInput(bilby_pipe.data_analysis.DataAnalysisInput):
    """Handles user-input for the data analysis script - with modifications for glitches

    Parameters
    ----------
    parser: BilbyArgParser, optional
        The parser containing the command line / ini file inputs
    args_list: list, optional
        A list of the arguments to parse. Defaults to `sys.argv[1:]`

    """

    def __init__(self, args, unknown_args, test=False):
        super().__init__(args, unknown_args, test=test)
        # Check whether to do joint conditionals
        self.likelihood_is_joint = args.likelihood_is_joint

        # Since we are doing joint, initialize extra stuff
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

        self.extra_waveform_generator_kwargs = args.extra_waveform_generator_kwargs


bilby_pipe.data_analysis.DataAnalysisInput = GlitchDataAnalysisInput


def main():
    """Data analysis main logic"""
    args, unknown_args = parse_args(sys.argv[1:], create_analysis_parser())
    log_version_information()
    analysis = GlitchDataAnalysisInput(args, unknown_args)
    analysis.run_sampler()
    if analysis.reweighting_configuration is not None:
        analysis.reweight_result()
    logger.info("Run completed")
