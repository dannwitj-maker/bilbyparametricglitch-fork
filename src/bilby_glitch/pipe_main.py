import os
import sys
import json
import importlib

import bilby_glitch.conversion
import bilby_glitch.interferometer
import bilby_glitch.pipe_nodes

import bilby
import bilby_pipe

from bilby_pipe.utils import (
    logger,
    BilbyPipeError,
    parse_args,
    log_version_information,
    get_command_line_arguments,
    get_outdir_name,
    tcolors,
)
from bilby_pipe.main import create_parser, write_complete_config_file, generate_dag


class GlitchMainInput(bilby_pipe.main.MainInput):
    """An object to hold all the inputs to bilby_pipe"""

    def __init__(self, args, unknown_args, perform_checks=True):
        super().__init__(args, unknown_args, perform_checks=perform_checks)

        self.likelihood_is_joint = args.likelihood_is_joint

        if self.likelihood_is_joint:
            self.joint_prior_files = args.joint_prior_files
            self.joint_prior_defaults = args.joint_prior_defaults

    def check_injection(self):
        """Check injection behaviour

        If injections are requested, either use the injection-dict,
        injection-file, or create an injection-file

        """
        default_injection_file_name = "{}/{}_injection_file.dat".format(
            self.data_directory, self.label
        )
        if self.injection_dict is not None:
            logger.debug(
                "Using injection dict from ini file {}".format(
                    json.dumps(self.injection_dict, indent=2)
                )
            )
        elif self.injection_file is not None:
            logger.debug(f"Using injection file {self.injection_file}")
        elif os.path.isfile(default_injection_file_name):
            # This is done to avoid overwriting the injection file
            logger.debug(f"Using injection file {default_injection_file_name}")
            self.injection_file = default_injection_file_name
        else:
            logger.debug("No injection file found, generating one now")

            if self.gps_file is not None or self.gps_tuple is not None:
                if self.n_simulation > 0 and self.n_simulation != len(self.gpstimes):
                    raise BilbyPipeError(
                        "gps_file/gps_tuple option and n_simulation are not matched"
                    )
                gpstimes = self.gpstimes
                n_injection = len(gpstimes)
            else:
                gpstimes = None
                n_injection = self.n_simulation

            if self.trigger_time is None:
                trigger_time_injections = 0
            else:
                trigger_time_injections = self.trigger_time

            bilby_pipe.create_injections.create_injection_file(
                filename=default_injection_file_name,
                prior_file=self.prior_file,
                prior_dict=self.prior_dict,
                n_injection=n_injection,
                trigger_time=trigger_time_injections,
                deltaT=self.deltaT,
                gpstimes=gpstimes,
                duration=self.duration,
                post_trigger_duration=self.post_trigger_duration,
                generation_seed=self.generation_seed,
                extension="dat",
                default_prior=self.default_prior,
                joint_prior_files=self.joint_prior_files,
            )
            self.injection_file = default_injection_file_name

        # Check the gps_file has the sample length as number of simulation
        if self.gps_file is not None:
            if len(self.gpstimes) != len(self.injection_df):
                raise BilbyPipeError("Injection file length does not match gps_file")

        if self.n_simulation > 0:
            if self.n_simulation != len(self.injection_df):
                raise BilbyPipeError(
                    "n-simulation does not match the number of injections: "
                    "please check your ini file"
                )
        elif self.n_simulation == 0 and self.gps_file is None:
            self.n_simulation = len(self.injection_df)
            logger.debug(
                f"Setting n_simulation={self.n_simulation} to match injections"
            )


bilby_pipe.main.MainInput = GlitchMainInput


def main():
    """Top-level interface for bilby_pipe"""
    parser = create_parser(top_level=True)
    args, unknown_args = parse_args(get_command_line_arguments(), parser)

    if args.analysis_executable_parser is not None:
        # Alternative parser requested, reload args
        module = ".".join(args.analysis_executable_parser.split(".")[:-1])
        function = args.analysis_executable_parser.split(".")[-1]
        parser = getattr(importlib.import_module(module), function)()
        args, unknown_args = parse_args(get_command_line_arguments(), parser)

    # Check and sort outdir
    args.outdir = args.outdir.replace("'", "").replace('"', "")
    if args.overwrite_outdir is False:
        args.outdir = get_outdir_name(args.outdir)

    log_version_information()
    inputs = GlitchMainInput(args, unknown_args)
    write_complete_config_file(parser, args, inputs)
    generate_dag(inputs)

    if len(unknown_args) > 0:
        msg = [tcolors.WARNING, f"Unrecognized arguments {unknown_args}", tcolors.END]
        logger.warning(" ".join(msg))
