import sys
import os
import pandas as pd
import numpy as np
import json

import bilby
from bilby_pipe.utils import (
    parse_args,
    logger,
    check_directory_exists_and_if_not_mkdir,
    get_geocent_time_with_uncertainty,
)
from bilby_pipe.create_injections import BilbyPipeCreateInjectionsError, create_parser

from .pipe_input import GlitchInput


class GlitchInjectionCreator(GlitchInput):
    """An object to hold inputs to create_injection for consistency"""

    def __init__(
        self,
        prior_file,
        prior_dict,
        default_prior,
        trigger_time,
        gpstimes,
        n_injection,
        generation_seed,
        deltaT=0.2,
        duration=4,
        post_trigger_duration=2,
        enforce_signal_duration=False,
        minimum_frequency=None,
        joint_prior_files=None,
        injection_is_joint=False,
    ):
        self.prior_file = prior_file
        self.prior_dict = prior_dict
        self.injection_is_joint = injection_is_joint
        self.likelihood_is_joint = injection_is_joint
        self.joint_prior_files = joint_prior_files
        self.check_prior()
        self.default_prior = default_prior
        self.trigger_time = trigger_time
        self.gpstimes = gpstimes
        self.deltaT = deltaT
        self.duration = duration
        self.post_trigger_duration = post_trigger_duration
        self.n_injection = n_injection
        self.generation_seed = generation_seed
        self.time_reference = "geocent"
        self.reference_frame = "sky"
        self.detectors = self.known_detectors
        self.minimum_frequency = minimum_frequency
        self.enforce_signal_duration = enforce_signal_duration

        if self.gpstimes is not None and self.trigger_time not in [0, None]:
            raise BilbyPipeCreateInjectionsError(
                "Cannot parse both gpstimes and trigger_time"
            )

    def check_prior(self):
        """Ensures at least prior/prior_dict set"""
        if (
            self.prior_file is None
            and self.prior_dict is None
            and self.joint_prior_files is None
        ):
            raise BilbyPipeCreateInjectionsError("No prior_file or prior_dict given")

    @property
    def n_injection(self):
        """The number of injections parameters to be stored."""
        return self._n_injection

    @n_injection.setter
    def n_injection(self, n_injection):
        if self.gpstimes is not None:
            gps_n_injection = len(self.gpstimes)
            if n_injection is not None:
                logger.warning(
                    f"n-injection={gps_n_injection} given but gps_file/gps_tuple is also given,"
                    f" ignoring n-injection={n_injection}"
                )
            n_injection = gps_n_injection

        if isinstance(n_injection, int) is False or n_injection < 1:
            raise BilbyPipeCreateInjectionsError(
                f"n_injection={n_injection}, but must be a positive integer"
            )
        self._n_injection = n_injection

    def get_injection_dataframe(self):
        """Samples parameters from the prior into a dataframe"""
        inj_df = pd.DataFrame.from_dict(self.priors.sample(self.n_injection))
        if self.gpstimes is not None:
            geocent_times = []
            for start_time in self.gpstimes:
                geocent_time = get_geocent_time_with_uncertainty(
                    geocent_time=start_time
                    + self.duration
                    - self.post_trigger_duration,
                    uncertainty=self.deltaT / 2.0,
                )
                geocent_times.append(geocent_time)
            inj_df["geocent_time"] = geocent_times
        return inj_df

    @staticmethod
    def write_injection_dataframe(dataframe, filename, extension):
        """Writes dataframe into a file with a dat/json extension"""
        path, extension = get_full_path(filename, extension)
        if extension == "json":
            injections = dict(injections=dataframe)
            with open(path, "w") as file:
                json.dump(
                    injections, file, indent=2, cls=bilby.core.result.BilbyJsonEncoder
                )
        elif extension == "dat":
            dataframe.to_csv(path, index=False, header=True, sep=" ")
        else:
            raise BilbyPipeCreateInjectionsError(
                f"Extension {extension} not implemented"
            )
        logger.info(f"Created injection file {path}")

    def generate_injection_file(self, filepath, extension):
        """Sets the generation seed and randomly generates parameters to create inj"""
        np.random.seed(self.generation_seed)
        logger.info(
            f"Generating injection file {filepath} from "
            f"prior={self.prior_file}, "
            f"n_injection={self.n_injection}, "
            f"generation_seed={self.generation_seed}"
        )
        injection_dataframe = self.get_injection_dataframe()
        self.write_injection_dataframe(injection_dataframe, filepath, extension)


def get_full_path(filename, extension):
    """Makes filename and ext consistent amongst user input"""
    ext_in_filename = os.path.splitext(filename)[1].lstrip(".")
    if ext_in_filename == "":
        path = f"{filename}.{extension}"
    elif ext_in_filename == extension:
        path = filename
    else:
        logger.debug("Overwriting given extension name")
        path = filename
        extension = ext_in_filename
    outdir = os.path.dirname(path)
    if outdir != "":
        check_directory_exists_and_if_not_mkdir(outdir)
    return path, extension


def create_injection_file(
    filename,
    n_injection,
    prior_file=None,
    prior_dict=None,
    trigger_time=None,
    deltaT=0.2,
    gpstimes=None,
    duration=4,
    post_trigger_duration=2,
    generation_seed=None,
    extension="dat",
    default_prior="BBHPriorDict",
    enforce_signal_duration=False,
    joint_prior_files=None,
):
    """Makes injection file using arguments from the namespace args parameter"""
    injection_creator = GlitchInjectionCreator(
        prior_file=prior_file,
        prior_dict=prior_dict,
        n_injection=n_injection,
        default_prior=default_prior,
        trigger_time=trigger_time,
        deltaT=deltaT,
        gpstimes=gpstimes,
        duration=duration,
        post_trigger_duration=post_trigger_duration,
        generation_seed=generation_seed,
        enforce_signal_duration=enforce_signal_duration,
        joint_prior_files=joint_prior_files,
    )
    injection_creator.generate_injection_file(filename, extension)


def glitch_create_injection_file(
    filename,
    n_injection,
    prior_file=None,
    prior_dict=None,
    trigger_time=None,
    deltaT=0.2,
    gpstimes=None,
    duration=4,
    post_trigger_duration=2,
    generation_seed=None,
    extension="dat",
    default_prior="BBHPriorDict",
    enforce_signal_duration=False,
    joint_prior_files=None,
):
    """Makes injection file using arguments from the namespace args parameter - with modifications for glitch use"""
    injection_creator = GlitchInjectionCreator(
        prior_file=prior_file,
        prior_dict=prior_dict,
        n_injection=n_injection,
        default_prior=default_prior,
        trigger_time=trigger_time,
        deltaT=deltaT,
        gpstimes=gpstimes,
        duration=duration,
        post_trigger_duration=post_trigger_duration,
        generation_seed=generation_seed,
        enforce_signal_duration=enforce_signal_duration,
        joint_prior_files=joint_prior_files,
    )
    injection_creator.generate_injection_file(filename, extension)


def main():
    """Driver to create an injection file"""
    args, unknown_args = parse_args(sys.argv[1:], create_parser())

    if args.gps_tuple is not None:
        gpstimes = GlitchInput.parse_gps_tuple(args.gps_tuple)
    elif args.gps_file is not None:
        gpstimes = GlitchInput.read_gps_file(args.gps_file)
    else:
        gpstimes = None

    glitch_create_injection_file(
        args.filename,
        prior_file=args.prior_file,
        prior_dict=None,
        n_injection=args.n_injection,
        trigger_time=args.trigger_time,
        deltaT=args.deltaT,
        gpstimes=gpstimes,
        duration=args.duration,
        post_trigger_duration=args.post_trigger_duration,
        generation_seed=args.generation_seed,
        enforce_signal_duration=args.enforce_signal_duration,
        default_prior=args.default_prior,
    )
