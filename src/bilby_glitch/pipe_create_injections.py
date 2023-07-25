import sys

import bilby_glitch.conversion
import bilby_glitch.interferometer

import bilby
import bilby_pipe

from bilby_pipe.utils import (
    parse_args,
)
from bilby_pipe.create_injections import BilbyPipeCreateInjectionsError, create_parser


class GlitchInjectionCreator(bilby_pipe.create_injections.InjectionCreator):
    """An object to hold inputs to create_injection for consistency - with modifications for glitches"""

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


bilby_pipe.create_injections.InjectionCreator = GlitchInjectionCreator


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


bilby_pipe.create_injections.create_injection_file = glitch_create_injection_file


def main():
    """Driver to create an injection file"""
    args, unknown_args = parse_args(sys.argv[1:], create_parser())

    if args.gps_tuple is not None:
        gpstimes = bilby_pipe.input.Input.parse_gps_tuple(args.gps_tuple)
    elif args.gps_file is not None:
        gpstimes = bilby_pipe.input.Input.read_gps_file(args.gps_file)
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
