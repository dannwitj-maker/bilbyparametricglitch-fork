import logging
import copy

import numpy as np

from ..models.parametric_glitch_models import (
    slow_scattering_g_of_f_array,
    wavelet_g_of_f_array,
    fast_scattering_g_of_f_array,
)

import inspect
import makefun

from typing import TYPE_CHECKING, Optional, Dict

logger = logging.getLogger(__name__)
xp = np

if TYPE_CHECKING:
    pass

################################################################################
################################################################################
####################   Utility Functions and Definitions    ####################
################################################################################
################################################################################

SLOW_SCATTERING_SAMPLING_PARAMETERS = [
    "geocent_time",
    "modulation_frequency",
    "base_harmonic_frequency",
    "harmonic_frequency_delta",
    "phi_0",
    "phi_1",
    "phi_2",
    "phi_3",
    "phi_4",
    "phi_5",
    "phi_6",
    "phi_7",
    "phi_8",
    "phi_9",
    "amplitude_0",
    "amplitude_1",
    "amplitude_2",
    "amplitude_3",
    "amplitude_4",
    "amplitude_5",
    "amplitude_6",
    "amplitude_7",
    "amplitude_8",
    "amplitude_9",
]

SLOW_SCATTERING_HELPER_PARAMETERS = [
    "sampling_frequency",
    "duration",
    "start_time",
    "number_of_arches",
    "groups_left_of_center",
    "groups_right_of_center",
]

SINGLE_WAVELET_SAMPLING_PARAMETERS = [
    "central_frequency",
    "geocent_time",
    "quality_factor",
    "amplitude",
    "reference_phase",
]

SINGLE_WAVELET_HELPER_PARAMETERS = ["start_time"]

FAST_SCATTERING_SAMPLING_PARAMETERS = [
    "geocent_time",
    "second_scatterer_phase",
    "frequency_1",
    "frequency_2",
    "motion_coefficient_1",
    "motion_coefficient_2",
    "phi",
    "amplitude",
]

FAST_SCATTERING_HELPER_PARAMETERS = [
    "sampling_frequency",
    "duration",
    "start_time",
    "scattering_lower_window",
    "scattering_upper_window",
]


class WrappedGlitchFunction:
    """A wrapped glitch function with some helper metadata,
    this is used to create a bilby API compatible glitch function
    out of a jax (array input) glitch function.
    """

    def __init__(self, function, sampling_parameters, helper_parameters):
        self.function = function
        self.sampling_parameters = sampling_parameters
        self.helper_parameters = helper_parameters

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def __repr__(self):
        return f"""WrappedGlitchFunction(
            function={self.function},
            sampling_parameters={self.sampling_parameters},
            helper_parameters={self.helper_parameters}
        )"""


single_wavelet_frequency_domain = WrappedGlitchFunction(
    wavelet_g_of_f_array,
    sampling_parameters=SINGLE_WAVELET_SAMPLING_PARAMETERS,
    helper_parameters=SINGLE_WAVELET_HELPER_PARAMETERS,
)
slow_scattering_frequency_domain = WrappedGlitchFunction(
    slow_scattering_g_of_f_array,
    sampling_parameters=SLOW_SCATTERING_SAMPLING_PARAMETERS,
    helper_parameters=SLOW_SCATTERING_HELPER_PARAMETERS,
)
fast_scattering_frequency_domain = WrappedGlitchFunction(
    fast_scattering_g_of_f_array,
    sampling_parameters=FAST_SCATTERING_SAMPLING_PARAMETERS,
    helper_parameters=FAST_SCATTERING_HELPER_PARAMETERS,
)


def populate_unused_parameters(
    original_dict: dict,
    full_parameter_list: list = SLOW_SCATTERING_SAMPLING_PARAMETERS,
    add_sky_params=False,
) -> dict:
    """Fill out a dictionary (either injection parameters or a priordict)
    with the parameters needed for the model, but unused.
    These are all set as 0.

    Parameters
    ==========
    original_dict : dict
        The dictionary to modify
    full_parameter_list : list, optional
        The list of parameters to fill out, defaults to those in the slow scattering SHO model
    add_sky_params : bool
        Whether to also add unused sky params needed by Bilby - should be turned off if a cbc model is also present

    Returns
    =======
    dict
        The updated dictionary with unused but necessary parameters set as 0
    """
    temp_dict = copy.deepcopy(original_dict)
    for key in full_parameter_list:
        if key not in original_dict:
            temp_dict[key] = 0
    if add_sky_params:
        temp_dict["ra"] = 0
        temp_dict["dec"] = 0
        temp_dict["psi"] = 0
    return temp_dict


def _blank_frequency_domain_source_model(f, geocent_time, **kwargs):
    """A placeholder for when no CBC is passed to a joint analysis"""
    return {"plus": xp.zeros(f.shape), "cross": xp.zeros(f.shape)}


def bilbify_parametric_glitch(
    duration: int,
    sampling_frequency: int,
    start_time: float,
    glitch_name: str,
    glitch_interferometer: str,
    time_domain_parametric_glitch_function: Optional[WrappedGlitchFunction] = None,
    frequency_domain_parametric_glitch_function: Optional[WrappedGlitchFunction] = None,
    fixed_glitch_waveform_arguments: Dict[str, any] = None,
):
    """Produces a wrapped version of the parametric
    function for a glitch which produces the outputs
    bilby expects to see, namely a frequency series
    with times appropriately managed.

    Parameters
    ==========
        duration : float
            The duration of the data segment
        sampling_frequency : float
            The sampling frequency of the data segment
        start_time : float
            The start time of the data segment
        glitch_name : str
            An identifying name for the glitch.
            Should be {ifo} when used with likelihood
            computations.
        time_domain_parametric_glitch_function : Optional[WrappedGlitchFunction]
            The TD callable glitch function to use, appropriately wrapped.
            Exactly one of time_domain_parametric_glitch_function or
            frequency_domain_parametric_glitch_function should be passed.
        frequency_domain_parametric_glitch_function : Optional[WrappedGlitchFunction]
            The FD callable glitch function to use, appropriately wrapped.
            Exactly one of time_domain_parametric_glitch_function or
            frequency_domain_parametric_glitch_function should be passed.
    """
    if (
        time_domain_parametric_glitch_function is None
        and frequency_domain_parametric_glitch_function is None
    ):
        raise ValueError("Must specify a function to wrap")

    fixed_glitch_waveform_arguments = (
        {}
        if fixed_glitch_waveform_arguments is None
        else fixed_glitch_waveform_arguments
    )

    fixed_glitch_waveform_arguments["duration"] = int(duration)
    fixed_glitch_waveform_arguments["start_time"] = start_time
    fixed_glitch_waveform_arguments["sampling_frequency"] = int(sampling_frequency)

    # Applying some convention to do introspection
    # Track "sampling_parameters" for the function
    # as the arguments that will be inspected out
    # Such that bilby can sample over them.
    # All other args will be handled as kwargs
    if time_domain_parametric_glitch_function is not None:
        # Produce the time array which will actually be used for TD analyses
        t = xp.linspace(
            start_time, duration + start_time, int(duration * sampling_frequency)
        )

        def frequency_domain_parametric_glitch_function(f, **kwargs):
            time_domain_parametric_glitch = time_domain_parametric_glitch_function(
                t, time_domain_parametric_glitch_function.sampling_parameters, **kwargs
            )
            return xp.fft.rfft(time_domain_parametric_glitch) / sampling_frequency

        frequency_domain_parametric_glitch_function = WrappedGlitchFunction(
            frequency_domain_parametric_glitch_function,
            time_domain_parametric_glitch_function.sampling_parameters,
            time_domain_parametric_glitch_function.helper_parameters,
        )

    # Get the end time of the segment, for handling time conventions
    segment_end_time = duration + start_time
    # Get the time delay for the glitch, so that it can be placed at the *detector* time in a way
    # Which appropriately compensates for bilby convention
    # By *our* convention glitches get a "sky location" of ra=dec=0

    signature_parameters = []
    signature_parameters.append(
        inspect.Parameter(
            name="f",
            kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=xp.typing.ArrayLike,
        )
    )
    for parameter in frequency_domain_parametric_glitch_function.sampling_parameters:
        signature_parameters.append(
            inspect.Parameter(
                name=parameter,
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=1,
                annotation=float,
            )
        )
    signature_parameters.append(
        inspect.Parameter(name="kwargs", kind=inspect.Parameter.VAR_KEYWORD)
    )

    signature = inspect.Signature(parameters=signature_parameters)

    fixed_waveform_arguments_to_use = {
        k: v
        for k, v in fixed_glitch_waveform_arguments.items()
        if k in frequency_domain_parametric_glitch_function.helper_parameters
    }

    @makefun.with_signature(
        func_signature=signature,
    )
    def bilby_glitch_function(f, **kwargs):
        sampling_parameters = xp.array(
            [
                kwargs[x]
                for x in frequency_domain_parametric_glitch_function.sampling_parameters
            ]
        )
        g_of_f = frequency_domain_parametric_glitch_function(
            f,
            sampling_parameters=sampling_parameters,
            **fixed_waveform_arguments_to_use,
        )

        # bilby assumes that function maps "geocent_time" to the end of the array, wrapping
        # for times after geocent time. So, e.g. the merger will occur at the end of the segment
        # and the ringdown will wrap back to the beginning.
        # It then applies its own rotation accordingly so that the merger actually happens at the correct time
        # Now, for our purposes parametric functions are assumed to take their own meaning of geocent time seriously
        # i.e. put the glitch at the right time. So, this function will apply a rotation such that
        # the stated geocent time gets rotated to be at the end of the segment.
        # This is just exp(-2*pi*i*(t_end - t_0))
        # t_0 being whatever time parameter the model uses, which this automatically assumes to be in
        # detector frame (applying the detector time delay accordingly)
        g_of_f *= xp.exp(
            -2 * xp.pi * 1j * (segment_end_time - kwargs["geocent_time"]) * f
        )

        return {
            "plus": xp.zeros(g_of_f.shape),
            "cross": xp.zeros(g_of_f.shape),
            glitch_name: xp.array(g_of_f),
        }

    return bilby_glitch_function
