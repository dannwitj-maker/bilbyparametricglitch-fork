import numpy as np
import pandas as pd

from gwpy.timeseries import TimeSeries

from bilby.core.utils import logger
from bilby.gw.waveform_generator import WaveformGenerator
from bilby.gw.conversion import identity_map_conversion

from .bilby_interface import bilbify_parametric_glitch


class GlitchWaveformGenerator(WaveformGenerator):
    def __init__(
        self,
        glitch_index_name,
        glitch_interferometer,
        duration,
        sampling_frequency,
        direct_injection=False,
        start_time=0,
        frequency_domain_source_model=None,
        parametric_time_domain_source_model=None,
        parametric_frequency_domain_source_model=None,
        parametric_source_model_converter=bilbify_parametric_glitch,
        parameters=None,
        parameter_conversion=None,
        waveform_arguments={},
    ):
        waveform_arguments["duration"] = duration
        waveform_arguments["sampling_frequency"] = sampling_frequency
        waveform_arguments["start_time"] = start_time
        if direct_injection:
            waveform_arguments["glitch_mode_name"] = glitch_interferometer
        else:
            waveform_arguments["glitch_mode_name"] = (
                f"glitch_{glitch_interferometer}_{glitch_index_name}"
            )
        if parameter_conversion is None:
            parameter_conversion = identity_map_conversion
        if "aux_channel" in waveform_arguments:
            aux_channel = waveform_arguments["aux_channel"]
            logger.info(f"Getting data for aux channel {aux_channel}")
            aux_time_series = TimeSeries.fetch(
                aux_channel, start_time, start_time + duration, allow_tape=True
            )
            waveform_arguments.pop("aux_channel")
            waveform_arguments["aux_time_series"] = aux_time_series
        elif "aux_channel_file" in waveform_arguments:
            aux_time_series = TimeSeries.read(waveform_arguments["aux_channel_file"])
            waveform_arguments.pop("aux_channel_file")
            waveform_arguments["aux_time_series"] = aux_time_series
        if "number_splines" in waveform_arguments:
            spline_interp_times = np.linspace(
                0, duration, waveform_arguments["number_splines"]
            )
            waveform_arguments["spline_interp_times"] = spline_interp_times

        if "post_trigger_duration" not in waveform_arguments:
            # Default to bilby default of 2
            waveform_arguments["post_trigger_duration"] = 2

        if (
            frequency_domain_source_model is None
            and parametric_frequency_domain_source_model is None
            and parametric_time_domain_source_model
        ):
            raise ValueError(
                "Must set source model directly or provide a model to convert"
            )
        elif (
            parametric_frequency_domain_source_model is not None
            or parametric_time_domain_source_model is not None
        ):
            frequency_domain_source_model = parametric_source_model_converter(
                duration=duration,
                sampling_frequency=sampling_frequency,
                start_time=start_time,
                glitch_name=waveform_arguments["glitch_mode_name"],
                glitch_interferometer=glitch_interferometer,
                time_domain_parametric_glitch_function=parametric_time_domain_source_model,
                frequency_domain_parametric_glitch_function=parametric_frequency_domain_source_model,
                fixed_glitch_waveform_arguments=waveform_arguments,
            )

        super(GlitchWaveformGenerator, self).__init__(
            duration=duration,
            sampling_frequency=sampling_frequency,
            start_time=start_time,
            frequency_domain_source_model=frequency_domain_source_model,
            parameters=parameters,
            parameter_conversion=parameter_conversion,
            waveform_arguments=waveform_arguments,
        )
        self.waveform_arguments["time_array"] = self.time_array
        self.time_domain_parametric_glitch_function = (
            parametric_time_domain_source_model
        )
        self.frequency_domain_parametric_glitch_function = (
            parametric_frequency_domain_source_model
        )
        self.glitch_index_name = glitch_index_name
        self.glitch_interferometer = glitch_interferometer
        self.direct_injection = direct_injection


class NegativeWaveformGenerator(WaveformGenerator):
    def __init__(
        self,
        parent=None,
        duration=None,
        sampling_frequency=None,
        start_time=0,
        frequency_domain_source_model=None,
        time_domain_source_model=None,
        parameters=None,
        parameter_conversion=None,
        waveform_arguments=None,
    ):
        if parent is not None:
            self.__dict__ = parent.__dict__.copy()
        else:
            super(NegativeWaveformGenerator, self).__init__(
                duration,
                sampling_frequency,
                start_time,
                frequency_domain_source_model,
                time_domain_source_model,
                parameters,
                parameter_conversion,
                waveform_arguments,
            )

    def frequency_domain_strain(self, parameters=None):
        base_fd_strain = super().frequency_domain_strain(parameters)
        negative_fd_strain = {key: -1 * val for key, val in base_fd_strain.items()}
        return negative_fd_strain


class InjectionResidualWaveformGenerator(WaveformGenerator):
    def __init__(
        self,
        injection_parameters,
        parent=None,
        duration=None,
        sampling_frequency=None,
        start_time=0,
        frequency_domain_source_model=None,
        time_domain_source_model=None,
        parameters=None,
        parameter_conversion=None,
        waveform_arguments=None,
    ):
        if parent is not None:
            self.__dict__ = parent.__dict__.copy()
        else:
            super(InjectionResidualWaveformGenerator, self).__init__(
                duration,
                sampling_frequency,
                start_time,
                frequency_domain_source_model,
                time_domain_source_model,
                parameters,
                parameter_conversion,
                waveform_arguments,
            )
        self.injection_parameters = injection_parameters

    @property
    def injection_parameters(self) -> pd.DataFrame:
        """The parameters of the injection to take the residual against"""
        return self._injection_parameters

    @injection_parameters.setter
    def injection_parameters(self, injection_parameters: pd.DataFrame) -> None:
        self._injection_parameters = injection_parameters

    def frequency_domain_strain(self, parameters=None):
        injection_fd_strain = super().frequency_domain_strain(self.injection_parameters)
        # get_detector_response will apply a timeshift based on *sample* parameters
        # so, proactively timeshift injection_fd_strain by the time difference
        dt = self.injection_parameters["geocent_time"] - parameters["geocent_time"]
        for mode, waveform in injection_fd_strain.items():
            injection_fd_strain[mode] = waveform * np.exp(
                -1j * 2 * np.pi * dt * self.frequency_array
            )
        estimated_fd_strain = super().frequency_domain_strain(parameters)
        return {
            key: injection_fd_strain[key] - val
            for key, val in estimated_fd_strain.items()
        }
