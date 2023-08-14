from gwpy.timeseries import TimeSeries

from bilby.core.utils import logger
from bilby.gw.waveform_generator import WaveformGenerator
from bilby.gw.conversion import identity_map_conversion


class GlitchWaveformGenerator(WaveformGenerator):
    def __init__(
        self,
        glitch_index_name,
        glitch_interferometer,
        duration,
        sampling_frequency,
        start_time=0,
        frequency_domain_source_model=None,
        parameters=None,
        parameter_conversion=None,
        waveform_arguments={},
    ):

        waveform_arguments["duration"] = duration
        waveform_arguments["sampling_frequency"] = sampling_frequency
        waveform_arguments["start_time"] = start_time
        waveform_arguments[
            "glitch_mode_name"
        ] = f"glitch_{glitch_interferometer}_{glitch_index_name}"
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
