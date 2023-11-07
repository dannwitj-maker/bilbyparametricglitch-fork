import os
import numpy as np
import pandas as pd
import copy
import gwpy
from typing import List
from astropy import units as u
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
from seaborn import set_palette

set_palette("colorblind")

from bilby.core.utils import latex_plot_format, infft, logger, safe_save_figure
from bilby.gw.detector import get_empty_interferometer, Interferometer
from bilby.gw.waveform_generator import WaveformGenerator
from bilby.gw.utils import asd_from_freq_series


def get_fd_and_td_whitened_waveforms(
    samples: pd.DataFrame,
    waveform_generator: WaveformGenerator,
    interferometer: Interferometer,
    frequency_idxs: np.ndarray,
    time_idxs: np.ndarray,
):
    fd_waveforms = list()
    td_waveforms = list()
    for _, params in samples.iterrows():
        params = dict(params)
        wf_pols = waveform_generator.frequency_domain_strain(params)
        fd_waveform = interferometer.get_detector_response(wf_pols, params)
        fd_waveforms.append(fd_waveform[frequency_idxs])
        td_waveform = infft(
            fd_waveform
            * np.sqrt(2.0 / interferometer.sampling_frequency)
            / interferometer.amplitude_spectral_density_array,
            interferometer.sampling_frequency,
        )[time_idxs]
        td_waveforms.append(td_waveform)
    fd_waveforms = asd_from_freq_series(
        fd_waveforms, 1 / interferometer.strain_data.duration
    )
    td_waveforms = np.array(td_waveforms)

    return fd_waveforms, td_waveforms


@latex_plot_format
def plot_interferometer_reconstruction(
    interferometer: Interferometer,
    waveform_generators: List[WaveformGenerator],
    posteriors: List[pd.DataFrame],
    labels: List[str] | None,
    prefixes: List[str | None] | None = None,
    fixed_waveforms: List[gwpy.timeseries.TimeSeries] | None = None,
    colors: List[str] | None = None,
    level: float = 0.9,
    n_samples: int | None = None,
    save: bool = True,
    format: str = "pdf",
    central_time: float | None = None,
    data_start_time: float | None = None,
    end_time: float | None = None,
    minimum_frequency: float | None = None,
    maximum_frequency: float | None = None,
    duration: int | None = None,
    sampling_frequency: int | None = None,
    start_time: float | None = None,
    filename: str | None = None,
    outdir: str = os.getcwd(),
    run_label: str = "analysis",
) -> matplotlib.figure.Figure | None:
    """
    Plot the posterior for the waveform in the frequency domain and
    whitened time domain.

    If the strain data is passed that will be plotted.

    If injection parameters can be found, the injection will be plotted.

    Parameters
    ==========
    interferometer: (str, bilby.gw.detector.interferometer.Interferometer)
        detector to use, if an Interferometer object is passed the data
        will be overlaid on the posterior
    waveform_generators : List[bilby.gw.waveform_generator.WaveformGenerator]
        A list of waveform generators to use in producing the posterior reconstructions
    posteriors : List[pd.DataFrame]
        A list of posterior data frames for reconstructing
    prefixes : List[str | None], optional
        If glitch waveforms are passed, a list of prefixes to remove from posterior parameter names.
        Pass None for cases where no prefix should be removed.
    fixed_waveforms : List[gwpy.timeseries.TimeSeries]
        A list of timeseries to also whiten and plot with the reconstructions.
    labels : List[str], optional
        A list of the labels for each posterior and fixed waveform.
        If not passed defaults to posterior{ii}, fixed_waveform{ii}.
    colors : List[str], optional
        The colors for each posterior and fixed waveform.
        If not passed will populate with defaults.
    level: float, optional
        symmetric confidence interval to show, default is 90%
    n_samples: int, optional
        number of samples to use to calculate the median/interval
        default is all
    save: bool, optional
        whether to save the image, default=True
        if False, figure handle is returned
    format: str, optional
        format to save the figure in, default is png
    central_time : float, optional
        The time reference for the start and end time. If none is past will read mean of the first posterior passed.
    start_time: float, optional
        The amount of time before the central time to begin the time domain plot.
        Default is - 0.4
    end_time: float, optional
        The amount of time after the central time to end the time domain plot.
        Default is 0.2
    duration : int, optional
        The duration of the data segment, only necessary if creating the ifo object
    sampling_frequency : int, optional
        The sampling frequency of the data segment, only necessary if creating the ifo object
    start_time : float, optional
        The start time of the data segment, onlynecessary if creating the ifo object
    filename : str, optional
        If passed, write to this file, else it will be inferred.
    outdir : str, optional
        If passed, write into this directory, else use cwd.
        filename will overwrite this behavior.
    run_label : str, optional
        If passed, label the plot with this. If not passed defaults to "analysis."


    Returns
    =======
    fig: figure-handle, only if save=False

    Notes
    -----
    To reduce the memory footprint we decimate the frequency domain
    waveforms to have ~4000 entries. This should be sufficient for decent
    resolution.
    """

    DATA_COLOR = "#ff7f0e"

    if prefixes is None:
        prefixes = [None] * len(posteriors)
    if labels is None:
        labels = [f"posterior {ii}" for ii in len(posteriors)] + [
            f"fixed waveform {ii}" for ii in len(fixed_waveforms)
        ]
    if colors is None:
        colors = [
            "C0",
            "C4",
            "C2",
            "C3",
            "C5",
            "C7",
            "C8",
            "C1",
            "C6",
            "C9",
        ]

    # Check whether there is any data to overplot
    if isinstance(interferometer, str):
        interferometer = get_empty_interferometer(interferometer)
        interferometer.set_strain_data_from_zero_noise(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=data_start_time,
        )
        PLOT_DATA = False
    elif not isinstance(interferometer, Interferometer):
        raise TypeError("interferometer must be either str or Interferometer")
    else:
        PLOT_DATA = True
    logger.info(f"Generating waveform figure for {interferometer.name}")

    if central_time is None:
        central_time = np.mean(posteriors[0].geocent_time)

    if start_time is None:
        start_time = -0.4
    start_time = central_time + start_time
    if end_time is None:
        end_time = 0.2
    end_time = central_time + end_time

    time_idxs = (interferometer.time_array >= start_time) & (
        interferometer.time_array <= end_time
    )
    frequency_idxs = np.where(interferometer.frequency_mask)[0]
    logger.debug(f"Frequency mask contains {len(frequency_idxs)} values")
    frequency_idxs = frequency_idxs[:: max(1, len(frequency_idxs) // 4000)]
    logger.debug(f"Downsampling frequency mask to {len(frequency_idxs)} values")
    frequency_window_factor = np.sum(interferometer.frequency_mask) / len(
        interferometer.frequency_mask
    )
    plot_times = interferometer.time_array[time_idxs]
    plot_times -= interferometer.strain_data.start_time
    start_time -= interferometer.strain_data.start_time
    end_time -= interferometer.strain_data.start_time
    plot_frequencies = interferometer.frequency_array[frequency_idxs]

    samples_used = list()
    fd_waveforms_used = list()
    td_waveforms_used = list()

    for (generator, posterior, prefix) in zip(
        waveform_generators, posteriors, prefixes
    ):
        if n_samples is None:
            samples = copy.deepcopy(posterior)
        elif n_samples > len(posterior):
            logger.debug(
                "Requested more waveform samples ({}) than we have "
                "posterior samples ({})!".format(n_samples, len(posterior))
            )
            samples = copy.deepcopy(posterior)
        else:
            samples = posterior.sample(n_samples, replace=False)

        if prefix is not None:
            # If a prefix is offered, this is a glitch waveform and we need to do some manipulations
            parameters = {
                key.split(prefix)[1].strip("_"): parameter
                for key, parameter in posterior.items()
                if prefix in key
            }
            parameters["ra"] = np.zeros_like(parameters["geocent_time"])
            parameters["dec"] = np.zeros_like(parameters["geocent_time"])
            parameters["psi"] = np.zeros_like(parameters["geocent_time"])

            samples = pd.DataFrame(parameters)

        fd_waveforms, td_waveforms = get_fd_and_td_whitened_waveforms(
            samples=samples,
            waveform_generator=generator,
            interferometer=interferometer,
            frequency_idxs=frequency_idxs,
            time_idxs=time_idxs,
        )

        samples_used.append(samples)
        fd_waveforms_used.append(fd_waveforms)
        td_waveforms_used.append(td_waveforms)

    plot_additional_tseries_data = list()
    if fixed_waveforms is not None:
        for tseries in fixed_waveforms:
            tseries = tseries.crop(
                start_time,
                end_time,
                copy=True,
            )
            tseries.times -= (start_time) * u.s
            tseries = tseries.whiten(
                asd=gwpy.frequencyseries.FrequencySeries(
                    interferometer.amplitude_spectral_density_array,
                    frequencies=interferometer.frequency_array,
                )
            )
            plot_additional_tseries_data.append(tseries)

    old_font_size = rcParams["font.size"]
    rcParams["font.size"] = 20
    fig, axs = plt.subplots(
        2, 1, gridspec_kw=dict(height_ratios=[1.5, 1]), figsize=(16, 12.5)
    )

    if PLOT_DATA:
        axs[0].loglog(
            plot_frequencies,
            asd_from_freq_series(
                interferometer.frequency_domain_strain[frequency_idxs],
                1 / interferometer.strain_data.duration,
            ),
            color=DATA_COLOR,
            label="Data",
            alpha=0.3,
        )
        axs[0].loglog(
            plot_frequencies,
            interferometer.amplitude_spectral_density_array[frequency_idxs],
            color=DATA_COLOR,
            label="ASD",
        )
        axs[1].plot(
            plot_times,
            np.fft.irfft(
                interferometer.whitened_frequency_domain_strain
                * np.sqrt(np.sum(interferometer.frequency_mask))
                / frequency_window_factor
            )[time_idxs],
            color=DATA_COLOR,
            alpha=0.3,
        )
        logger.debug("Plotted interferometer data.")

    delta = (1 + level) / 2
    upper_percentile = delta * 100
    lower_percentile = (1 - delta) * 100
    logger.debug(
        "Plotting posterior between the {} and {} percentiles".format(
            lower_percentile, upper_percentile
        )
    )

    lower_limit = 1e-21
    for ii, (fd_waveforms, td_waveforms) in enumerate(
        zip(fd_waveforms_used, td_waveforms_used)
    ):
        lower_limit = min(np.mean(fd_waveforms, axis=0)[0] / 1e3, lower_limit)
        axs[0].loglog(
            plot_frequencies,
            np.mean(fd_waveforms, axis=0),
            color=colors[ii],
            label=f"{labels[ii]} Mean reconstructed",
        )
        axs[0].fill_between(
            plot_frequencies,
            np.percentile(fd_waveforms, lower_percentile, axis=0),
            np.percentile(fd_waveforms, upper_percentile, axis=0),
            color=colors[ii],
            label=r"{} {}% credible interval".format(
                labels[ii], int(upper_percentile - lower_percentile)
            ),
            alpha=0.3,
        )
        axs[1].plot(
            plot_times,
            np.mean(td_waveforms, axis=0),
            color=colors[ii],
        )
        axs[1].fill_between(
            plot_times,
            np.percentile(td_waveforms, lower_percentile, axis=0),
            np.percentile(td_waveforms, upper_percentile, axis=0),
            color=colors[ii],
            alpha=0.3,
        )

    f_domain_x_label = "$f [\\mathrm{Hz}]$"
    f_domain_y_label = "$\\mathrm{ASD} \\left[\\mathrm{Hz}^{-1/2}\\right]$"
    t_domain_x_label = f"$t - {interferometer.strain_data.start_time} [s]$"
    t_domain_y_label = "Whitened Strain"
    if format == "html":
        fig.update_xaxes(title_text=f_domain_x_label, type="log", row=1)
        fig.update_yaxes(title_text=f_domain_y_label, type="log", row=1)
        fig.update_xaxes(title_text=t_domain_x_label, type="linear", row=2)
        fig.update_yaxes(title_text=t_domain_y_label, type="linear", row=2)
    else:
        if minimum_frequency is None:
            minimum_frequency = interferometer.minimum_frequency
        if maximum_frequency is None:
            maximum_frequency = interferometer.maximum_frequency
        axs[0].set_xlim(minimum_frequency, maximum_frequency)
        axs[1].set_xlim(start_time, end_time)
        axs[0].set_ylim(lower_limit)
        axs[0].set_xlabel(f_domain_x_label)
        axs[0].set_ylabel(f_domain_y_label)
        axs[1].set_xlabel(t_domain_x_label)
        axs[1].set_ylabel(t_domain_y_label)
        axs[0].legend(loc="upper left", ncol=2)

    if save:
        if filename is None:
            filename = os.path.join(
                outdir,
                run_label + f"_{interferometer.name}_waveform.{format}",
            )
        plt.tight_layout()
        safe_save_figure(fig=fig, filename=filename, format=format, dpi=600)
        plt.close()
        logger.debug(f"Waveform figure saved to {filename}")
        rcParams["font.size"] = old_font_size
    else:
        rcParams["font.size"] = old_font_size
        return fig
