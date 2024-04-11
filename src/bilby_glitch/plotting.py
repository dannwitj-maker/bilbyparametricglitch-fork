import os
import numpy as np
import pandas as pd
import copy
import gwpy

from astropy import units as u


from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
from seaborn import set_palette
from mpl_toolkits.axes_grid1 import make_axes_locatable

set_palette("colorblind")

from bilby.core.utils import latex_plot_format, infft, logger, safe_save_figure
from bilby.gw.detector import get_empty_interferometer, Interferometer
from bilby.gw.waveform_generator import WaveformGenerator
from bilby.gw.utils import asd_from_freq_series


def get_fd_and_td_whitened_waveform(
    params: dict,
    waveform_generator: WaveformGenerator,
    interferometer: Interferometer,
    time_idxs: np.array,
):
    params = dict(params)
    wf_pols = waveform_generator.frequency_domain_strain(params)
    fd_waveform = interferometer.get_detector_response(wf_pols, params)
    td_waveform = infft(
        fd_waveform
        * np.sqrt(2.0 / interferometer.sampling_frequency)
        / interferometer.amplitude_spectral_density_array,
        interferometer.sampling_frequency,
    )[time_idxs]

    return fd_waveform, td_waveform


def get_fd_and_td_whitened_waveform_posterior(
    posterior: pd.DataFrame,
    waveform_generator: WaveformGenerator,
    interferometer: Interferometer,
    frequency_idxs: np.ndarray,
    time_idxs: np.ndarray,
    prefix: str | None = None,
    n_samples: int | None = None,
    seed: int | None = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:

    if prefix is not None:
        # If a prefix is offered, this is a glitch waveform and we need to do some manipulations
        parameters = {
            key.split(prefix)[1].strip("_"): parameter
            for key, parameter in posterior.items()
            if prefix in key
        }
        parameters["ra"] = np.zeros_like([x for x in parameters.values()][0])
        parameters["dec"] = np.zeros_like([x for x in parameters.values()][0])
        parameters["psi"] = np.zeros_like([x for x in parameters.values()][0])

        if "_" in waveform_generator.waveform_arguments["glitch_mode_name"]:
            waveform_generator.waveform_arguments[
                "glitch_mode_name"
            ] = waveform_generator.waveform_arguments["glitch_mode_name"].split("_")[1]

        posterior = pd.DataFrame(parameters)

    if n_samples is None:
        samples = copy.deepcopy(posterior)
    elif n_samples > len(posterior):
        logger.debug(
            "Requested more waveform samples ({}) than we have "
            "posterior samples ({})!".format(n_samples, len(posterior))
        )
        samples = copy.deepcopy(posterior)
    else:
        samples = posterior.sample(n_samples, replace=False, random_state=seed)

    fd_waveforms = list()
    td_waveforms = list()
    for _, params in samples.iterrows():
        fd_waveform, td_waveform = get_fd_and_td_whitened_waveform(
            params=params,
            waveform_generator=waveform_generator,
            interferometer=interferometer,
            time_idxs=time_idxs,
        )
        fd_waveforms.append(fd_waveform[frequency_idxs])
        td_waveforms.append(td_waveform)
    fd_waveforms = asd_from_freq_series(
        fd_waveforms, 1 / interferometer.strain_data.duration
    )
    td_waveforms = np.array(td_waveforms)

    return samples, fd_waveforms, td_waveforms


@latex_plot_format
def plot_spectrogram(
    timeseries: gwpy.timeseries.TimeSeries,
    title: Optional[str] = None,
    frequency_range: Tuple[float, float] = (16, 100),
    spectrogram_color_maximum: float = 25,
    q_value: int = 20,
    figsize=(16, 8),
    outseg=None,
    fig=None,
    ax=None,
    whiten=True,
    cmap="viridis",
    scale="log",
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    # Make a spectrogram
    qspecgram = timeseries.q_transform(
        qrange=[
            q_value,
            q_value,
        ],
        frange=frequency_range,
        outseg=outseg,
        whiten=whiten,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    colormesh = ax.pcolormesh(
        qspecgram,
        vmin=0,
        vmax=spectrogram_color_maximum,
        cmap=cmap,
        rasterized=True,
        norm="linear",
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0.225)
    fig.colorbar(colormesh, cax=cax, orientation="horizontal")
    cax.set_xlabel("Normalized Energy", labelpad=-60)

    ax.grid(visible=True, which="minor")
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(10))
    ax.yaxis.set_minor_formatter("{y:.0f}")
    ax.set_yscale(scale)
    if title is not None:
        ax.set_title(title)

    return fig, ax, cax


@latex_plot_format
def plot_interferometer_reconstruction(
    interferometer: Interferometer,
    waveform_generators: List[WaveformGenerator],
    posteriors: List[pd.DataFrame],
    labels: List[str] | None,
    prefixes: List[str | None] | None = None,
    fixed_waveforms: Optional[List[gwpy.timeseries.TimeSeries]] = None,
    colors: Optional[List[str]] = None,
    level: Optional[float] = 0.9,
    n_samples: Optional[int] = None,
    save: Optional[bool] = True,
    format: Optional[str] = "pdf",
    central_time: Optional[float] = None,
    data_start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    minimum_frequency: Optional[float] = None,
    maximum_frequency: Optional[float] = None,
    duration: Optional[int] = None,
    sampling_frequency: Optional[int] = None,
    start_time: Optional[float] = None,
    filename: Optional[str] = None,
    outdir: str = os.getcwd(),
    run_label: str = "analysis",
    add_spectrogram: bool = False,
    q_value: float = 20,
    spectrogram_color_maximum: float = 25,
    residual_index: Optional[int] = None,
    plot_residual_generator: Optional[bool] = False,
    plot_start_time: Optional[float] = None,
    plot_end_time: Optional[float] = None,
    spectrogram_cmap="viridis",
    spectrogram_timeseries=None,
    no_legend=False,
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
    if plot_start_time is None:
        plot_start_delta = 0
    else:
        plot_start_delta = plot_start_time - start_time
    if plot_end_time is None:
        plot_end_delta = 0
    else:
        plot_end_delta = plot_end_time - end_time

    PLOT_RESIDUAL_DATA = residual_index is not None and PLOT_DATA
    if PLOT_RESIDUAL_DATA:
        data_subtraction_posterior = posteriors[residual_index]
        data_subtraction_waveform_generator = waveform_generators[residual_index]
        data_subtraction_prefix = prefixes[residual_index]
        # data_subtraction_label = labels[residual_index]
    if (not plot_residual_generator) and PLOT_RESIDUAL_DATA:
        del posteriors[residual_index]
        del waveform_generators[residual_index]
        del prefixes[residual_index]
        del labels[residual_index]

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
    plot_start_time = start_time + plot_start_delta
    plot_end_time = end_time + plot_end_delta
    plot_frequencies = interferometer.frequency_array[frequency_idxs]

    samples_set = list()
    fd_waveforms_set = list()
    td_waveforms_set = list()

    for (generator, posterior, prefix) in zip(
        waveform_generators, posteriors, prefixes
    ):
        samples, fd_waveforms, td_waveforms = get_fd_and_td_whitened_waveform_posterior(
            posterior=posterior,
            waveform_generator=generator,
            interferometer=interferometer,
            frequency_idxs=frequency_idxs,
            time_idxs=time_idxs,
            prefix=prefix,
            n_samples=n_samples,
        )

        fd_waveforms_set.append(fd_waveforms)
        td_waveforms_set.append(td_waveforms)
        samples_set.append(samples)

    if PLOT_RESIDUAL_DATA:
        (
            data_subtraction_samples,
            data_subtraction_fd_waveforms,
            data_subtraction_td_waveforms,
        ) = get_fd_and_td_whitened_waveform_posterior(
            posterior=data_subtraction_posterior,
            waveform_generator=data_subtraction_waveform_generator,
            interferometer=interferometer,
            frequency_idxs=frequency_idxs,
            time_idxs=time_idxs,
            prefix=data_subtraction_prefix,
            n_samples=n_samples,
        )

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
        2, 1, gridspec_kw=dict(height_ratios=[1, 1.25]), figsize=(16, 12.5)
    )

    if minimum_frequency is None:
        minimum_frequency = interferometer.minimum_frequency
    if maximum_frequency is None:
        maximum_frequency = interferometer.maximum_frequency
    delta = (1 + level) / 2
    upper_percentile = delta * 100
    lower_percentile = (1 - delta) * 100
    logger.debug(
        "Plotting posterior between the {} and {} percentiles".format(
            lower_percentile, upper_percentile
        )
    )

    lower_limit = 1e-21

    if PLOT_DATA:
        fd_data = asd_from_freq_series(
            interferometer.frequency_domain_strain[frequency_idxs],
            1 / interferometer.strain_data.duration,
        )
        td_data = np.fft.irfft(
            interferometer.whitened_frequency_domain_strain
            * np.sqrt(np.sum(interferometer.frequency_mask))
            / frequency_window_factor
        )[time_idxs]

        if PLOT_RESIDUAL_DATA:
            residual_td_data_posterior = np.repeat(
                np.reshape(td_data, (1, -1)), n_samples, axis=0
            )
            residual_td_data_posterior -= data_subtraction_td_waveforms

            residual_fd_data_posterior = np.repeat(
                np.reshape(fd_data, (1, -1)), n_samples, axis=0
            )
            residual_fd_data_posterior -= data_subtraction_fd_waveforms
            residual_fd_data_posterior = asd_from_freq_series(
                residual_fd_data_posterior, 1 / interferometer.duration
            )
        if add_spectrogram:
            if spectrogram_timeseries is not None:
                spectrogram_timeseries.times -= (
                    interferometer.strain_data.start_time * u.s
                )
            elif PLOT_RESIDUAL_DATA:
                spectrogram_timeseries = gwpy.timeseries.TimeSeries(
                    np.mean(residual_td_data_posterior, axis=0), times=plot_times
                )
            else:
                spectrogram_timeseries = interferometer.strain_data.to_gwpy_timeseries()
                spectrogram_timeseries.times -= (
                    interferometer.strain_data.start_time * u.s
                )

            fig, axs[0], cax = plot_spectrogram(
                spectrogram_timeseries,
                frequency_range=(minimum_frequency, maximum_frequency),
                spectrogram_color_maximum=spectrogram_color_maximum,
                q_value=q_value,
                outseg=(plot_start_time, plot_end_time),
                fig=fig,
                ax=axs[0],
                whiten=(not PLOT_RESIDUAL_DATA),
                cmap=spectrogram_cmap,
            )

        else:
            if PLOT_RESIDUAL_DATA:
                axs[0].loglog(
                    plot_frequencies,
                    np.mean(residual_fd_data_posterior, axis=0),
                    color=DATA_COLOR,
                    label="Residual Data Mean",
                    alpha=0.6,
                )
                axs[0].fill_between(
                    plot_frequencies,
                    np.percentile(residual_fd_data_posterior, lower_percentile, axis=0),
                    np.percentile(residual_fd_data_posterior, upper_percentile, axis=0),
                    color=DATA_COLOR,
                    label=r"Residual Data {}\% credible interval".format(
                        int(upper_percentile - lower_percentile)
                    ),
                    alpha=0.3,
                )
            else:
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
            cax = None
        if PLOT_RESIDUAL_DATA:
            axs[1].plot(
                plot_times,
                np.mean(residual_td_data_posterior, axis=0),
                color=DATA_COLOR,
                alpha=0.6,
            )
            axs[1].fill_between(
                plot_times,
                np.percentile(residual_td_data_posterior, lower_percentile, axis=0),
                np.percentile(residual_td_data_posterior, upper_percentile, axis=0),
                color=DATA_COLOR,
                alpha=0.3,
            )
        else:
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

    for ii, (fd_waveforms, td_waveforms) in enumerate(
        zip(fd_waveforms_set, td_waveforms_set)
    ):
        if not add_spectrogram:
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
                label=r"{} {}\% credible interval".format(
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

    axs[1].set_xlim(plot_start_time, plot_end_time)
    axs[1].set_xlabel(t_domain_x_label)
    axs[1].set_ylabel(t_domain_y_label)
    if not add_spectrogram:
        axs[0].set_xlim(minimum_frequency, maximum_frequency)
        axs[0].set_ylim(lower_limit)
        axs[0].set_xlabel(f_domain_x_label)
        axs[0].set_ylabel(f_domain_y_label)
        axs[0].legend(loc="upper left", ncol=2)
    else:
        axs[0].set_xlabel(t_domain_x_label)
        axs[0].set_ylabel(f_domain_x_label)

        legend_handles = []
        if PLOT_RESIDUAL_DATA:
            legend_handles.append(
                matplotlib.lines.Line2D([], [], color="C1", label="Residual Data Mean")
            )
            legend_handles.append(
                matplotlib.lines.Line2D(
                    [],
                    [],
                    color="C1",
                    alpha=0.3,
                    label=r"Residual Data {}\% credible interval".format(
                        int(upper_percentile - lower_percentile)
                    ),
                )
            )
        elif not no_legend:
            legend_handles.append(
                matplotlib.lines.Line2D(
                    [], [], color=DATA_COLOR, alpha=0.3, label="Data"
                )
            )
        for ii, _ in enumerate(fd_waveforms_set):
            if no_legend:
                continue
            legend_handles.append(
                matplotlib.lines.Line2D(
                    [], [], color=colors[ii], label=f"{labels[ii]} Mean reconstructed"
                )
            )
            legend_handles.append(
                matplotlib.lines.Line2D(
                    [],
                    [],
                    color=colors[ii],
                    alpha=0.3,
                    label=r"{} {}\% credible interval".format(
                        labels[ii], int(upper_percentile - lower_percentile)
                    ),
                )
            )

        axs[1].legend(handles=legend_handles, loc="lower right", ncol=2)

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
        return fig, axs, cax
