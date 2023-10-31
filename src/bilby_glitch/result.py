from bilby.gw.result import CBCResult
import bilby
import os
import numpy as np
import pandas as pd
import copy

from bilby.core.utils import latex_plot_format, infft, logger, safe_save_figure
from bilby.gw.detector import get_empty_interferometer, Interferometer
from bilby.gw.utils import asd_from_freq_series


class JointCBCGlitchResult(CBCResult):
    def __init__(self, parent=None, **kwargs):
        self.parent = parent
        if self.parent is None:
            super(JointCBCGlitchResult, self).__init__(**kwargs)

    def __getattr__(self, name):
        if self.parent is not None:
            try:
                return getattr(self.parent, name)
            except AttributeError:
                raise AttributeError(
                    f"JointCBCGlitchResult object has no attribute '{name}'"
                )

    @latex_plot_format
    def plot_interferometer_waveform_posterior(
        self,
        interferometer,
        glitch_prefix,
        glitch_waveform_generator=None,
        cbc_waveform_generator=None,
        level=0.9,
        n_samples=None,
        save=True,
        format="png",
        start_time=None,
        end_time=None,
        minimum_frequency=None,
        maximum_frequency=None,
        arch_to_plot=None,
        filename=None,
    ):
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
        start_time: float, optional
            the amount of time before merger to begin the time domain plot.
            the merger time is defined as the mean of the geocenter time
            posterior. Default is - 0.4
        end_time: float, optional
            the amount of time before merger to end the time domain plot.
            the merger time is defined as the mean of the geocenter time
            posterior. Default is 0.2

        Returns
        =======
        fig: figure-handle, only is save=False

        Notes
        -----
        To reduce the memory footprint we decimate the frequency domain
        waveforms to have ~4000 entries. This should be sufficient for decent
        resolution.
        """

        DATA_COLOR = "#ff7f0e"
        GLITCH_WAVEFORM_COLOR = "#328C53"
        CBC_WAVEFORM_COLOR = "#1f77b4"
        INJECTION_COLOR = "#000000"

        if glitch_waveform_generator is not None:
            glitch_waveform_generator = copy.deepcopy(glitch_waveform_generator)
        else:
            glitch_waveform_generator = self.joint_waveform_generators[glitch_prefix]
        if cbc_waveform_generator is not None:
            cbc_waveform_generator = copy.deepcopy(cbc_waveform_generator)

        if format == "html":
            try:
                import plotly.graph_objects as go
                from plotly.offline import plot
                from plotly.subplots import make_subplots
            except ImportError:
                logger.warning(
                    "HTML plotting requested, but plotly cannot be imported, "
                    "falling back to png format for waveform plot."
                )
                format = "png"

        if isinstance(interferometer, str):
            interferometer = get_empty_interferometer(interferometer)
            interferometer.set_strain_data_from_zero_noise(
                sampling_frequency=self.sampling_frequency,
                duration=self.duration,
                start_time=self.start_time,
            )
            PLOT_DATA = False
        elif not isinstance(interferometer, Interferometer):
            raise TypeError("interferometer must be either str or Interferometer")
        else:
            PLOT_DATA = True
        logger.info(f"Generating waveform figure for {interferometer.name}")

        if n_samples is None:
            samples = self.posterior
        elif n_samples > len(self.posterior):
            logger.debug(
                "Requested more waveform samples ({}) than we have "
                "posterior samples ({})!".format(n_samples, len(self.posterior))
            )
            samples = self.posterior
        else:
            samples = self.posterior.sample(n_samples, replace=False)

        glitch_parameters_of_interest = {
            key.split(glitch_prefix)[1].strip("_"): parameter
            for key, parameter in samples.items()
            if glitch_prefix in key
        }
        if arch_to_plot is not None:
            for key in glitch_parameters_of_interest.keys():
                if "amp" in key and str(arch_to_plot) not in key:
                    glitch_parameters_of_interest[key] = 0
        glitch_parameters_of_interest["ra"] = np.zeros_like(samples["ra"])
        glitch_parameters_of_interest["dec"] = np.zeros_like(samples["dec"])
        glitch_parameters_of_interest["psi"] = np.zeros_like(samples["psi"])

        glitch_samples = pd.DataFrame(glitch_parameters_of_interest)

        if cbc_waveform_generator is not None:
            cbc_parameters_of_interest = {
                key: val for key, val in samples.items() if glitch_prefix not in key
            }
            cbc_samples = pd.DataFrame(cbc_parameters_of_interest)

        if start_time is None:
            start_time = -0.4
        start_time = np.mean(glitch_samples.geocent_time) + start_time
        if end_time is None:
            end_time = 0.2
        end_time = np.mean(glitch_samples.geocent_time) + end_time
        if format == "html":
            start_time = -np.inf
            end_time = np.inf
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

        if format == "html":
            fig = make_subplots(
                rows=2,
                cols=1,
                row_heights=[0.5, 0.5],
            )
            fig.update_layout(
                template="plotly_white",
                font=dict(
                    family="Computer Modern",
                ),
            )
        else:
            import matplotlib.pyplot as plt
            from matplotlib import rcParams

            old_font_size = rcParams["font.size"]
            rcParams["font.size"] = 20
            fig, axs = plt.subplots(
                2, 1, gridspec_kw=dict(height_ratios=[1.5, 1]), figsize=(16, 12.5)
            )

        if PLOT_DATA:
            if format == "html":
                fig.add_trace(
                    go.Scatter(
                        x=plot_frequencies,
                        y=asd_from_freq_series(
                            interferometer.frequency_domain_strain[frequency_idxs],
                            1 / interferometer.strain_data.duration,
                        ),
                        fill=None,
                        mode="lines",
                        line_color=DATA_COLOR,
                        opacity=0.5,
                        name="Data",
                        legendgroup="data",
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=plot_frequencies,
                        y=interferometer.amplitude_spectral_density_array[
                            frequency_idxs
                        ],
                        fill=None,
                        mode="lines",
                        line_color=DATA_COLOR,
                        opacity=0.8,
                        name="ASD",
                        legendgroup="asd",
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=plot_times,
                        y=np.fft.irfft(
                            interferometer.whitened_frequency_domain_strain
                            * np.sqrt(np.sum(interferometer.frequency_mask))
                            / frequency_window_factor
                        )[time_idxs],
                        fill=None,
                        mode="lines",
                        line_color=DATA_COLOR,
                        opacity=0.5,
                        name="Data",
                        legendgroup="data",
                        showlegend=False,
                    ),
                    row=2,
                    col=1,
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

        glitch_fd_waveforms = list()
        glitch_td_waveforms = list()
        for _, params in glitch_samples.iterrows():
            params = dict(params)
            glitch_wf_pols = glitch_waveform_generator.frequency_domain_strain(params)
            glitch_fd_waveform = interferometer.get_detector_response(
                glitch_wf_pols, params
            )
            glitch_fd_waveforms.append(glitch_fd_waveform[frequency_idxs])
            glitch_td_waveform = infft(
                glitch_fd_waveform
                * np.sqrt(2.0 / interferometer.sampling_frequency)
                / interferometer.amplitude_spectral_density_array,
                interferometer.sampling_frequency,
            )[time_idxs]
            glitch_td_waveforms.append(glitch_td_waveform)
        glitch_fd_waveforms = asd_from_freq_series(
            glitch_fd_waveforms, 1 / interferometer.strain_data.duration
        )
        glitch_td_waveforms = np.array(glitch_td_waveforms)

        if cbc_waveform_generator is not None:
            cbc_fd_waveforms = list()
            cbc_td_waveforms = list()
            for _, params in cbc_samples.iterrows():
                params = dict(params)
                cbc_wf_pols = cbc_waveform_generator.frequency_domain_strain(params)
                cbc_fd_waveform = interferometer.get_detector_response(
                    cbc_wf_pols, params
                )
                cbc_fd_waveforms.append(cbc_fd_waveform[frequency_idxs])
                cbc_td_waveform = infft(
                    cbc_fd_waveform
                    * np.sqrt(2.0 / interferometer.sampling_frequency)
                    / interferometer.amplitude_spectral_density_array,
                    self.sampling_frequency,
                )[time_idxs]
                cbc_td_waveforms.append(cbc_td_waveform)
            cbc_fd_waveforms = asd_from_freq_series(
                cbc_fd_waveforms, 1 / interferometer.strain_data.duration
            )
            cbc_td_waveforms = np.array(cbc_td_waveforms)

        delta = (1 + level) / 2
        upper_percentile = delta * 100
        lower_percentile = (1 - delta) * 100
        logger.debug(
            "Plotting posterior between the {} and {} percentiles".format(
                lower_percentile, upper_percentile
            )
        )

        if format == "html":
            fig.add_trace(
                go.Scatter(
                    x=plot_frequencies,
                    y=np.median(glitch_fd_waveforms, axis=0),
                    fill=None,
                    mode="lines",
                    line_color=GLITCH_WAVEFORM_COLOR,
                    opacity=1,
                    name="Median reconstructed",
                    legendgroup="median",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=plot_frequencies,
                    y=np.percentile(glitch_fd_waveforms, lower_percentile, axis=0),
                    fill=None,
                    mode="lines",
                    line_color=GLITCH_WAVEFORM_COLOR,
                    opacity=0.1,
                    name="{:.2f}% credible interval".format(
                        upper_percentile - lower_percentile
                    ),
                    legendgroup="uncertainty",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=plot_frequencies,
                    y=np.percentile(glitch_fd_waveforms, upper_percentile, axis=0),
                    fill="tonexty",
                    mode="lines",
                    line_color=GLITCH_WAVEFORM_COLOR,
                    opacity=0.1,
                    name="{:.2f}% credible interval".format(
                        upper_percentile - lower_percentile
                    ),
                    legendgroup="uncertainty",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=plot_times,
                    y=np.median(glitch_td_waveforms, axis=0),
                    fill=None,
                    mode="lines",
                    line_color=GLITCH_WAVEFORM_COLOR,
                    opacity=1,
                    name="Median reconstructed",
                    legendgroup="median",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=plot_times,
                    y=np.percentile(glitch_td_waveforms, lower_percentile, axis=0),
                    fill=None,
                    mode="lines",
                    line_color=GLITCH_WAVEFORM_COLOR,
                    opacity=0.1,
                    name="{:.2f}% credible interval".format(
                        upper_percentile - lower_percentile
                    ),
                    legendgroup="uncertainty",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=plot_times,
                    y=np.percentile(glitch_td_waveforms, upper_percentile, axis=0),
                    fill="tonexty",
                    mode="lines",
                    line_color=GLITCH_WAVEFORM_COLOR,
                    opacity=0.1,
                    name="{:.2f}% credible interval".format(
                        upper_percentile - lower_percentile
                    ),
                    legendgroup="uncertainty",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
        else:
            lower_limit = np.mean(glitch_fd_waveforms, axis=0)[0] / 1e3
            axs[0].loglog(
                plot_frequencies,
                np.mean(glitch_fd_waveforms, axis=0),
                color=GLITCH_WAVEFORM_COLOR,
                label="Mean reconstructed",
            )
            axs[0].fill_between(
                plot_frequencies,
                np.percentile(glitch_fd_waveforms, lower_percentile, axis=0),
                np.percentile(glitch_fd_waveforms, upper_percentile, axis=0),
                color=GLITCH_WAVEFORM_COLOR,
                label=r"{}% credible interval".format(
                    int(upper_percentile - lower_percentile)
                ),
                alpha=0.3,
            )
            axs[1].plot(
                plot_times,
                np.mean(glitch_td_waveforms, axis=0),
                color=GLITCH_WAVEFORM_COLOR,
            )
            axs[1].fill_between(
                plot_times,
                np.percentile(glitch_td_waveforms, lower_percentile, axis=0),
                np.percentile(glitch_td_waveforms, upper_percentile, axis=0),
                color=GLITCH_WAVEFORM_COLOR,
                alpha=0.3,
            )

            if cbc_waveform_generator is not None:
                axs[0].loglog(
                    plot_frequencies,
                    np.mean(cbc_fd_waveforms, axis=0),
                    color=CBC_WAVEFORM_COLOR,
                    label="Mean reconstructed",
                )
                axs[0].fill_between(
                    plot_frequencies,
                    np.percentile(cbc_fd_waveforms, lower_percentile, axis=0),
                    np.percentile(cbc_fd_waveforms, upper_percentile, axis=0),
                    color=CBC_WAVEFORM_COLOR,
                    label=r"{}% credible interval".format(
                        int(upper_percentile - lower_percentile)
                    ),
                    alpha=0.3,
                )
                axs[1].plot(
                    plot_times,
                    np.mean(cbc_td_waveforms, axis=0),
                    color=CBC_WAVEFORM_COLOR,
                )
                axs[1].fill_between(
                    plot_times,
                    np.percentile(cbc_td_waveforms, lower_percentile, axis=0),
                    np.percentile(cbc_td_waveforms, upper_percentile, axis=0),
                    color=CBC_WAVEFORM_COLOR,
                    alpha=0.3,
                )

        if self.injection_parameters is not None:
            injection_glitch_parameters_of_interest = {
                key.split(glitch_prefix)[1].strip("_"): parameter
                for key, parameter in self.injection_parameters.items()
                if glitch_prefix in key
            }
            if arch_to_plot is not None:
                for key in injection_glitch_parameters_of_interest.keys():
                    if "amp" in key and str(arch_to_plot) not in key:
                        injection_glitch_parameters_of_interest[key] = 0
            injection_glitch_parameters_of_interest["ra"] = 0
            injection_glitch_parameters_of_interest["dec"] = 0
            injection_glitch_parameters_of_interest["psi"] = 0

            injection_glitch_parameters_of_interest[
                "geocent_time"
            ] = injection_glitch_parameters_of_interest[
                f"{interferometer.name}_time"
            ] - interferometer.time_delay_from_geocenter(
                0,
                0,
                injection_glitch_parameters_of_interest[f"{interferometer.name}_time"],
            )

            try:
                hf_inj = glitch_waveform_generator.frequency_domain_strain(
                    injection_glitch_parameters_of_interest
                )
                hf_inj_det = interferometer.get_detector_response(
                    hf_inj, injection_glitch_parameters_of_interest
                )
                ht_inj_det = infft(
                    hf_inj_det
                    * np.sqrt(2.0 / interferometer.sampling_frequency)
                    / interferometer.amplitude_spectral_density_array,
                    self.sampling_frequency,
                )[time_idxs]
                if format == "html":
                    fig.add_trace(
                        go.Scatter(
                            x=plot_frequencies,
                            y=asd_from_freq_series(
                                hf_inj_det[frequency_idxs],
                                1 / interferometer.strain_data.duration,
                            ),
                            fill=None,
                            mode="lines",
                            line=dict(color=INJECTION_COLOR, dash="dot"),
                            name="Injection",
                            legendgroup="injection",
                        ),
                        row=1,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=plot_times,
                            y=ht_inj_det,
                            fill=None,
                            mode="lines",
                            line=dict(color=INJECTION_COLOR, dash="dot"),
                            name="Injection",
                            legendgroup="injection",
                            showlegend=False,
                        ),
                        row=2,
                        col=1,
                    )
                else:
                    axs[0].loglog(
                        plot_frequencies,
                        asd_from_freq_series(
                            hf_inj_det[frequency_idxs],
                            1 / interferometer.strain_data.duration,
                        ),
                        color=INJECTION_COLOR,
                        label="Injection",
                        linestyle=":",
                    )
                    axs[1].plot(
                        plot_times, ht_inj_det, color=INJECTION_COLOR, linestyle=":"
                    )
                logger.debug("Plotted injection.")
            except IndexError as e:
                logger.info(f"Failed to plot injection with message {e}.")

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
                    self.outdir,
                    self.label
                    + f"_arch{arch_to_plot}_{interferometer.name}_waveform.{format}",
                )
            if format == "html":
                plot(fig, filename=filename, include_mathjax="cdn", auto_open=False)
            else:
                plt.tight_layout()
                safe_save_figure(fig=fig, filename=filename, format=format, dpi=600)
                plt.close()
            logger.debug(f"Waveform figure saved to {filename}")
            rcParams["font.size"] = old_font_size
        else:
            rcParams["font.size"] = old_font_size
            return fig
