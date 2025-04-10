import logging
import copy
import warnings
import attr
import numpy as np
from bilby.core.prior import Interped

# warnings.simplefilter("ignore", np.ComplexWarning)
logger = logging.getLogger(__name__)

from bilby.gw.likelihood.base import GravitationalWaveTransient
from bilby.gw.detector import (
    get_empty_interferometer,
    Interferometer,
    InterferometerList,
)
from bilby.gw.waveform_generator import WaveformGenerator
from .bilby_interface import _blank_frequency_domain_source_model
from .waveform_generator import GlitchWaveformGenerator

from typing import TYPE_CHECKING, Any, Dict, Optional


class JointGlitchGravitationalWaveTransient(GravitationalWaveTransient):
    """A likelihood class for joint inference of glitches and gravitational waves.

    Parameters
    ==========
    See superclass `GravitationalWaveTransient`
    cbc_waveform_generator : `bilby.gw.waveform_generator.WaveformGenerator`
        Equivalent to the 'waveform_generator' in `
    glitch_generators : Dict[str, `bilby.gw.waveform_generator.WaveformGenerator`]
        A dict mapping names of glitches to the waveform generators which generate them.
        Priors passed to these should have parameter names formatted as
        "glitch_{glitch_name}_{param_name}".
        The mode output should take the format
        "glitch_{IFO}_{glitch_name}", where here the {glitch_name} element is simply
        for bookkeeping.
    """

    @attr.s(slots=True, weakref_slot=False)
    class _CalculatedSNRs:
        d_inner_h = attr.ib(default=0j, converter=complex)
        d_inner_g = attr.ib(default=0j, converter=complex)
        g_inner_h = attr.ib(default=0j, converter=complex)
        optimal_gw_snr_squared = attr.ib(default=0, converter=float)
        optimal_glitch_snr_squared = attr.ib(default=0, converter=float)
        complex_matched_filter_gw_snr = attr.ib(default=0j, converter=complex)
        complex_matched_filter_glitch_snr = attr.ib(default=0j, converter=complex)

        d_inner_h_array = attr.ib(default=None)
        d_inner_g_array = attr.ib(default=None)
        g_inner_h_array = attr.ib(default=None)
        optimal_gw_snr_squared_array = attr.ib(default=None)
        optimal_glitch_snr_squared_array = attr.ib(default=None)

        def __add__(self, other_snr):
            new = copy.deepcopy(self)
            new += other_snr
            return new

        def __iadd__(self, other_snr):
            for key in self.__slots__:
                this = getattr(self, key)
                other = getattr(other_snr, key)
                if this is not None and other is not None:
                    setattr(self, key, this + other)
                elif this is None:
                    setattr(self, key, other)
            return self

        @property
        def snrs_as_sample(self) -> dict:
            """Get the SNRs of this object as a sample dictionary

            Returns
            =======
            dict
                The dictionary of SNRs labelled accordingly
            """
            return {
                "gw_matched_filter_snr": self.complex_matched_filter_gw_snr,
                "gw_optimal_snr": self.optimal_gw_snr_squared.real**0.5,
                "glitch_matched_filter_snr": self.complex_matched_filter_glitch_snr,
                "glitch_optimal_snr": self.optimal_glitch_snr_squared.real**0.5,
            }

    def __init__(
        self,
        interferometers: InterferometerList,
        cbc_waveform_generator: WaveformGenerator,
        glitch_generators: Dict[str, GlitchWaveformGenerator],
        time_marginalization: Optional[bool] = False,
        distance_marginalization: Optional[bool] = False,
        phase_marginalization: Optional[bool] = False,
        calibration_marginalization: Optional[bool] = False,
        priors: Optional[dict] = None,
        distance_marginalization_lookup_table: Optional[str] = None,
        calibration_lookup_table: Optional[str] = None,
        number_of_response_curves: Optional[int] = 1000,
        starting_index: Optional[int] = 0,
        jitter_time: Optional[bool] = True,
        reference_frame: Optional[str] = "sky",
        time_reference: Optional[str] = "geocenter",
    ):

        # For now, simplify by only applying distance marginalization
        # TODO if possible apply others (should be for e.g. time, shouldn't be for e.g. phase)
        assert (
            not time_marginalization
        ), "Time marginalization is not presently supported."
        assert (
            not phase_marginalization
        ), "Phase marginalization is not presently supported."
        assert (
            not calibration_marginalization
        ), "Calibration marginalization is not presently supported."
        if (
            cbc_waveform_generator.frequency_domain_source_model
            == _blank_frequency_domain_source_model
        ):
            assert (
                len(glitch_generators.keys()) != 0
            ), "Cannot pass no CBC Waveform AND no glitches"
        assert isinstance(
            glitch_generators, dict
        ), "glitch_generators should be a dict of glitch_name:glitch_generator"
        priors = self._set_glitch_geocent_priors(priors, glitch_generators)
        super(JointGlitchGravitationalWaveTransient, self).__init__(
            interferometers,
            cbc_waveform_generator,
            time_marginalization=False,
            distance_marginalization=distance_marginalization,
            phase_marginalization=False,
            calibration_marginalization=False,
            priors=priors,
            distance_marginalization_lookup_table=distance_marginalization_lookup_table,
            calibration_lookup_table=calibration_lookup_table,
            number_of_response_curves=number_of_response_curves,
            starting_index=starting_index,
            jitter_time=jitter_time,
            reference_frame=reference_frame,
            time_reference=time_reference,
        )
        self.glitch_generators = glitch_generators

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(interferometers={},\n\twaveform_generator={},\n\t "
            "glitch_generators={},\n\tdistance_marginalization={}, priors={})".format(
                self.interferometers,
                self.waveform_generator,
                self.glitch_generators,
                self.distance_marginalization,
                self.priors,
            )
        )

    def log_likelihood_ratio(self):
        waveform_polarizations = dict()
        # assert self.glitch_generators != dict(), "At least one glitch generator\
        #     should be passed, else why are you using this?"
        for glitch_name, glitch_generator in self.glitch_generators.items():
            # Input parameters are formatted as "glitch_{glitch_name}_{param_name}"
            # We want the parameters associated with the glitch in question
            # Then, transform to what frequency domain strain will expect
            parameters_of_interest = {
                key.split(glitch_name)[1].strip("_"): parameter
                for key, parameter in self.parameters.items()
                if glitch_name in key
            }
            waveform_polarizations.update(
                glitch_generator.frequency_domain_strain(parameters_of_interest)
            )
        if (
            not self.waveform_generator.frequency_domain_source_model
            == _blank_frequency_domain_source_model
        ):
            # Conditional in case of glitch only
            cbc_polarizations = self.waveform_generator.frequency_domain_strain(
                self.parameters
            )
            if cbc_polarizations is None:
                return np.nan_to_num(-np.inf)
            else:
                waveform_polarizations.update(cbc_polarizations)
            self.parameters.update(self.get_sky_frame_parameters())

        total_snrs = self._CalculatedSNRs()

        for interferometer in self.interferometers:
            per_detector_snr = self.calculate_snrs(
                waveform_polarizations=waveform_polarizations,
                interferometer=interferometer,
            )

            total_snrs += per_detector_snr

        log_l = self.compute_log_likelihood_from_snrs(total_snrs)

        return float(log_l.real)

    def distance_marginalized_likelihood(
        self, d_inner_h, g_inner_h, h_inner_h, g_inner_g, d_inner_g
    ):
        d_inner_h_ref, g_inner_h_ref, h_inner_h_ref = self._setup_rho(
            d_inner_h, g_inner_h, h_inner_h
        )
        # The trick is that g_inner_h has the same D_L scaling as d_inner_h
        # Hence we just map d_inner_h -> d_inner_h - g_inner_h
        # Because the array is defined completely w/o caring about the values that's fine
        # Sticking this in *should* yield the likelihood equivalently, then we just add
        # glitch only terms afterwards
        d_inner_h_ref = np.real(d_inner_h_ref)
        g_inner_h_ref = np.real(g_inner_h_ref)
        # Call this kappa_sq per https://arxiv.org/pdf/1809.02293.pdf C.3
        kappa_sq = d_inner_h_ref - g_inner_h_ref

        dist_marg = self._interp_dist_margd_loglikelihood(kappa_sq, h_inner_h_ref)

        return dist_marg + d_inner_g - g_inner_g / 2

    def generate_posterior_sample_from_marginalized_likelihood(self):
        """
        Reconstruct the distance posterior from a run which used a likelihood
        which explicitly marginalised over time/distance/phase.

        See Eq. (C29-C32) of https://arxiv.org/abs/1809.02293

        Returns
        =======
        sample: dict
            Returns the parameters with new samples.

        Notes
        =====
        This involves a deepcopy of the signal to avoid issues with waveform
        caching, as the signal is overwritten in place.
        """
        if len(self._marginalized_parameters) > 0:
            signal_polarizations = copy.deepcopy(
                self.waveform_generator.frequency_domain_strain(self.parameters)
            )
        else:
            return self.parameters

        if self.distance_marginalization:
            new_distance = self.generate_distance_sample_from_marginalized_likelihood(
                signal_polarizations=signal_polarizations
            )
            self.parameters["luminosity_distance"] = new_distance
        return self.parameters.copy()

    def generate_distance_sample_from_marginalized_likelihood(
        self, signal_polarizations=None
    ):
        self.parameters.update(self.get_sky_frame_parameters())
        if signal_polarizations is None:
            signal_polarizations = self.waveform_generator.frequency_domain_strain(
                self.parameters
            )

        d_inner_h, g_inner_h, h_inner_h, _, _ = self._calculate_inner_products(
            signal_polarizations
        )

        d_inner_h_dist = (
            d_inner_h * self.parameters["luminosity_distance"] / self._distance_array
        )

        g_inner_h_dist = (
            g_inner_h * self.parameters["luminosity_distance"] / self._distance_array
        )

        h_inner_h_dist = (
            h_inner_h
            * self.parameters["luminosity_distance"] ** 2
            / self._distance_array**2
        )

        distance_log_like = (
            d_inner_h_dist.real - g_inner_h_dist.real - h_inner_h_dist.real / 2
        )

        distance_post = (
            np.exp(distance_log_like - max(distance_log_like))
            * self.distance_prior_array
        )

        new_distance = Interped(self._distance_array, distance_post).sample()

        self._rescale_signal(signal_polarizations, new_distance)
        return new_distance

    def _rescale_signal(self, signal, new_distance):
        for mode in signal:
            if "glitch" not in mode:
                signal[mode] *= self._ref_dist / new_distance

    def _calculate_inner_products(self, signal_polarizations):
        d_inner_h = 0
        g_inner_h = 0
        h_inner_h = 0
        d_inner_g = 0
        g_inner_g = 0
        for interferometer in self.interferometers:
            per_detector_snr = self.calculate_snrs(signal_polarizations, interferometer)

            d_inner_h += per_detector_snr.d_inner_h
            g_inner_h += per_detector_snr.g_inner_h
            h_inner_h += per_detector_snr.optimal_gw_snr_squared
            d_inner_g += per_detector_snr.d_inner_g
            g_inner_g += per_detector_snr.optimal_glitch_snr_squared

        return d_inner_h, g_inner_h, h_inner_h, d_inner_g, g_inner_g

    def calculate_snrs(
        self, waveform_polarizations, interferometer: Interferometer, return_array=True
    ):
        """
        Compute the snrs

        Parameters
        ==========
        waveform_polarizations: dict
            A dictionary of waveform polarizations and the corresponding array
        interferometer: bilby.gw.detector.Interferometer
            The bilby interferometer object

        """
        glitch_signal = np.zeros(
            interferometer.frequency_array.shape, dtype=np.complex128
        )
        for mode_name, waveform_polarization in waveform_polarizations.items():
            if "glitch" in mode_name and interferometer.name in mode_name:
                glitch_name = mode_name.split(interferometer.name)[-1].strip("_")
                glitch_signal += interferometer.get_detector_response(
                    {interferometer.name: waveform_polarization},
                    dict(
                        ra=0,
                        dec=0,
                        psi=0,
                        geocent_time=self.parameters[
                            f"glitch_{glitch_name}_geocent_time"
                        ],
                    ),
                )
        # In glitch only case, just make the GW signal 0s
        if (
            not self.waveform_generator.frequency_domain_source_model
            == _blank_frequency_domain_source_model
        ):
            gw_signal = interferometer.get_detector_response(
                {
                    k: v
                    for k, v in waveform_polarizations.items()
                    if k in ["plus", "cross"]
                },
                self.parameters,
            )
        else:
            gw_signal = np.zeros(
                interferometer.frequency_array.shape, dtype=np.complex128
            )

        d_inner_h = interferometer.inner_product(signal=gw_signal)
        d_inner_g = interferometer.inner_product(signal=glitch_signal)
        g_inner_h = interferometer.template_template_inner_product(
            signal_1=gw_signal, signal_2=glitch_signal
        )
        optimal_gw_snr_squared = interferometer.optimal_snr_squared(signal=gw_signal)
        optimal_glitch_snr_squared = interferometer.optimal_snr_squared(
            signal=glitch_signal
        )

        if optimal_gw_snr_squared != 0:
            complex_matched_filter_gw_snr = d_inner_h / (optimal_gw_snr_squared**0.5)
        else:
            complex_matched_filter_gw_snr = 0

        if optimal_glitch_snr_squared != 0:
            complex_matched_filter_glitch_snr = d_inner_g / (
                optimal_glitch_snr_squared**0.5
            )
        else:
            complex_matched_filter_glitch_snr = 0

        # These won't be set in cases w/o time or calib marginalization
        # Since we currently don't do either, just set to None
        # Can deal with this if/when those are adde
        d_inner_h_array = None
        d_inner_g_array = None
        g_inner_h_array = None
        optimal_glitch_snr_squared_array = None
        optimal_gw_snr_squared_array = None

        return self._CalculatedSNRs(
            d_inner_h=d_inner_h,
            d_inner_g=d_inner_g,
            g_inner_h=g_inner_h,
            optimal_gw_snr_squared=optimal_gw_snr_squared,
            optimal_glitch_snr_squared=optimal_glitch_snr_squared,
            complex_matched_filter_gw_snr=complex_matched_filter_gw_snr,
            complex_matched_filter_glitch_snr=complex_matched_filter_glitch_snr,
            d_inner_h_array=d_inner_h_array,
            d_inner_g_array=d_inner_g_array,
            g_inner_h_array=g_inner_h_array,
            optimal_glitch_snr_squared_array=optimal_glitch_snr_squared_array,
            optimal_gw_snr_squared_array=optimal_gw_snr_squared_array,
        )

    def _setup_rho(self, d_inner_h, g_inner_h, optimal_snr_squared):
        optimal_snr_squared_ref = (
            optimal_snr_squared.real
            * self.parameters["luminosity_distance"] ** 2
            / self._ref_dist**2.0
        )
        d_inner_h_ref = (
            d_inner_h * self.parameters["luminosity_distance"] / self._ref_dist
        )
        g_inner_h_ref = (
            g_inner_h * self.parameters["luminosity_distance"] / self._ref_dist
        )
        return d_inner_h_ref, g_inner_h_ref, optimal_snr_squared_ref

    @staticmethod
    def _set_glitch_geocent_priors(priors, glitch_generators):
        """Glitch priors will presumably be in detector time, but we want them in geocent"""
        # For each glitch generator, check there is already an ifo time prior, and it's the right ifo
        # If it is, then make that glitch's corresponding geocent time prior, assuming ra = dec = 0
        for glitch_name, glitch_generator in glitch_generators.items():
            # glitch mode_names will look like glitch_{ifo}_{glitch_name}
            ifo = glitch_generator.waveform_arguments["glitch_mode_name"].split("_")[1]
            if f"glitch_{glitch_name}_{ifo}_time" in priors.keys():
                priors[f"glitch_{glitch_name}_geocent_time"] = copy.copy(
                    priors[f"glitch_{glitch_name}_{ifo}_time"]
                )
                # Fiducial glitch "sky location" is ra=dec=0
                inteferometer = get_empty_interferometer(ifo)
                # We are converting from IFO time to geocenter time, so a -1 should be applied
                # Since the geocent prior will be small
                # we can just use the minimum for computing delay and this is basically correct
                time_shift = -1 * inteferometer.time_delay_from_geocenter(
                    0, 0, priors[f"glitch_{glitch_name}_geocent_time"].minimum
                )
                priors[f"glitch_{glitch_name}_geocent_time"].minimum += time_shift
                priors[f"glitch_{glitch_name}_geocent_time"].maximum += time_shift
                priors[f"glitch_{glitch_name}_geocent_time"].name = "geocent_time"
                priors[f"glitch_{glitch_name}_geocent_time"].latex_label = r"$t_{gc}$"
                priors.pop(f"glitch_{glitch_name}_{ifo}_time")
        assert (
            f"glitch_{glitch_name}_geocent_time" in priors.keys()
        ), "Must set a glitch time prior"
        return priors

    def compute_log_likelihood_from_snrs(self, total_snrs):
        if self.distance_marginalization:
            log_l = self.distance_marginalized_likelihood(
                d_inner_h=total_snrs.d_inner_h,
                g_inner_h=total_snrs.g_inner_h,
                h_inner_h=total_snrs.optimal_gw_snr_squared,
                g_inner_g=total_snrs.optimal_glitch_snr_squared,
                d_inner_g=total_snrs.d_inner_g,
            )
        else:
            log_l = (
                np.real(total_snrs.d_inner_h)
                + np.real(total_snrs.d_inner_g)
                - np.real(total_snrs.g_inner_h)
                - total_snrs.optimal_gw_snr_squared / 2
                - total_snrs.optimal_glitch_snr_squared / 2
            )
        return log_l

    @property
    def meta_data(self):
        metadata_dict = dict(
            interferometers=self.interferometers.meta_data,
            time_marginalization=self.time_marginalization,
            phase_marginalization=self.phase_marginalization,
            distance_marginalization=self.distance_marginalization,
            calibration_marginalization=self.calibration_marginalization,
            time_reference=self.time_reference,
            reference_frame=self._reference_frame_str,
            lal_version=self.lal_version,
            lalsimulation_version=self.lalsimulation_version,
            glitch_names=list(),
        )
        if (
            not self.waveform_generator.frequency_domain_source_model
            == _blank_frequency_domain_source_model
        ):
            cbc_metadata = dict(
                waveform_generator_class=self.waveform_generator.__class__,
                waveform_arguments=self.waveform_generator.waveform_arguments,
                frequency_domain_source_model=self.waveform_generator.frequency_domain_source_model,
                time_domain_source_model=self.waveform_generator.time_domain_source_model,
                parameter_conversion=self.waveform_generator.parameter_conversion,
                sampling_frequency=self.waveform_generator.sampling_frequency,
                duration=self.waveform_generator.duration,
                start_time=self.waveform_generator.start_time,
            )
            metadata_dict.update(cbc_metadata)
        for glitch_name, glitch_wf_generator in self.glitch_generators.items():
            glitch_metadata = {
                f"{glitch_name}_generator_class": glitch_wf_generator.__class__,
                f"{glitch_name}_generator_arguments": glitch_wf_generator.waveform_arguments,
                f"{glitch_name}_generator_frequency_domain_source_model": glitch_wf_generator.frequency_domain_source_model,
                f"{glitch_name}_generator_time_domain_source_model": glitch_wf_generator.time_domain_source_model,
                f"{glitch_name}_generator_parameter_conversion": glitch_wf_generator.parameter_conversion,
                f"{glitch_name}_generator_sampling_frequency": glitch_wf_generator.sampling_frequency,
                f"{glitch_name}_generator_duration": glitch_wf_generator.duration,
                f"{glitch_name}_generator_start_time": glitch_wf_generator.start_time,
            }
            metadata_dict.update(glitch_metadata)
            metadata_dict["glitch_names"].append(glitch_name)
            if "sampling_frequency" not in metadata_dict:
                # Assume in case of multiple glitch waveform generators
                # that they share a sampling frequency, duration, start_time
                metadata_dict.update(
                    dict(
                        sampling_frequency=glitch_wf_generator.sampling_frequency,
                        duration=glitch_wf_generator.duration,
                        start_time=glitch_wf_generator.start_time,
                    )
                )
        return metadata_dict
