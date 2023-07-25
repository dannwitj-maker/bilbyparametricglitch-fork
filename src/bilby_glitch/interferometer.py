import numpy as np

import bilby.gw.detector.interferometer


class GlitchInterferometer(bilby.gw.detector.interferometer.Interferometer):
    """Class for the Interferometer, adapted to include glitch modes"""
    def get_detector_response(self, waveform_polarizations, parameters, frequencies=None):
        """ Get the detector response for a particular waveform, 
        potentially including glitch modes.

        Parameters
        ==========
        waveform_polarizations: dict
            polarizations of the waveform
        parameters: dict
            parameters describing position and time of arrival of the signal
        frequencies: array-like, optional
        The frequency values to evaluate the response at. If
        not provided, the response is computed using
        :code:`self.frequency_array`. If the frequencies are
        specified, no frequency masking is performed.
        Returns
        =======
        array_like: A 3x3 array representation of the detector response (signal observed in the interferometer)
        """
        if frequencies is None:
            frequencies = self.frequency_array[self.frequency_mask]
            mask = self.frequency_mask
        else:
            mask = np.ones(len(frequencies), dtype=bool)

        signal = {}
        for mode in waveform_polarizations.keys():
            if "glitch" in mode:
                # If it's a glitch mode, check if it is in this detector or not
                # In the joint inference likelihood this should be handled already
                # But just in case
                if mode.split("_")[1] == self.name:
                    signal[mode] = waveform_polarizations[mode]
                else:
                    signal[mode] = np.zeros(waveform_polarizations[mode].shape)
            else:
                det_response = self.antenna_response(
                    parameters['ra'],
                    parameters['dec'],
                    parameters['geocent_time'],
                    parameters['psi'], mode)
                signal[mode] = waveform_polarizations[mode] * det_response
        signal_ifo = sum(signal.values()) * mask

        time_shift = self.time_delay_from_geocenter(
            parameters['ra'], parameters['dec'], parameters['geocent_time'])

        # Be careful to first subtract the two GPS times which are ~1e9 sec.
        # And then add the time_shift which varies at ~1e-5 sec
        dt_geocent = parameters['geocent_time'] - self.strain_data.start_time
        dt = dt_geocent + time_shift

        signal_ifo[mask] = signal_ifo[mask] * np.exp(-1j * 2 * np.pi * dt * frequencies)

        signal_ifo[mask] *= self.calibration_model.get_calibration_factor(
            frequencies, prefix='recalib_{}_'.format(self.name), **parameters
        )

        return signal_ifo
    
bilby.gw.detector.interferometer.Interferometer = GlitchInterferometer