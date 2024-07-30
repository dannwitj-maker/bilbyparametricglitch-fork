import ptfp


# def get_fd_and_td_whitened_waveform_posterior(
#     posterior: pd.DataFrame,
#     waveform_generator: WaveformGenerator,
#     interferometer: Interferometer,
#     frequency_idxs: np.ndarray,
#     time_idxs: np.ndarray,
#     prefix: str | None = None,
#     n_samples: int | None = None,
#     seed: int | None = None,
# ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:

#     if prefix is not None:
#         # If a prefix is offered, this is a glitch waveform and we need to do some manipulations
#         parameters = {
#             key.split(prefix)[1].strip("_"): parameter
#             for key, parameter in posterior.items()
#             if prefix in key
#         }
#         parameters["ra"] = np.zeros_like([x for x in parameters.values()][0])
#         parameters["dec"] = np.zeros_like([x for x in parameters.values()][0])
#         parameters["psi"] = np.zeros_like([x for x in parameters.values()][0])

#         if "_" in waveform_generator.waveform_arguments["glitch_mode_name"]:
#             waveform_generator.waveform_arguments[
#                 "glitch_mode_name"
#             ] = waveform_generator.waveform_arguments["glitch_mode_name"].split("_")[1]

#         posterior = pd.DataFrame(parameters)

#     if n_samples is None:
#         samples = copy.deepcopy(posterior)
#     elif n_samples > len(posterior):
#         logger.debug(
#             "Requested more waveform samples ({}) than we have "
#             "posterior samples ({})!".format(n_samples, len(posterior))
#         )
#         samples = copy.deepcopy(posterior)
#     else:
#         samples = posterior.sample(n_samples, replace=False, random_state=seed)

#     fd_waveforms = list()
#     td_waveforms = list()
#     for _, params in samples.iterrows():
#         fd_waveform, td_waveform = get_fd_and_td_whitened_waveform(
#             params=params,
#             waveform_generator=waveform_generator,
#             interferometer=interferometer,
#             time_idxs=time_idxs,
#         )
#         fd_waveforms.append(fd_waveform[frequency_idxs])
#         td_waveforms.append(td_waveform)
#     fd_waveforms = asd_from_freq_series(
#         fd_waveforms, 1 / interferometer.strain_data.duration
#     )
#     td_waveforms = np.array(td_waveforms)

#     return samples, fd_waveforms, td_waveforms
