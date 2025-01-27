import os
import inspect
import copy

from bilby_pipe.input import Input
import bilby.gw.source

from bilby_pipe.utils import (
    logger,
    get_function_from_string_path,
    BilbyPipeError,
    convert_string_to_dict,
    import_module,
)

from ..models.bilby_interface import (
    populate_unused_parameters,
    single_wavelet_frequency_domain,
    slow_scattering_frequency_domain,
)

from ..models.legacy_bilby_interface import (
    slow_scattering,
    slow_scattering_shifting_frequency,
    SLOW_SCATTERING_PARAMETERS,
    SLOW_SCATTERING_SHIFTING_PARAMETERS,
)


class GlitchInput(Input):
    """Superclass of input handlers - with glitch additions"""

    @property
    def injection_bilby_frequency_domain_source_model(self):
        """
        The bilby function to pass to the waveform_generator

        This can be a function defined in an external package.
        """
        if (
            self.injection_frequency_domain_source_model
            in bilby.gw.source.__dict__.keys()
        ):
            model = self._injection_frequency_domain_source_model
            logger.debug(f"Using the {model} source model")
            return bilby.gw.source.__dict__[model]
        elif "." in self.injection_frequency_domain_source_model:
            return get_function_from_string_path(
                self._injection_frequency_domain_source_model
            )
        else:
            raise BilbyPipeError(
                f"No source model {self._injection_frequency_domain_source_model} found."
            )

    @property
    def injection_frequency_domain_source_model(self):
        """String of which frequency domain source model to use"""
        return self._injection_frequency_domain_source_model

    @injection_frequency_domain_source_model.setter
    def injection_frequency_domain_source_model(self, frequency_domain_source_model):
        self._injection_frequency_domain_source_model = frequency_domain_source_model

    @property
    def joint_injection_files(self):
        return self._joint_injection_files

    @joint_injection_files.setter
    def joint_injection_files(self, injection_files: dict | str):
        if isinstance(injection_files, str):
            injection_files = convert_string_to_dict(injection_files)
        self._joint_injection_files = injection_files

    @property
    def joint_prior_files(self):
        """Prior files for joint priors, indexed by glitch index name"""
        return self._joint_prior_files

    @joint_prior_files.setter
    def joint_prior_files(self, joint_prior_files: dict | str | None):
        """Sets joint prior file paths

        Parameters
        ==========
        joint_prior_files : dict
            Prior file paths, indexed by glitch index_name
        """
        if joint_prior_files is None:
            self._joint_prior_files = None
        else:
            if isinstance(joint_prior_files, str):
                joint_prior_files = convert_string_to_dict(joint_prior_files)
            self._joint_prior_files = dict()
            for index_name, prior_file in joint_prior_files.items():
                if prior_file is None:
                    self._joint_prior_files[index_name] = None
                elif os.path.isfile(prior_file):
                    self._joint_prior_files[index_name] = prior_file
                elif os.path.isfile(os.path.basename(prior_file)):
                    self._joint_prior_files[index_name] = os.path.basename(prior_file)
                else:
                    raise FileNotFoundError(
                        f"For glitch {index_name},\
                        prior file {prior_file} is not available"
                    )

    @property
    def joint_prior_defaults(self):
        return getattr(self, "_joint_prior_defaults", dict())

    @joint_prior_defaults.setter
    def joint_prior_defaults(self, joint_prior_defaults):
        if isinstance(joint_prior_defaults, dict):
            self._joint_prior_defaults = joint_prior_defaults
        elif isinstance(joint_prior_defaults, str):
            self._joint_prior_defaults = convert_string_to_dict(joint_prior_defaults)
        elif isinstance(joint_prior_defaults, None):
            self._joint_prior_defaults = None
        else:
            raise TypeError(
                f"Could not interpret joint prior defaults {joint_prior_defaults}"
            )

    def _get_priors(self, add_time=True):
        """Construct the priors

        Parameters
        ----------
        add_time: bool
            If True, the time prior is constructed from either the
            prior file or the trigger time. If False (used for the overview
            page where a single time-prior doesn't make sense), this isn't
            added to the prior

        Returns
        -------
        prior: bilby.core.prior.PriorDict
            The generated prior
        """
        if self.default_prior in self.combined_default_prior_dicts.keys():
            prior_class = self.combined_default_prior_dicts[self.default_prior]
        elif "." in self.default_prior:
            prior_class = get_function_from_string_path(self.default_prior)
        else:
            raise ValueError("Unable to set prior: default_prior unavailable")

        if self.prior_dict is not None:
            priors = prior_class(dictionary=self.prior_dict)
        elif self.prior_file is not None:
            priors = prior_class(filename=self.prior_file)
        elif self.likelihood_is_joint:
            priors = prior_class(dictionary={})
            # A special case when likelihood is joint, and we can set blank CBC priors
            # This keeps bilby happy without having to pass an extra file
            priors["ra"] = 0
            priors["dec"] = 0
            priors["psi"] = 0
            priors["geocent_time"] = 0

        priors = self._update_default_prior_to_sky_frame_parameters(priors)

        if self.time_parameter in priors:
            logger.debug(f"Using {self.time_parameter} prior from prior_file")
        elif add_time:
            priors[self.time_parameter] = self.create_time_prior()
        else:
            logger.debug("No time prior available or requested")

        if self.calibration_model is not None:
            priors.update(self.calibration_prior)

        if self.likelihood_is_joint:
            self.joint_priors = dict()
            for index_name, prior_file in self.joint_prior_files.items():
                # Do the above procedure, over each prior in the joint prior dict
                assert (
                    index_name in self.joint_prior_defaults
                ), self.joint_prior_defaults
                prior_default = self.joint_prior_defaults[index_name]
                if prior_default in self.combined_default_prior_dicts.keys():
                    prior_class = self.combined_default_prior_dicts[prior_default]
                elif "." in prior_default:
                    prior_class = get_function_from_string_path(prior_default)
                else:
                    raise ValueError(
                        f"Unable to set prior for glitch {index_name}: default_prior unavailable"
                    )

                glitch_prior = prior_class(filename=prior_file)

                if hasattr(self, "joint_waveform_generator_source_models"):
                    if (
                        self.joint_waveform_generator_source_models[index_name]
                        == slow_scattering
                    ):
                        glitch_prior = populate_unused_parameters(
                            glitch_prior, full_parameter_list=SLOW_SCATTERING_PARAMETERS
                        )
                    if (
                        self.joint_waveform_generator_source_models[index_name]
                        == slow_scattering_shifting_frequency
                    ):
                        glitch_prior = populate_unused_parameters(
                            glitch_prior,
                            full_parameter_list=SLOW_SCATTERING_SHIFTING_PARAMETERS,
                        )

                # Now, the special bit for our purposes: add designator information for the glitch
                glitch_named_prior = dict()
                for key, val in glitch_prior.items():
                    glitch_named_prior[f"glitch_{index_name}_{key}"] = val

                self.joint_priors[index_name] = glitch_named_prior

                priors.update(glitch_named_prior)

        return priors

    @property
    def likelihood(self):
        self.search_priors = self.priors.copy()

        likelihood_kwargs = dict(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            priors=self.search_priors,
            phase_marginalization=self.phase_marginalization,
            distance_marginalization=self.distance_marginalization,
            distance_marginalization_lookup_table=self.distance_marginalization_lookup_table,
            time_marginalization=self.time_marginalization,
            reference_frame=self.reference_frame,
            time_reference=self.time_reference,
            calibration_marginalization=self.calibration_marginalization,
            calibration_lookup_table=self.calibration_lookup_table,
            number_of_response_curves=self.number_of_response_curves,
        )
        relative_binning_kwargs = dict(
            fiducial_parameters=self.fiducial_parameters,
            update_fiducial_parameters=self.update_fiducial_parameters,
            epsilon=self.epsilon,
        )
        if self.likelihood_is_joint:
            likelihood_kwargs["cbc_waveform_generator"] = likelihood_kwargs.pop(
                "waveform_generator"
            )
            likelihood_kwargs["glitch_generators"] = self.joint_waveform_generators

        if getattr(self, "likelihood_lookup_table", None) is not None:
            logger.debug("Using internally loaded likelihood_lookup_table")
            likelihood_kwargs["distance_marginalization_lookup_table"] = getattr(
                self, "likelihood_lookup_table"
            )

        if self.likelihood_type in ["GravitationalWaveTransient", "zero"]:
            Likelihood = bilby.gw.likelihood.GravitationalWaveTransient
            likelihood_kwargs.update(jitter_time=self.jitter_time)

        elif self.likelihood_type == "ROQGravitationalWaveTransient":
            Likelihood = bilby.gw.likelihood.ROQGravitationalWaveTransient
            likelihood_kwargs.update(
                self.roq_likelihood_kwargs, jitter_time=self.jitter_time
            )
        elif self.likelihood_type == "RelativeBinningGravitationalWaveTransient":
            Likelihood = bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient
            likelihood_kwargs.update(relative_binning_kwargs)
        elif self.likelihood_type == "MBGravitationalWaveTransient":
            Likelihood = bilby.gw.likelihood.MBGravitationalWaveTransient
            likelihood_kwargs.update(self.multiband_likelihood_kwargs)
            likelihood_kwargs.update(self.extra_likelihood_kwargs)
        elif "." in self.likelihood_type:
            split_path = self.likelihood_type.split(".")
            module = ".".join(split_path[:-1])
            likelihood_class = split_path[-1]
            Likelihood = getattr(import_module(module), likelihood_class)
            likelihood_kwargs.update(self.extra_likelihood_kwargs)
            if "roq" in self.likelihood_type.lower():
                likelihood_kwargs.update(self.roq_likelihood_kwargs)
            elif "relative" in self.likelihood_type.lower():
                likelihood_kwargs.update(relative_binning_kwargs)
            if "multiband" in self.likelihood_type.lower():
                likelihood_kwargs.update(self.multiband_likelihood_kwargs)
        else:
            raise ValueError("Unknown Likelihood class {}")

        likelihood_kwargs = {
            key: likelihood_kwargs[key]
            for key in likelihood_kwargs
            if key in inspect.getfullargspec(Likelihood.__init__).args
        }

        logger.debug(
            f"Initialise likelihood {Likelihood} with kwargs: \n{likelihood_kwargs}"
        )

        if likelihood_kwargs.get("update_fiducial_parameters", False):
            for key in self.calibration_prior:
                self.search_priors[key] = self.calibration_prior[key].rescale(0.5)

        likelihood = Likelihood(**likelihood_kwargs)

        if likelihood_kwargs.get("update_fiducial_parameters", False):
            for key in self.calibration_prior:
                self.search_priors[key] = self.calibration_prior[key]

        # If requested, use a zero likelihood: for testing purposes
        if self.likelihood_type == "zero":
            logger.debug("Using a ZeroLikelihood")
            likelihood = bilby.core.likelihood.ZeroLikelihood(likelihood)

        return likelihood

    @property
    def extra_waveform_generator_kwargs(self):
        return self._extra_waveform_generator_kwargs

    @extra_waveform_generator_kwargs.setter
    def extra_waveform_generator_kwargs(self, waveform_generator_kwargs):
        if isinstance(waveform_generator_kwargs, str):
            waveform_generator_kwargs = convert_string_to_dict(
                waveform_generator_kwargs
            )
        elif waveform_generator_kwargs is None:
            waveform_generator_kwargs = dict()
        elif not isinstance(waveform_generator_kwargs, dict):
            raise TypeError(
                f"Type {type(waveform_generator_kwargs)} not understood for extra waveform generator skwargs."
            )
        self._extra_waveform_generator_kwargs = waveform_generator_kwargs

    @property
    def joint_waveform_generators(self):
        if getattr(self, "_joint_waveform_generators", None) is None:
            self._joint_waveform_generators = dict()
            for (
                index_name,
                waveform_generator_class,
            ) in self.joint_waveform_generator_classes.items():
                waveform_generator = waveform_generator_class(
                    duration=self.duration,
                    start_time=self.start_time,
                    sampling_frequency=self.sampling_frequency,
                    frequency_domain_source_model=self.joint_waveform_generator_source_models[
                        index_name
                    ],
                    parametric_frequency_domain_source_model=self.joint_waveform_generator_parametric_models[
                        index_name
                    ],
                    parameter_conversion=bilby.gw.conversion.identity_map_conversion,
                    waveform_arguments=self.joint_waveform_generator_model_kwargs[
                        index_name
                    ],
                    **self.joint_waveform_generator_kwargs[index_name],
                )
                logger.info(f"For glitch {index_name}:")
                logger.info(
                    f"Using waveform arguments: {self.joint_waveform_generator_model_kwargs[index_name]}"
                )
                logger.info(
                    f"Using generator arguments: {self.joint_waveform_generator_kwargs[index_name]}"
                )
                logger.info(
                    f"Parameter conversion function is: {self.parameter_conversion}"
                )

                self._joint_waveform_generators[index_name] = waveform_generator
        return self._joint_waveform_generators

    @joint_waveform_generators.setter
    def joint_waveform_generators(self, joint_waveform_generators):
        self._joint_waveform_generators = joint_waveform_generators

    @property
    def joint_waveform_generator_classes(self):
        return self._joint_waveform_generator_classes

    @joint_waveform_generator_classes.setter
    def joint_waveform_generator_classes(self, generator_classes: dict | str):
        self._joint_waveform_generator_classes = dict()
        if isinstance(generator_classes, str):
            generator_classes = convert_string_to_dict(generator_classes)
        for index_name, class_name in generator_classes.items():
            if "." in class_name:
                module = ".".join(class_name.split(".")[:-1])
                class_name = class_name.split(".")[-1]
            else:
                module = "bilby.gw.waveform_generator"
            wfg_class = getattr(import_module(module), class_name, None)
            if wfg_class is not None:
                self._joint_waveform_generator_classes[index_name] = wfg_class
            else:
                raise BilbyPipeError(
                    f"Cannot import waveform generator class {module}.{class_name}"
                )

    @property
    def joint_waveform_generator_source_models(self):
        return self._joint_waveform_generator_source_models

    @property
    def joint_waveform_generator_parametric_models(self):
        return self._joint_waveform_generator_parametric_models

    @joint_waveform_generator_parametric_models.setter
    def joint_waveform_generator_parametric_models(self, parametric_models: str | dict):
        if not hasattr(self, "_joint_waveform_generator_parametric_models"):
            self._joint_waveform_generator_parametric_models = dict()
        if isinstance(parametric_models, str):
            parametric_models = convert_string_to_dict(parametric_models)
        if parametric_models is None:
            parametric_models = {}
        for index_name, model_name in parametric_models.items():
            if model_name == "slow_scattering":
                self._joint_waveform_generator_parametric_models[
                    index_name
                ] = slow_scattering_frequency_domain
            elif model_name == "single_wavelet":
                self._joint_waveform_generator_parametric_models[
                    index_name
                ] = single_wavelet_frequency_domain
            elif "." in model_name:
                self._joint_waveform_generator_parametric_models[
                    index_name
                ] = get_function_from_string_path(model_name)
        self._cleanup_joint_models()

    @joint_waveform_generator_source_models.setter
    def joint_waveform_generator_source_models(self, source_models: str | dict):
        if not hasattr(self, "_joint_waveform_generator_source_models"):
            self._joint_waveform_generator_source_models = dict()
        if isinstance(source_models, str):
            source_models = convert_string_to_dict(source_models)
        if source_models is None:
            source_models = {}
        for index_name, model_name in source_models.items():
            if "." in model_name:
                self._joint_waveform_generator_source_models[
                    index_name
                ] = get_function_from_string_path(model_name)
            else:
                self._joint_waveform_generator_source_models[index_name] = None
        self._cleanup_joint_models()

    @staticmethod
    def _set_none_where_keys_are_missing(keys, update_dict):
        missing_keys = set(keys) - set(update_dict.keys())
        for key in missing_keys:
            update_dict[key] = None
        return update_dict

    def _cleanup_joint_models(self):
        if not hasattr(self, "_joint_waveform_generator_source_models"):
            self._joint_waveform_generator_source_models = {}
        if not hasattr(self, "_joint_waveform_generator_parametric_models"):
            self._joint_waveform_generator_parametric_models = {}
        self._joint_waveform_generator_parametric_models = (
            self._set_none_where_keys_are_missing(
                self._joint_waveform_generator_source_models.keys(),
                self._joint_waveform_generator_parametric_models,
            )
        )
        self._joint_waveform_generator_source_models = (
            self._set_none_where_keys_are_missing(
                self._joint_waveform_generator_parametric_models.keys(),
                self._joint_waveform_generator_source_models,
            )
        )

    @property
    def joint_waveform_generator_kwargs(self):
        return self._joint_waveform_generator_kwargs

    @joint_waveform_generator_kwargs.setter
    def joint_waveform_generator_kwargs(self, waveform_generator_kwargs):
        if isinstance(waveform_generator_kwargs, str):
            waveform_generator_kwargs = convert_string_to_dict(
                waveform_generator_kwargs
            )
        elif waveform_generator_kwargs is None:
            waveform_generator_kwargs = dict()
        elif not isinstance(waveform_generator_kwargs, dict):
            raise TypeError(
                f"Type {type(waveform_generator_kwargs)} not understood for joint waveform generator kwargs."
            )
        self._joint_waveform_generator_kwargs = waveform_generator_kwargs

    @property
    def joint_waveform_generator_model_kwargs(self):
        return self._joint_waveform_generator_model_kwargs

    @joint_waveform_generator_model_kwargs.setter
    def joint_waveform_generator_model_kwargs(self, waveform_model_kwargs):
        if isinstance(waveform_model_kwargs, str):
            waveform_model_kwargs = convert_string_to_dict(waveform_model_kwargs)
        elif waveform_model_kwargs is None:
            waveform_model_kwargs = dict()
        elif not isinstance(waveform_model_kwargs, dict):
            raise TypeError(
                f"Type {type(waveform_model_kwargs)} not understood for joint waveform generator kwargs."
            )
        self._joint_waveform_generator_model_kwargs = waveform_model_kwargs

    @property
    def joint_injection_waveform_generators(self):
        if getattr(self, "_joint_injection_waveform_generators", None) is None:
            self._joint_injection_waveform_generators = dict()
            for (
                index_name,
                waveform_generator_class,
            ) in self.joint_injection_waveform_generator_classes.items():
                waveform_generator = waveform_generator_class(
                    duration=self.duration,
                    start_time=self.start_time,
                    sampling_frequency=self.sampling_frequency,
                    frequency_domain_source_model=self.joint_injection_waveform_generator_source_models[
                        index_name
                    ],
                    parametric_frequency_domain_source_model=self.joint_injection_waveform_generator_parametric_models[
                        index_name
                    ],
                    parameter_conversion=bilby.gw.conversion.identity_map_conversion,
                    waveform_arguments=self.joint_injection_waveform_generator_model_kwargs[
                        index_name
                    ],
                    direct_injection=True,
                    **self.joint_injection_waveform_generator_kwargs[index_name],
                )
                logger.info(f"For glitch {index_name}:")
                logger.info(
                    f"Using waveform arguments: {self.joint_injection_waveform_generator_model_kwargs[index_name]}"
                )
                logger.info(
                    f"Using generator arguments: {self.joint_injection_waveform_generator_kwargs[index_name]}"
                )
                logger.info(
                    f"Parameter conversion function is: {self.parameter_conversion}"
                )
                logger.info(
                    "Parametric frequency domain source model is: "
                    f"{self.joint_injection_waveform_generator_parametric_models}"
                )
                logger.info(
                    f"Frequency domain source model is: {self.joint_injection_waveform_generator_source_models}"
                )

                self._joint_injection_waveform_generators[
                    index_name
                ] = waveform_generator
        return self._joint_injection_waveform_generators

    @joint_injection_waveform_generators.setter
    def joint_injection_waveform_generators(self, joint_injection_waveform_generators):
        self._joint_injection_waveform_generators = joint_injection_waveform_generators

    @property
    def joint_injection_waveform_generator_classes(self):
        return self._joint_injection_waveform_generator_classes

    @joint_injection_waveform_generator_classes.setter
    def joint_injection_waveform_generator_classes(self, generator_classes: dict | str):
        self._joint_injection_waveform_generator_classes = dict()
        if isinstance(generator_classes, str):
            generator_classes = convert_string_to_dict(generator_classes)
        for index_name, class_name in generator_classes.items():
            if "." in class_name:
                module = ".".join(class_name.split(".")[:-1])
                class_name = class_name.split(".")[-1]
            else:
                module = "bilby.gw.waveform_generator"
            wfg_class = getattr(import_module(module), class_name, None)
            if wfg_class is not None:
                self._joint_injection_waveform_generator_classes[index_name] = wfg_class
            else:
                raise BilbyPipeError(
                    f"Cannot import waveform generator class {module}.{class_name}"
                )

    @property
    def joint_injection_waveform_generator_source_models(self):
        return self._joint_injection_waveform_generator_source_models

    @property
    def joint_injection_waveform_generator_parametric_models(self):
        return self._joint_injection_waveform_generator_parametric_models

    @joint_injection_waveform_generator_parametric_models.setter
    def joint_injection_waveform_generator_parametric_models(
        self, parametric_models: str | dict
    ):
        if not hasattr(self, "_joint_injection_waveform_generator_parametric_models"):
            self._joint_injection_waveform_generator_parametric_models = dict()
        if isinstance(parametric_models, str):
            parametric_models = convert_string_to_dict(parametric_models)
        if parametric_models is None:
            parametric_models = {}
        for index_name, model_name in parametric_models.items():
            if model_name == "slow_scattering":
                self._joint_injection_waveform_generator_parametric_models[
                    index_name
                ] = slow_scattering_frequency_domain
            elif model_name == "single_wavelet":
                self._joint_injection_waveform_generator_parametric_models[
                    index_name
                ] = single_wavelet_frequency_domain
            elif "." in model_name:
                self._joint_injection_waveform_generator_parametric_models[
                    index_name
                ] = get_function_from_string_path(model_name)
        self._cleanup_joint_injection_models()

    @joint_injection_waveform_generator_source_models.setter
    def joint_injection_waveform_generator_source_models(
        self, source_models: str | dict
    ):
        if not hasattr(self, "_joint_injection_waveform_generator_source_models"):
            self._joint_injection_waveform_generator_source_models = dict()
        if isinstance(source_models, str):
            source_models = convert_string_to_dict(source_models)
        if source_models is None:
            source_models = {}
        for index_name, model_name in source_models.items():
            if "." in model_name:
                self._joint_injection_waveform_generator_source_models[
                    index_name
                ] = get_function_from_string_path(model_name)
            else:
                self._joint_injection_waveform_generator_source_models[
                    index_name
                ] = None
        self._cleanup_joint_injection_models()

    def _cleanup_joint_injection_models(self):
        if not hasattr(self, "_joint_injection_waveform_generator_source_models"):
            self._joint_injection_waveform_generator_source_models = {}
        if not hasattr(self, "_joint_injection_waveform_generator_parametric_models"):
            self._joint_injection_waveform_generator_parametric_models = {}
        self._joint_injection_waveform_generator_parametric_models = (
            self._set_none_where_keys_are_missing(
                self._joint_injection_waveform_generator_source_models.keys(),
                self._joint_injection_waveform_generator_parametric_models,
            )
        )
        self._joint_injection_waveform_generator_source_models = (
            self._set_none_where_keys_are_missing(
                self._joint_injection_waveform_generator_parametric_models.keys(),
                self._joint_injection_waveform_generator_source_models,
            )
        )

    @property
    def joint_injection_waveform_generator_kwargs(self):
        return self._joint_injection_waveform_generator_kwargs

    @joint_injection_waveform_generator_kwargs.setter
    def joint_injection_waveform_generator_kwargs(self, waveform_generator_kwargs):
        if isinstance(waveform_generator_kwargs, str):
            waveform_generator_kwargs = convert_string_to_dict(
                waveform_generator_kwargs
            )
        elif waveform_generator_kwargs is None:
            waveform_generator_kwargs = dict()
        elif not isinstance(waveform_generator_kwargs, dict):
            raise TypeError(
                f"Type {type(waveform_generator_kwargs)} not understood for joint waveform generator kwargs."
            )
        self._joint_injection_waveform_generator_kwargs = waveform_generator_kwargs

    @property
    def joint_injection_waveform_generator_model_kwargs(self):
        return self._joint_injection_waveform_generator_model_kwargs

    @joint_injection_waveform_generator_model_kwargs.setter
    def joint_injection_waveform_generator_model_kwargs(self, waveform_model_kwargs):
        if isinstance(waveform_model_kwargs, str):
            waveform_model_kwargs = convert_string_to_dict(waveform_model_kwargs)
        elif waveform_model_kwargs is None:
            waveform_model_kwargs = dict()
        elif not isinstance(waveform_model_kwargs, dict):
            raise TypeError(
                f"Type {type(waveform_model_kwargs)} not understood for joint waveform generator kwargs."
            )
        self._joint_injection_waveform_generator_model_kwargs = waveform_model_kwargs
