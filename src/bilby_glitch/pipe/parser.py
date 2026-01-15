import bilby_pipe

from bilby_pipe.utils import nonestr
from bilby_pipe.parser import StoreBoolean


def glitch_create_parser(top_level=True):
    """Creates the BilbyArgParser for bilby_pipe - with glitch modifications

    Parameters
    ----------
    top_level:
        If true, parser is to be used at the top-level with requirement
        checking etc, else it is an internal call and will be ignored.

    Returns
    -------
    parser: BilbyArgParser instance
        Argument parser

    """

    parser = bilby_pipe.parser.create_parser(top_level=top_level)
    injection_parser = [
        x for x in parser._action_groups if x.title == "Injection arguments"
    ][0]
    injection_parser.add(
        "--injection-is-joint",
        action=StoreBoolean,
        default=False,
        help="If true, then this injection is assumed to be include joint injection objects"
        "If this is true, then the standard waveform-generator and prior parser options"
        "will point to the CBC, while"
        "joint-injection-waveform-generator-dict, joint-injection-waveform-generator-dict-generator-args,"
        "joint-injection-waveform-generator-model-args, joint-injection-glitch-priors, etc. will point"
        "to the glitch elements, indexed by their index name",
    )
    injection_parser.add(
        "--joint-injection-files",
        type=nonestr,
        default=None,
        help="If likelihood is joint, this provides paths to injection files"
        "for each of the joint waveforms, indexed by glitch index name",
    )
    injection_parser.add(
        "--joint-injection-waveform-generators",
        default=None,
        type=nonestr,
        help="If a joint injection is being used,"
        "these are the waveform generators for the joint (glitch) objects,"
        "indexed by glitch index_name",
    )
    injection_parser.add(
        "--joint-injection-waveform-generator-source-models",
        default=None,
        type=nonestr,
        help="If a joint injection is being used"
        "these are the source models for the joint (glitch) objects"
        "indexed by glitch index_name",
    )
    injection_parser.add(
        "--joint-injection-waveform-generator-parametric-models",
        default=None,
        type=nonestr,
        help="If a joint likelihood is being used"
        "these are the parametric models to convert for joint use"
        "indexed by glitch index_name",
    )
    injection_parser.add(
        "--joint-injection-waveform-generator-args",
        default=None,
        type=nonestr,
        help="Generator args for wf generators used in a joint injection,"
        "indexed by glitch index name",
    )
    injection_parser.add(
        "--joint-injection-waveform-generator-model-args",
        default=None,
        type=nonestr,
        help="Model args for wf generators used in a joint injection,"
        "indexed by glitch index name",
    )
    injection_parser.add(
        "--joint-injection-frequency-domain-source-model",
        default=None,
        type=nonestr,
        help=(
            "Name of the frequency domain source model. Can be one of"
            "[lal_binary_black_hole, lal_binary_neutron_star,"
            "lal_eccentric_binary_black_hole_no_spins, sinegaussian, "
            "supernova, supernova_pca_model] or any python  path to a bilby "
            " source function the users installation, e.g. examp.source.bbh."
            "When passed as None (default) falls back to standard"
            "frequency-domain-source-model argument"
        ),
    )
    likelihood_parser = [
        x for x in parser._action_groups if x.title == "Likelihood arguments"
    ][0]
    likelihood_parser.add(
        "--likelihood-is-joint",
        default=False,
        action=StoreBoolean,
        help="If true, then this likelihood is assumed to be a joint likelihood."
        "Currently this scenario supports glitch+glitch(+glitch...)"
        "and cbc+glitch(+glitch...), cbc+cbc is feasible but not implemented."
        "If this is true, then the standard waveform-generator and prior parser options"
        "will point to the CBC, while"
        "joint-waveform-generator-dict, joint-waveform-generator-dict-generator-args,"
        "joint-waveform-generator-model-args, joint-glitch-priors, etc. will point"
        "to the glitch elements, indexed by their index name",
    )
    prior_parser = [x for x in parser._action_groups if x.title == "Prior arguments"][0]
    prior_parser.add(
        "--joint-prior-files",
        type=nonestr,
        default=None,
        help="If joint likelihood is being used, put joint (glitch) prior links here."
        "These should each be prior files, indexed by the glitch index name",
    )
    prior_parser.add(
        "--joint-prior-defaults",
        type=nonestr,
        default=None,
        help="If joint likelihood is being used, for each joint prior a default prior"
        "is to be used, indexed by glitch index name"
        "This is probably PriorDict for each glitch",
    )
    waveform_parser = [
        x for x in parser._action_groups if x.title == "Waveform arguments"
    ][0]
    waveform_parser.add(
        "--extra-waveform-generator-kwargs",
        default=None,
        type=nonestr,
        help="A dictionary of extra arguments to pass to the waveform generator call",
    )
    waveform_parser.add(
        "--joint-waveform-generators",
        default=None,
        type=nonestr,
        help="If a joint likelihood is being used,"
        "these are the waveform generators for the joint (glitch) objects,"
        "indexed by glitch index_name",
    )
    waveform_parser.add(
        "--joint-waveform-generator-source-models",
        default=None,
        type=nonestr,
        help="If a joint likelihood is being used"
        "these are the source models for the joint (glitch) objects"
        "indexed by glitch index_name",
    )
    waveform_parser.add(
        "--joint-waveform-generator-parametric-models",
        default=None,
        type=nonestr,
        help="If a joint likelihood is being used"
        "these are the parametric models to convert for joint use"
        "indexed by glitch index_name",
    )
    waveform_parser.add(
        "--joint-waveform-generator-parametric-sampling-parameters",
        default=None,
        type=nonestr,
        help="If a joint likelihood is being used"
        "these are the parameter names for the parametric model to use"
        "indexed by glitch index_name",
    )
    waveform_parser.add(
        "--joint-waveform-generator-args",
        default=None,
        type=nonestr,
        help="Generator args for wf generators used in a joint likelihood,"
        "indexed by glitch index name",
    )
    waveform_parser.add(
        "--joint-waveform-generator-model-args",
        default=None,
        type=nonestr,
        help="Model args for wf generators used in a joint likelihood,"
        "indexed by glitch index name",
    )
    return parser
