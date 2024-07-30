import os
import copy

import bilby
from bilby_pipe.utils import DataDump
from ptfp.bilby_pipe_interface import get_analysis_from_dag

from .pipe_data_analysis import GlitchDataAnalysisInput
from .result import JointCBCGlitchResult
from .joint_likelihood import (
    GravitationalWaveTransient,
    JointGlitchGravitationalWaveTransient,
)

from typing import Union, List, Tuple


def get_all_analysis_components(
    base_directory: str,
    case: int = 0,
    analysis_ifos: str = "H1L1",
    frequency_cut_ifo: str = "L1",
    frequencies: Union[None, List[Tuple]] = None,
    joint=True,
) -> Tuple[
    bilby.core.result.Result,
    DataDump,
    GlitchDataAnalysisInput,
    bilby.gw.likelihood.GravitationalWaveTransient,
    List[bilby.gw.likelihood.GravitationalWaveTransient],
]:
    """Get all the junk we need from a run

    Parameters
    ==========
    base_directory : str
        The analysis directory
    case : int
        If multiple data gens, the data index
    analysis_ifos : str
        For coherence tests, the result to use (i.e. from which combo of ifos)
    frequency_cut_ifo : str
        The ifo to apply the frequency cut in
    frequencies : Union[None, List[Tuple]]
        If None, just returns original likelihood, if provided also returns likelihoods with different min/max frequencies
        First element min, second element max, if either is None the original value is used
    joint : bool
        Whether to use the joint likelihood class

    Returns
    =======
    bilby.core.result.Result
        The analysis result
    DataDump
        The data dump object
    GlitchDataAnalysisInput
        The reconsructed analysis object
    bilby.gw.likelihood.GravitationalWaveTransient
        The likelihood used in the analysis
    List[bilby.gw.likelihood.GravitationalWaveTransient]
        A list of likelihoods as the base, but with different frequency cuts
    """
    merge_result_file = [
        os.path.join(base_directory, "final_result", x)
        for x in os.listdir(os.path.join(base_directory, "final_result"))
        if f"data{case}_" in x and f"_{analysis_ifos}_" in x and "result.hdf5" in x
    ][0]
    result = bilby.core.result.read_in_result(merge_result_file)
    result = JointCBCGlitchResult(result)

    data_file = [
        os.path.join(base_directory, "data", x)
        for x in os.listdir(os.path.join(base_directory, "data"))
        if f"data{case}_" in x and "data_dump.pickle" in x
    ][0]
    data = DataDump.from_pickle(data_file)

    owd = os.getcwd()
    os.chdir(os.path.join(base_directory, ".."))
    submit_dag = [
        os.path.join(base_directory, "submit", x)
        for x in os.listdir(os.path.join(base_directory, "submit"))
        if "dag" in x and x.split(".")[-1] == "submit"
    ][0]
    analysis = get_analysis_from_dag(submit_dag, case=case)
    os.chdir(owd)

    likelihood_args = [
        analysis.interferometers,
        analysis.waveform_generator,
    ]
    likelihood_kwargs = dict(
        priors=copy.deepcopy(result.priors), distance_marginalization=False
    )
    if joint:
        likelihood_class = JointGlitchGravitationalWaveTransient
        likelihood_args.append(analysis.joint_waveform_generators)
    else:
        likelihood_class = GravitationalWaveTransient

    base_likelihood = likelihood_class(*likelihood_args, **likelihood_kwargs)

    if frequencies is None:
        return result, data, analysis, base_likelihood, []
