import bilby_pipe


class GlitchAnalysisNode(bilby_pipe.job_creation.nodes.analysis_node.AnalysisNode):
    @property
    def executable(self):
        if self.inputs.use_mpi:
            return self._get_executable_path("mpiexec")
        elif self.inputs.analysis_executable:
            return self._get_executable_path(self.inputs.analysis_executable)
        else:
            return self._get_executable_path("glitch_bilby_pipe_analysis")


class GlitchGenerationNode(
    bilby_pipe.job_creation.nodes.generation_node.GenerationNode
):
    @property
    def executable(self):
        return self._get_executable_path("glitch_bilby_pipe_generation")


bilby_pipe.job_creation.nodes.analysis_node.AnalysisNode = GlitchAnalysisNode
bilby_pipe.job_creation.nodes.generation_node.GenerationNode = GlitchGenerationNode
