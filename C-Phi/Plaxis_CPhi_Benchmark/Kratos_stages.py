import ctypes
import os
import subprocess
from pathlib import Path

import KratosMultiphysics as Kratos

from KratosMultiphysics.GeoMechanicsApplication.geomechanics_analysis import GeoMechanicsAnalysis


def EnsureUdsmShowExtraStub() -> str:
    """Build/load a small shim exporting show_extra_info_ for example64.so."""
    stub_source = Path("/tmp/show_extra_stub.c")
    stub_library = Path("/tmp/libshow_extra_stub.so")

    if not stub_library.exists():
        stub_source.write_text("void show_extra_info_(void) {}\n", encoding="ascii")
        subprocess.run(
            ["gcc", "-shared", "-fPIC", "-o", str(stub_library), str(stub_source)],
            check=True,
            capture_output=True,
            text=True,
        )

    ctypes.CDLL(str(stub_library), mode=ctypes.RTLD_GLOBAL)
    return str(stub_library)

if __name__ == "__main__":

    currentWorking = os.getcwd()
    shim_path = EnsureUdsmShowExtraStub()
    print(f"[UDSM shim] loaded: {shim_path}")

    # construct parameterfile names of stages to run
    project_path = r"."
    n_stages = 2
    parameter_file_names = [os.path.join(project_path, 'ProjectParameters_stage' + str(i + 1) + '.json') for i in
                            range(n_stages)]

    # change to project directory
    os.chdir(project_path)

    # setup stages from parameterfiles
    parameters_stages = [None] * n_stages
    for idx, parameter_file_name in enumerate(parameter_file_names):
        with open(parameter_file_name, 'r') as parameter_file:
            parameters_stages[idx] = Kratos.Parameters(parameter_file.read())

    model = Kratos.Model()
    stages = [GeoMechanicsAnalysis(model, stage_parameters) for stage_parameters in parameters_stages]

    # execute the stages
    [stage.Run() for stage in stages]

    # back to working directory
    os.chdir(currentWorking)
