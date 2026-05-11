import ctypes
import os
import subprocess
from pathlib import Path

import KratosMultiphysics as Kratos

from KratosMultiphysics.GeoMechanicsApplication.geomechanics_analysis import GeoMechanicsAnalysis


def EnsureUdsmShowExtraStub() -> str:
    """Build/load a shim exporting show_extra_info_ for example64.so in repo-local runtime."""
    runtime_dir = Path(__file__).resolve().parent / ".runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    stub_source = runtime_dir / "show_extra_stub.c"
    stub_library = runtime_dir / "libshow_extra_stub.so"
    stub_code = "void show_extra_info_(void) {}\n"

    if not stub_source.exists() or stub_source.read_text(encoding="ascii") != stub_code:
        stub_source.write_text(stub_code, encoding="ascii")

    needs_rebuild = (
        not stub_library.exists()
        or stub_library.stat().st_mtime < stub_source.stat().st_mtime
    )
    if needs_rebuild:
        try:
            subprocess.run(
                ["gcc", "-shared", "-fPIC", "-o", str(stub_library), str(stub_source)],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Could not build UDSM preload stub because 'gcc' is not available. "
                "Install gcc or provide a prebuilt '.runtime/libshow_extra_stub.so'."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Failed to compile '.runtime/libshow_extra_stub.so'. "
                f"gcc stderr:\n{exc.stderr}"
            ) from exc

    ctypes.CDLL(str(stub_library), mode=ctypes.RTLD_GLOBAL)
    return str(stub_library)

def ConfigureLdPreload(stub_library_path: str) -> None:
    existing_preload = os.environ.get("LD_PRELOAD", "").strip()
    if existing_preload:
        preloads = existing_preload.split()
        if stub_library_path not in preloads:
            os.environ["LD_PRELOAD"] = f"{stub_library_path} {existing_preload}"
    else:
        os.environ["LD_PRELOAD"] = stub_library_path

if __name__ == "__main__":

    currentWorking = os.getcwd()
    shim_path = EnsureUdsmShowExtraStub()
    ConfigureLdPreload(shim_path)
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
