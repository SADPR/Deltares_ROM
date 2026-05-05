import json
import os
from pathlib import Path
import KratosMultiphysics
import numpy as np
from matplotlib import pyplot as plt
from custom_rom_manager import RomManager

# =============================================================================
# Configuration & Flags
# =============================================================================
FIGURES_DIR = Path("figures")
ROM_ERROR_INDICATOR_TOL = 1e-5
SVD_TRUNCATION_TOLERANCE = 0.0

# Stage switches
RUN_STAGE0 = True   # Plot Sampling
RUN_STAGE1 = True   # FOM Training
RUN_STAGE2 = True   # POD Basis
RUN_STAGE3 = True   # ROM Verification
RUN_STAGE4 = True   # ROM Test
RUN_STAGE5 = True   # Postprocess Plots

# Recompute controls
STAGE1_FORCE = False
STAGE3_FORCE_FOM = False
STAGE3_FORCE_ROM = False
STAGE4_FORCE_FOM = False
STAGE4_FORCE_ROM = False

# Sampling setup aligned with the JSON and SQL variants
N_TRAIN = 15
N_TEST = 6
TRAIN_SEEDS = (42, 72)
TEST_SEEDS = (44, 74)

# Head sweep used by each case: [H_min, H_max, dH]
# This defines the "Time Steps" for each simulation case
MU_HEAD_RANGE = [3.0, 10.0, 0.1]

# =============================================================================
# Custom Hooks
# =============================================================================

def CustomizeSimulation(cls, global_model, parameters, type_of_simulation="FOM"):
    class CustomSimulation(cls):
        def __init__(self, model, project_parameters, type_of_simulation=type_of_simulation):
            super().__init__(model, project_parameters)
            self.type_of_simulation = type_of_simulation
            self.ErroIndicator = True
            self.ResidualNorm = 0.0

        def IsErroIndicatorAcceptable(self):
            return self.ErroIndicator

        def GetResidualNorm(self):
            return self.ResidualNorm

        def Finalize(self):
            super().Finalize()
            self.r = np.array(self._GetSolver().builder_and_solver.GetCurrentResidual())
            self.ResidualNorm = np.linalg.norm(self.r)
            if self.type_of_simulation == "ROM" and self.ResidualNorm > ROM_ERROR_INDICATOR_TOL:
                self.ErroIndicator = False

    return CustomSimulation(global_model, parameters, type_of_simulation)


def UpdateMaterialParametersFile(materials_file_name, mu=None):
    if mu is None: return
    with open(materials_file_name, "r") as f:
        data = json.load(f)
    
    # Map mu[0] -> Permeability, mu[1] -> d70
    data["properties"][0]["Material"]["Variables"]["PERMEABILITY_XX"] = mu[0]
    data["properties"][1]["Material"]["Variables"]["PERMEABILITY_XX"] = mu[0]
    data["properties"][3]["Material"]["Variables"]["PIPE_D_70"] = mu[1]

    with open(materials_file_name, "w") as f:
        json.dump(data, f, indent=4)

def GetRomManagerParameters():
    return KratosMultiphysics.Parameters(f"""{{
        "rom_stages_to_train" : ["ROM"],
        "rom_stages_to_test" : ["ROM"],
        "paralellism" : null,
        "projection_strategy": "galerkin",
        "save_gid_output": false,
        "save_vtk_output": true,
        "output_name": "id",
        "assembling_strategy": "global",
        "rom_error_indicator_tolerance": {ROM_ERROR_INDICATOR_TOL},
        "ROM":{{
            "svd_truncation_tolerance": {SVD_TRUNCATION_TOLERANCE},
            "model_part_name": "PorousDomain",
            "nodal_unknowns": ["WATER_PRESSURE"]
        }}
    }}""")

# =============================================================================
# Helper Logic
# =============================================================================

def get_percentual_min_max(value, percentage):
    return value - value * (percentage / 100.0), value + value * (percentage / 100.0)


def get_multiple_params(n, s1, s2):
    permeability_min, permeability_max = get_percentual_min_max(5e-12, 80)
    np.random.seed(s1)
    k = np.random.uniform(permeability_min, permeability_max, size=n)

    d70_min, d70_max = get_percentual_min_max(1e-4, 50)
    np.random.seed(s2)
    d = np.random.uniform(d70_min, d70_max, size=n)
    return np.vstack((k, d)).T.tolist()

def plot_mu_values(mu_train, mu_test):
    mu_train_a = np.zeros(len(mu_train))
    mu_train_m = np.zeros(len(mu_train))
    mu_test_a = np.zeros(len(mu_test))
    mu_test_m = np.zeros(len(mu_test))

    for i in range(len(mu_train)):
        mu_train_a[i] = mu_train[i][0]
        mu_train_m[i] = mu_train[i][1]

    for i in range(len(mu_test)):
        mu_test_a[i] = mu_test[i][0]
        mu_test_m[i] = mu_test[i][1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(mu_train_m, mu_train_a, "bs", label="Train Values")
    ax.plot(mu_test_m, mu_test_a, "ro", label="Test Values")
    ax.set_title("Mu Values")
    ax.set_ylabel(r"$d_{70}$")
    ax.set_xlabel(r"Permeability XX")
    ax.grid(True)
    ax.legend(bbox_to_anchor=(0.85, 1.03, 1.0, 0.102), loc="upper left", borderaxespad=0.0)
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / "sampling.png", dpi=200)
    plt.close(fig)

def plot_results(fom, rom, label, qoi_name):
    plt.figure(figsize=(10, 5))
    x = np.arange(len(fom))
    plt.bar(x, fom, 0.35, label="FOM", alpha=0.8)
    plt.bar(x + 0.35, rom, 0.35, label="ROM", alpha=0.8)
    plt.title(f"{label}: {qoi_name}")
    plt.ylabel(qoi_name)
    plt.legend()
    plt.savefig(FIGURES_DIR / f"{label.lower().replace(' ', '_')}_{qoi_name.lower().replace(' ', '_')}.png")
    plt.close()

# =============================================================================
# Main Stage Loop
# =============================================================================

if __name__ == "__main__":
    rom_manager = RomManager(
        project_parameters_name="ProjectParameters.json",
        general_rom_manager_parameters=GetRomManagerParameters(),
        UpdateMaterialParametersFile=UpdateMaterialParametersFile,
        CustomizeSimulation=CustomizeSimulation,
        mu_names=["permeability_xx", "d70"],
    )

    mu_train = get_multiple_params(N_TRAIN, TRAIN_SEEDS[0], TRAIN_SEEDS[1])
    mu_test = get_multiple_params(N_TEST, TEST_SEEDS[0], TEST_SEEDS[1])

    if RUN_STAGE0: plot_mu_values(mu_train, mu_test)

    if RUN_STAGE1:
        rom_manager.stage1_fom_training(mu_train, MU_HEAD_RANGE, force_recompute=STAGE1_FORCE)

    if RUN_STAGE2:
        rom_manager.stage2_build_pod_basis(mu_train, load_basis_if_available=True)

    if RUN_STAGE3:
        rom_manager.stage3_rom_verification(mu_train, MU_HEAD_RANGE, STAGE3_FORCE_FOM, STAGE3_FORCE_ROM)

    if RUN_STAGE4:
        rom_manager.stage4_rom_test(mu_test, MU_HEAD_RANGE, STAGE4_FORCE_FOM, STAGE4_FORCE_ROM)
        rom_manager.plot_parameter_space(mu_train, mu_test)

    if RUN_STAGE5:
        # Verification Plots
        qoi_fom = rom_manager.load_qoi_vector(mu_train, "train", "FOM", "critical_head")
        qoi_rom = rom_manager.load_qoi_vector(mu_train, "train", "ROM", "critical_head")
        plot_results(qoi_fom, qoi_rom, "Verification", "Critical Head")

        # Test Plots
        qoi_fom_t = rom_manager.load_qoi_vector(mu_test, "test", "FOM", "critical_head")
        qoi_rom_t = rom_manager.load_qoi_vector(mu_test, "test", "ROM", "critical_head")
        plot_results(qoi_fom_t, qoi_rom_t, "Test", "Critical Head")

    rom_manager.PrintErrors()
