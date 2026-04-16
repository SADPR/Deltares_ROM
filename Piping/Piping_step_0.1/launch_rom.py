import json
from pathlib import Path

import KratosMultiphysics
import numpy as np
from matplotlib import pyplot as plt

from custom_rom_manager import RomManager

FIGURES_DIR = Path("figures")
ROM_ERROR_INDICATOR_TOL = 1e-5


# =============================================================================
# RomManager hooks
# =============================================================================

def CustomizeSimulation(cls, global_model, parameters, type_of_simulation="FOM"):
    class CustomSimulation(cls):
        def __init__(self, model, project_parameters, type_of_simulation=type_of_simulation):
            super().__init__(model, project_parameters)
            self.type_of_simulation = type_of_simulation
            self.ErroIndicator = True
            self.residual_norms = []

        def ModifyInitialGeometry(self):
            super().ModifyInitialGeometry()

        def InitializeSolutionStep(self):
            super().InitializeSolutionStep()

        def CustomMethod(self):
            return self.custom_param

        def IsErroIndicatorAcceptable(self):
            return self.ErroIndicator

        def GetResidualNorm(self):
            return self.ResidualNorm

        def Finalize(self):
            super().Finalize()

            self.r = np.array(self._GetSolver().builder_and_solver.GetCurrentResidual())
            self.ResidualNorm = np.linalg.norm(self.r)
            self.residual_norms.append(self.ResidualNorm)

            if self.type_of_simulation == "ROM" and self.ResidualNorm > ROM_ERROR_INDICATOR_TOL:
                # TODO: expose this threshold as a configurable parameter
                self.ErroIndicator = False

    return CustomSimulation(global_model, parameters, type_of_simulation)


def UpdateMaterialParametersFile(materials_file_name, mu=None):
    """Update permeability and d70 for each parameter sample."""
    permeability_xx = mu[0]
    d70 = mu[1]

    with open(materials_file_name, "r", encoding="utf-8") as parameter_file:
        parameters = json.load(parameter_file)
        parameters["properties"][0]["Material"]["Variables"]["PERMEABILITY_XX"] = permeability_xx
        parameters["properties"][1]["Material"]["Variables"]["PERMEABILITY_XX"] = permeability_xx
        parameters["properties"][3]["Material"]["Variables"]["PIPE_D_70"] = d70

    with open(materials_file_name, "w", encoding="utf-8") as parameter_file:
        json.dump(parameters, parameter_file, indent=4)


# =============================================================================
# ROM configuration
# =============================================================================

def GetRomManagerParameters():
    """Define ROM workflow settings used by the staged RomManager."""
    return KratosMultiphysics.Parameters(
        """{
            "rom_stages_to_train" : ["ROM"],
            "rom_stages_to_test" : ["ROM"],
            "paralellism" : null,
            "projection_strategy": "galerkin",
            "save_gid_output": false,
            "save_vtk_output": true,
            "output_name": "id",
            "assembling_strategy": "global",
            "rom_error_indicator_tolerance": %e,
            "ROM":{
                "svd_truncation_tolerance": 0,
                "model_part_name": "PorousDomain",
                "nodal_unknowns": ["WATER_PRESSURE"]
            }
        }""" % ROM_ERROR_INDICATOR_TOL
    )


# =============================================================================
# Plot helpers
# =============================================================================


def _figure_path(filename):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR / filename

def _relative_l2_error_percent(reference, approximation):
    reference = np.asarray(reference, dtype=float)
    approximation = np.asarray(approximation, dtype=float)
    reference_norm = np.linalg.norm(reference)
    if reference_norm == 0.0:
        return np.nan
    return 100.0 * np.linalg.norm(reference - approximation) / reference_norm


def plot_pipe_length_plus_residual(
    fom,
    rom,
    residual_fom,
    residual_rom,
    stage_label,
    qoi_label,
    output_png_name,
):
    fom = np.squeeze(np.asarray(fom, dtype=float))
    rom = np.squeeze(np.asarray(rom, dtype=float))
    residual_fom = np.squeeze(np.asarray(residual_fom, dtype=float))
    residual_rom = np.squeeze(np.asarray(residual_rom, dtype=float))

    num_cases = fom.size
    case_indices = np.arange(num_cases)
    bar_width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(case_indices, fom.tolist(), bar_width, label="FOM", alpha=0.85)
    ax1.bar(case_indices + bar_width, rom.tolist(), bar_width, label="ROM", alpha=0.85)
    ax1.set_ylabel(qoi_label)
    ax1.set_xlabel("Case index")
    ax1.set_title(f"{stage_label} | {qoi_label}: FOM vs ROM + residual norms")
    ax1.grid(True, axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(case_indices, residual_fom, "bo-", label="Residual Norm FOM")
    ax2.plot(case_indices, residual_rom, "ro-", label="Residual Norm ROM")
    ax2.set_ylabel("Residual norm")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    error_percent = _relative_l2_error_percent(fom, rom)
    print(
        f"[{stage_label}] Relative L2 error for '{qoi_label}' (ROM vs FOM): "
        f"{error_percent:.6f}%"
    )

    fig.tight_layout()
    fig.savefig(_figure_path(output_png_name), dpi=200)
    plt.show()
    plt.close(fig)


def plot_pipe_length_roms(
    fom,
    rom,
    stage_label,
    qoi_label,
    output_png_name,
    name_1="FOM",
    name_2="ROM",
):
    fom = np.squeeze(np.asarray(fom, dtype=float))
    rom = np.squeeze(np.asarray(rom, dtype=float))

    num_cases = fom.size
    case_indices = np.arange(num_cases)
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(case_indices, fom.tolist(), bar_width, label=name_1, alpha=0.85)
    ax.bar(case_indices + bar_width, rom.tolist(), bar_width, label=name_2, alpha=0.85)
    ax.set_ylabel(qoi_label)
    ax.set_xlabel("Case index")
    ax.set_title(f"{stage_label} | {qoi_label}: {name_1} vs {name_2}")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    error_percent = _relative_l2_error_percent(fom, rom)
    print(
        f"[{stage_label}] Relative L2 error for '{qoi_label}' ({name_2} vs {name_1}): "
        f"{error_percent:.6f}%"
    )

    fig.tight_layout()
    fig.savefig(_figure_path(output_png_name), dpi=200)
    plt.show()
    plt.close(fig)


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
    fig.savefig(_figure_path("sampling.png"), dpi=200)
    plt.show()
    plt.close(fig)


# =============================================================================
# Sampling helpers
# =============================================================================

def get_percentual_min_max(val, percentage):
    return val - val * (percentage / 100), val + val * (percentage / 100)


def get_multiple_params(number_of_total_samples, seed1, seed2):
    permeability_min, permeability_max = get_percentual_min_max(5e-12, 80)
    np.random.seed(seed1)
    row1 = np.random.uniform(permeability_min, permeability_max, size=number_of_total_samples)

    d70_min, d70_max = get_percentual_min_max(0.0001, 50)
    np.random.seed(seed2)
    row2 = np.random.uniform(d70_min, d70_max, size=number_of_total_samples)

    mu = np.vstack((row1, row2))
    return mu.T.tolist()


# =============================================================================
# Main stages
# =============================================================================

if __name__ == "__main__":
    # Stage switches
    RUN_STAGE0_PLOT_SAMPLING = True
    RUN_STAGE1_FOM_TRAIN = True
    RUN_STAGE2_POD_BASIS = True
    RUN_STAGE3_ROM_VERIFICATION = True
    RUN_STAGE4_ROM_TEST = True
    RUN_STAGE5_POSTPROCESS_VERIFICATION = True
    RUN_STAGE6_POSTPROCESS_TEST = True

    # Stage controls
    STAGE1_FORCE_RECOMPUTE = False
    STAGE2_LOAD_BASIS_IF_AVAILABLE = True
    STAGE3_FORCE_RECOMPUTE_FOM = False
    STAGE3_FORCE_RECOMPUTE_ROM = False
    STAGE4_FORCE_RECOMPUTE_FOM = False
    STAGE4_FORCE_RECOMPUTE_ROM = False

    # Sampling setup
    N_TRAIN = 15
    N_TEST = 6
    TRAIN_SEEDS = (42, 72)
    TEST_SEEDS = (44, 74)

    # Head sweep used by each case: [H_min, H_max, dH]
    MU_HEAD_RANGE = [3.0, 10.0, 0.1]

    rom_manager = RomManager(
        project_parameters_name="ProjectParameters.json",
        general_rom_manager_parameters=GetRomManagerParameters(),
        UpdateMaterialParametersFile=UpdateMaterialParametersFile,
        CustomizeSimulation=CustomizeSimulation,
        mu_names=["permeability_xx", "d70"],
    )

    mu_train = get_multiple_params(N_TRAIN, TRAIN_SEEDS[0], TRAIN_SEEDS[1])
    mu_test = get_multiple_params(N_TEST, TEST_SEEDS[0], TEST_SEEDS[1])

    # Stage 0: visualize sampled train/test parameters
    if RUN_STAGE0_PLOT_SAMPLING:
        plot_mu_values(mu_train, mu_test)

    # Stage 1: FOM training snapshots
    if RUN_STAGE1_FOM_TRAIN:
        rom_manager.stage1_fom_training(
            mu_train=mu_train,
            min_max_step=MU_HEAD_RANGE,
            force_recompute=STAGE1_FORCE_RECOMPUTE,
        )

    # Stage 2: POD basis from training FOM snapshots
    if RUN_STAGE2_POD_BASIS:
        rom_manager.stage2_build_pod_basis(
            mu_train=mu_train,
            load_basis_if_available=STAGE2_LOAD_BASIS_IF_AVAILABLE,
        )

    # Stage 3: ROM verification (ROM vs FOM on training set)
    if RUN_STAGE3_ROM_VERIFICATION:
        rom_manager.stage3_rom_verification(
            mu_train=mu_train,
            basis_mu_train=mu_train,
            min_max_step=MU_HEAD_RANGE,
            force_recompute_fom=STAGE3_FORCE_RECOMPUTE_FOM,
            force_recompute_rom=STAGE3_FORCE_RECOMPUTE_ROM,
        )

    # Stage 4: ROM test on unseen parameters
    if RUN_STAGE4_ROM_TEST:
        rom_manager.stage4_rom_test(
            mu_test=mu_test,
            basis_mu_train=mu_train,
            min_max_step=MU_HEAD_RANGE,
            force_recompute_fom=STAGE4_FORCE_RECOMPUTE_FOM,
            force_recompute_rom=STAGE4_FORCE_RECOMPUTE_ROM,
        )

    # Stage 5: postprocess verification set (train)
    if RUN_STAGE5_POSTPROCESS_VERIFICATION:
        qoi_fom_pipe = rom_manager.load_qoi_vector(
            mu_train,
            case_tag="train",
            simulation_type="FOM",
            qoi_name="pipe_length",
        )
        qoi_rom_pipe = rom_manager.load_qoi_vector(
            mu_train,
            case_tag="train",
            simulation_type="ROM",
            qoi_name="pipe_length",
        )
        plot_pipe_length_roms(
            qoi_fom_pipe,
            qoi_rom_pipe,
            stage_label="Verification (train set)",
            qoi_label="Pipe Length",
            output_png_name="verification_pipe_length.png",
        )

        qoi_fom_head = rom_manager.load_qoi_vector(
            mu_train,
            case_tag="train",
            simulation_type="FOM",
            qoi_name="critical_head",
        )
        qoi_rom_head = rom_manager.load_qoi_vector(
            mu_train,
            case_tag="train",
            simulation_type="ROM",
            qoi_name="critical_head",
        )
        plot_pipe_length_roms(
            qoi_fom_head,
            qoi_rom_head,
            stage_label="Verification (train set)",
            qoi_label="Critical Head",
            output_png_name="verification_critical_head.png",
        )

        qoi_fom_res = rom_manager.load_qoi_vector(
            mu_train,
            case_tag="train",
            simulation_type="FOM",
            qoi_name="residual_norm",
        )
        qoi_rom_res = rom_manager.load_qoi_vector(
            mu_train,
            case_tag="train",
            simulation_type="ROM",
            qoi_name="residual_norm",
        )
        plot_pipe_length_plus_residual(
            qoi_fom_head,
            qoi_rom_head,
            qoi_fom_res,
            qoi_rom_res,
            stage_label="Verification (train set)",
            qoi_label="Critical Head",
            output_png_name="verification_critical_head_with_residual.png",
        )

    # Stage 6: postprocess test set
    if RUN_STAGE6_POSTPROCESS_TEST:
        qoi_fom_head = rom_manager.load_qoi_vector(
            mu_test,
            case_tag="test",
            simulation_type="FOM",
            qoi_name="critical_head",
        )
        qoi_rom_head = rom_manager.load_qoi_vector(
            mu_test,
            case_tag="test",
            simulation_type="ROM",
            qoi_name="critical_head",
        )

        qoi_fom_res = rom_manager.load_qoi_vector(
            mu_test,
            case_tag="test",
            simulation_type="FOM",
            qoi_name="residual_norm",
        )
        qoi_rom_res = rom_manager.load_qoi_vector(
            mu_test,
            case_tag="test",
            simulation_type="ROM",
            qoi_name="residual_norm",
        )
        plot_pipe_length_plus_residual(
            qoi_fom_head,
            qoi_rom_head,
            qoi_fom_res,
            qoi_rom_res,
            stage_label="Test (unseen set)",
            qoi_label="Critical Head",
            output_png_name="test_critical_head_with_residual.png",
        )

    rom_manager.PrintErrors()
