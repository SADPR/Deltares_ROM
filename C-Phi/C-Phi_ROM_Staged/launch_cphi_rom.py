import ctypes
import json
import os
import subprocess
from pathlib import Path
import KratosMultiphysics
import numpy as np
from matplotlib import pyplot as plt
from cphi_rom_manager import RomManager


def EnsureUdsmShowExtraStub():
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

def ConfigureLdPreload(stub_library_path):
    existing_preload = os.environ.get("LD_PRELOAD", "").strip()
    if existing_preload:
        preloads = existing_preload.split()
        if stub_library_path not in preloads:
            os.environ["LD_PRELOAD"] = f"{stub_library_path} {existing_preload}"
    else:
        os.environ["LD_PRELOAD"] = stub_library_path


def CustomizeSimulation(cls, global_model, parameters, type_of_simulation="FOM"):
    class CustomSimulation(cls):
        def __init__(self, model, project_parameters, type_of_simulation=type_of_simulation):
            super().__init__(model, project_parameters)
            self.type_of_simulation = type_of_simulation

    return CustomSimulation(global_model, parameters, type_of_simulation)

def UpdateMaterialParametersFile(materials_file_name, mu=None):
    if mu is None:
        return
    c0 = mu[0]
    phi0 = mu[1]
    with open(materials_file_name, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    try:
        data["properties"][0]["Material"]["Variables"]["UMAT_PARAMETERS"][2] = c0
        data["properties"][0]["Material"]["Variables"]["UMAT_PARAMETERS"][3] = phi0
    except KeyError:
        pass

    with open(materials_file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def UpdateProjectParameters(parameters, mu=None, sim_type=None):
    # Keep FOM/ROM solver settings identical (read directly from ProjectParameters_stage2.json).
    return parameters

def get_grid_params(c_points=3, phi_points=3):
    # Cohesion from 10000 to 15000 Pa (guarantees Stage 1 baseline stability)
    c0_vals = np.linspace(10000, 15000, c_points)
    # Friction Angle from 30 to 45 degrees
    phi0_vals = np.linspace(30, 45, phi_points)
    
    C, P = np.meshgrid(c0_vals, phi0_vals)
    mu = np.vstack((C.ravel(), P.ravel()))
    return mu.T.tolist()

def plot_mu_values(mu_train, mu_test, filename="figures/sampling_cphi.png", title="Training Parameter Space (C-Phi)"):
    mu_train_a = [m[0] for m in mu_train]
    mu_train_m = [m[1] for m in mu_train]
    mu_test_a = [m[0] for m in mu_test]
    mu_test_m = [m[1] for m in mu_test]

    fig, ax = plt.subplots(figsize=(8, 6))
    if mu_train:
        ax.plot(mu_train_m, mu_train_a, "bs", label="Train Values")
    if mu_test:
        ax.plot(mu_test_m, mu_test_a, "ro", label="Test Values")
    ax.set_title(title)
    ax.set_ylabel(r"Initial Cohesion ($c_0$) [Pa]")
    ax.set_xlabel(r"Initial Friction Angle ($\phi_0$) [deg]")
    ax.grid(True)
    ax.legend(bbox_to_anchor=(0.85, 1.03, 1.0, 0.102), loc="upper left", borderaxespad=0.0)
    fig.tight_layout()
    
    out_path = Path(filename)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)
    plt.close(fig)

def plot_verification_results(mu_list, fom_factors, rom_factors):
    valid_pairs = [
        (fom, rom) for fom, rom in zip(fom_factors, rom_factors)
        if fom is not None and rom is not None
    ]
    if not valid_pairs:
        print("No valid converged FOM/ROM pairs available for parity plot.")
        return

    fom_factors = [pair[0] for pair in valid_pairs]
    rom_factors = [pair[1] for pair in valid_pairs]

    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Parity line
    min_val = min(min(fom_factors), min(rom_factors)) * 0.95
    max_val = max(max(fom_factors), max(rom_factors)) * 1.05
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label="Ideal (1:1)")
    
    # Data points
    errors = [abs(f - r)/f * 100 for f, r in zip(fom_factors, rom_factors)]
    scatter = ax.scatter(fom_factors, rom_factors, c=errors, cmap='viridis', edgecolors='k', s=80, label="Cases")
    
    cbar = plt.colorbar(scatter)
    cbar.set_label("Relative Error [%]")
    
    ax.set_xlabel("FOM Factor of Safety")
    ax.set_ylabel("ROM Factor of Safety")
    ax.set_title("C-Phi ROM Verification: Factor of Safety Parity")
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend()
    
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/verification_parity.png", dpi=200)
    print("Verification parity plot saved to: figures/verification_parity.png")
    plt.close(fig)

def GetRomManagerParameters():
    return KratosMultiphysics.Parameters(f"""{{
        "rom_stages_to_train" : ["ROM"],
        "rom_stages_to_test" : ["ROM"],
        "paralellism" : null,
        "projection_strategy": "galerkin",
        "save_gid_output": false,
        "save_vtk_output": false,
        "output_name": "id",
        "assembling_strategy": "global",
        "ROM":{{
            "svd_truncation_tolerance": {SVD_TRUNCATION_TOLERANCE},
            "model_part_name": "PorousDomain",
            "nodal_unknowns": ["DISPLACEMENT_X", "DISPLACEMENT_Y", "WATER_PRESSURE"]
        }}
    }}""")

if __name__ == "__main__":
    stub_library_path = EnsureUdsmShowExtraStub()
    ConfigureLdPreload(stub_library_path)

    SINGLE_POINT_MODE = True
    SINGLE_CPHI_PAIR = [10000.0, 35.0]

    RUN_STAGE0 = False
    RUN_STAGE1 = True
    RUN_STAGE2 = True
    RUN_STAGE3 = True
    RUN_STAGE4 = False

    STAGE1_FORCE_RECOMPUTE = True
    STAGE3_FORCE_RECOMPUTE_FOM = True
    STAGE3_FORCE_RECOMPUTE_ROM = True

    STAGE4_FORCE_RECOMPUTE_FOM = True
    STAGE4_FORCE_RECOMPUTE_ROM = True

    SVD_TRUNCATION_TOLERANCE = 1e-8 # <--- Exposed here now!
    ITERATION_SNAPSHOTS_PER_SOLVE_STEP = -1  # keep a subset of Newton states (always includes last)

    if SINGLE_POINT_MODE:
        mu_train = [SINGLE_CPHI_PAIR]
        mu_test = [SINGLE_CPHI_PAIR]
    else:
        # 3x3 boundary grid
        mu_train = get_grid_params(3, 3)
        mu_test = [[10000.0, 35.0], [11750.0, 33.5]]
    
    rom_manager = RomManager(
        project_parameters_name="ProjectParameters_stage2.json",
        general_rom_manager_parameters=GetRomManagerParameters(),
        UpdateMaterialParametersFile=UpdateMaterialParametersFile,
        UpdateProjectParameters=UpdateProjectParameters,
        CustomizeSimulation=CustomizeSimulation,
        mu_names=["cohesion", "friction_angle"],
        capture_nonconverged_snapshots_for_fom=True,
        iteration_snapshots_per_solve_step=ITERATION_SNAPSHOTS_PER_SOLVE_STEP,
    )

    if RUN_STAGE0:
        print("Stage 0: Plotting parameter space...")
        plot_mu_values(mu_train, mu_test)

    if RUN_STAGE1:
        print(f"Stage 1: Running FOM Training for {len(mu_train)} points...")
        rom_manager.stage1_fom_training(
            mu_train=mu_train,
            force_recompute=STAGE1_FORCE_RECOMPUTE,
        )
    
    if RUN_STAGE2:
        print("Stage 2: Building POD Basis...")
        rom_manager.stage2_build_pod_basis(
            mu_train=mu_train,
            load_basis_if_available=False,
        )
        print("SVD Process Finished successfully.")
        
    if RUN_STAGE3:
        print("Stage 3: Running ROM Verification...")
        # Since cphi breaks traditional sweeping, we assure the format is identical 
        verification_summary = rom_manager.stage3_rom_verification(
            mu_train=mu_train,
            basis_mu_train=mu_train,
            force_recompute_fom=STAGE3_FORCE_RECOMPUTE_FOM,
            force_recompute_rom=STAGE3_FORCE_RECOMPUTE_ROM,
        )
        
        # --- NEW: Visibility/Reporting ---
        print("\n" + "="*60)
        print(f"{'VERIFICATION SUMMARY (Factor of Safety)':^60}")
        print("="*60)
        print(f"{'Case':<8} | {'Mu (c, phi)':<20} | {'FOM FoS':<10} | {'ROM FoS':<10} | {'Error %':<8}")
        print("-"*60)
        
        fom_fos = []
        rom_fos = []
        for i, case in enumerate(verification_summary["per_case"]):
            mu_str = f"({case['mu'][0]:.0f}, {case['mu'][1]:.1f})"
            fom_val = case['critical_factor_fom']
            rom_val = case['critical_factor_rom']
            rom_attempted = case.get("critical_factor_rom_attempted")
            err = case['relative_l2_error_critical_factor_percent']

            if fom_val is not None and rom_val is not None:
                fom_fos.append(fom_val)
                rom_fos.append(rom_val)

            fom_str = f"{fom_val:.4f}" if fom_val is not None else "N/A"
            if rom_val is not None:
                rom_str = f"{rom_val:.4f}"
            elif rom_attempted is not None:
                rom_str = f"{rom_attempted:.4f}*"
            else:
                rom_str = "N/A"
            err_str = f"{err:.2f}%" if err is not None else "N/A"
            print(f"{i:<8} | {mu_str:<20} | {fom_str:<10} | {rom_str:<10} | {err_str:<8}")
        
        print("="*60)
        global_err = verification_summary["global_relative_l2_error_critical_factor_percent"]
        if global_err is None:
            print("Global Relative Error: N/A (insufficient converged ROM cases)")
        else:
            print(f"Global Relative Error: {global_err:.4f}%")
        print(f"Failed ROM cases: {verification_summary.get('num_failed_rom_cases', 'N/A')}")
        print("* ROM FoS shown with '*' is attempted (nonconverged) value.")
        print("="*60 + "\n")
        
        plot_verification_results(mu_train, fom_fos, rom_fos)
        
    if RUN_STAGE4:
        print("Stage 4: Running ROM Testing on Unseen Parameters...")
        
        # Generation of the sampling plot for Stage 4 documentation
        plot_mu_values(
            mu_train, 
            mu_test, 
            filename="rom_data/stage4_test/test_sampling_grid.png", 
            title="Stage 4: Evaluation on Unseen Parameters"
        )
        
        test_summary = rom_manager.stage4_rom_testing(
            mu_test=mu_test,
            force_recompute_fom=STAGE4_FORCE_RECOMPUTE_FOM,
            force_recompute_rom=STAGE4_FORCE_RECOMPUTE_ROM,
        )
        
        # --- Stage 4 Reporting ---
        print("\n" + "="*60)
        print(f"{'TESTING SUMMARY (Absolute Generalization)':^60}")
        print("="*60)
        print(f"{'Case':<8} | {'Mu (c, phi)':<20} | {'FOM FoS':<10} | {'ROM FoS':<10}")
        print("-"*60)
        
        for i, case in enumerate(test_summary["per_case"]):
            mu_str = f"({case['mu'][0]:.0f}, {case['mu'][1]:.1f})"
            fom_val = case['critical_factor_fom']
            rom_val = case['critical_factor_rom']
            rom_attempted = case.get("critical_factor_rom_attempted")
            fom_str = f"{fom_val:.4f}" if fom_val is not None else "N/A"
            if rom_val is not None:
                rom_str = f"{rom_val:.4f}"
            elif rom_attempted is not None:
                rom_str = f"{rom_attempted:.4f}*"
            else:
                rom_str = "N/A"
            print(f"Test_{i:<3} | {mu_str:<20} | {fom_str:<10} | {rom_str:<10}")
        
        print("="*60 + "\n")
