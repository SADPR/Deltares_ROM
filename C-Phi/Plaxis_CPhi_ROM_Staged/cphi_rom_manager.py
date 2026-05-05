import json
import hashlib
from datetime import datetime
from pathlib import Path
import numpy as np

import KratosMultiphysics
import KratosMultiphysics.GeoMechanicsApplication as KratosGeo
from KratosMultiphysics.RomApplication.calculate_rom_basis_output_process import (
    CalculateRomBasisOutputProcess,
)
from KratosMultiphysics.RomApplication.rom_testing_utilities import SetUpSimulationInstance


class CPhiAnalysis:
    """
    Core analysis wrapper for Strength Reduction (C-Phi) simulations.
    It runs Stage 1 (Initial) and then Stage 2 (Reduction) using a CustomSimulation.
    """
    def __init__(self, simulation_type, base_parameters, customize_simulation):
        self.simulation_type = simulation_type
        self.base_parameters = base_parameters
        self.customize_simulation = customize_simulation
        self.critical_factor = float('nan')
        self.residual_norm = 0.0
        self.solutions = None

    def _extract_snapshots_matrix(self, simulation):
        for process in simulation._GetListOfOutputProcesses():
            if isinstance(process, CalculateRomBasisOutputProcess):
                return process._GetSnapshotsMatrix()
        raise Exception("CalculateRomBasisOutputProcess not found in simulation.")

    def Run(self):
        from KratosMultiphysics.GeoMechanicsApplication.geomechanics_analysis import GeoMechanicsAnalysis
        
        # 1. Run Baseline (Stage 1) - Gravity Loading
        with open("ProjectParameters_stage1.json", "r") as f:
            p1 = KratosMultiphysics.Parameters(f.read())
        
        model = KratosMultiphysics.Model()
        stage1 = GeoMechanicsAnalysis(model, p1)
        stage1.Run()
        
        # 2. Run Strength Reduction (Stage 2)
        p2 = self.base_parameters
        if self.simulation_type == "FOM":
            analysis_stage_class = GeoMechanicsAnalysis
        else:
            # ROM Setup
            if not p2.Has("analysis_stage"):
                p2.AddString("analysis_stage", "KratosMultiphysics.GeoMechanicsApplication.geomechanics_analysis")
            
            # Alias for casing mismatch in RomApplication (expects GeomechanicsAnalysis)
            import KratosMultiphysics.GeoMechanicsApplication.geomechanics_analysis as gma
            if not hasattr(gma, "GeomechanicsAnalysis"):
                gma.GeomechanicsAnalysis = gma.GeoMechanicsAnalysis
                
            analysis_stage_class = type(SetUpSimulationInstance(model, p2))
            
        simulation = self.customize_simulation(
            analysis_stage_class,
            model,
            p2,
            self.simulation_type,
        )
        
        try:
            simulation.Run()
        except BaseException:
            # Failure is expected in Strength Reduction (collapse = non-convergence)
            pass
            
        self.solutions = self._extract_snapshots_matrix(simulation)
        
        # Factor of Safety = 1.0 (baseline) + increment (time)
        # We extract directly from ProcessInfo to ensure we get the last converged step
        # even if simulation.Run() crashed on the current step.
        try:
            converged_time = simulation._GetSolver().GetComputingModelPart().ProcessInfo[KratosMultiphysics.TIME]
        except:
            converged_time = simulation.time
            
        self.critical_factor = 1.0 + converged_time

    def GetFinalData(self):
        return {
            "critical_factor": float(self.critical_factor)
        }

    def GetSnapshotsMatrix(self):
        return self.solutions


class RomManager:
    """
    Transparent C-Phi ROM Manager. 
    Simplified for research: human-readable filenames and explicit stage folders.
    """
    def __init__(
        self,
        project_parameters_name="ProjectParameters.json",
        general_rom_manager_parameters=None,
        CustomizeSimulation=None,
        UpdateProjectParameters=None,
        UpdateMaterialParametersFile=None,
        mu_names=None,
    ):
        self.project_parameters_name = project_parameters_name
        self.mu_names = mu_names if mu_names is not None else ["cohesion", "friction_angle"]
        self.CustomizeSimulation = CustomizeSimulation or self.DefaultCustomizeSimulation
        self.UpdateProjectParameters = UpdateProjectParameters or self.DefaultUpdateProjectParameters
        self.UpdateMaterialParametersFile = UpdateMaterialParametersFile or self.DefaultUpdateMaterialParametersFile

        # Defaults
        if general_rom_manager_parameters is None:
            general_rom_manager_parameters = KratosMultiphysics.Parameters("{}")
        self.rom_params = general_rom_manager_parameters
        self._ValidateDefaults()

        # Paths
        self.storage_root = Path("rom_data")
        self.stage1_fom = self.storage_root / "stage1_fom"
        self.stage2_pod = self.storage_root / "stage2_pod"
        self.stage3_rom = self.storage_root / "stage3_rom"
        self.stage4_test = self.storage_root / "stage4_test"
        self.reports_root = self.storage_root / "reports"

        for d in [self.stage1_fom, self.stage2_pod, self.stage3_rom, self.stage4_test, self.reports_root]:
            d.mkdir(parents=True, exist_ok=True)

    def _ValidateDefaults(self):
        defaults = KratosMultiphysics.Parameters("""{
            "rom_stages_to_train" : ["ROM"],
            "rom_stages_to_test" : ["ROM"],
            "paralellism" : null,
            "save_gid_output": false,
            "save_vtk_output": false,
            "output_name": "id",
            "projection_strategy": "galerkin",
            "assembling_strategy": "global",
            "rom_error_indicator_tolerance": 1e-4,
            "ROM": {
                "svd_truncation_tolerance": 1e-5,
                "model_part_name": "PorousDomain",
                "nodal_unknowns": ["DISPLACEMENT_X", "DISPLACEMENT_Y", "WATER_PRESSURE"],
                "rom_basis_output_format": "numpy",
                "rom_basis_output_name": "RomParameters",
                "rom_basis_output_folder": "rom_data",
                "snapshots_control_type": "step",
                "snapshots_interval": 1
            }
        }""")
        self.rom_params.RecursivelyValidateAndAssignDefaults(defaults)

    def _MuToken(self, mu):
        # Format: c10000_p30.5
        return f"c{mu[0]:.0f}_phi{mu[1]:.1f}"

    def stage1_fom_training(self, mu_train, force_recompute=False):
        print(f"\n>>> Stage 1: FOM Training ({len(mu_train)} cases)")
        for mu in mu_train:
            token = self._MuToken(mu)
            snap_path = self.stage1_fom / f"fom_{token}.npy"
            qoi_path = self.stage1_fom / f"qoi_{token}.json"

            if snap_path.exists() and not force_recompute:
                print(f" [Skip] Found existing FOM for {token}")
                continue

            print(f" [Run] FOM for {token}...")
            sim = self._CreateSimulation("FOM", mu)
            sim.Run()
            
            np.save(snap_path, sim.GetSnapshotsMatrix())
            with open(qoi_path, "w") as f:
                json.dump(sim.GetFinalData(), f, indent=4)
        
        print(">>> Stage 1 Complete.\n")

    def stage2_build_pod_basis(self, mu_train, load_basis_if_available=True):
        print("\n>>> Stage 2: Building POD Basis")
        basis_path = self.stage2_pod / "basis.npy"
        sigma_path = self.stage2_pod / "singular_values.npy"

        if basis_path.exists() and load_basis_if_available:
            print(" [Load] Existing Basis found.")
            return

        # Stack snapshots
        all_snaps = []
        for mu in mu_train:
            token = self._MuToken(mu)
            all_snaps.append(np.load(self.stage1_fom / f"fom_{token}.npy"))
        
        snapshots_matrix = np.hstack(all_snaps)
        
        # Compute SVD using Kratos Process
        basis_process = self._CreateBasisOutputProcess()
        right_basis, sigmas = basis_process._ComputeSVD(snapshots_matrix)
        
        np.save(basis_path, right_basis)
        np.save(sigma_path, sigmas)
        
        # Also print/save RomParameters.json for Kratos
        basis_process._PrintRomBasis(right_basis, sigmas)
        print(f" [Done] Basis shape: {right_basis.shape}. Saved to {basis_path}")

    def stage3_rom_verification(self, mu_train, **kwargs):
        print(f"\n>>> Stage 3: ROM Verification ({len(mu_train)} cases)")
        results = []

        for mu in mu_train:
            token = self._MuToken(mu)
            snap_path = self.stage3_rom / f"rom_{token}.npy"
            qoi_path = self.stage3_rom / f"qoi_{token}.json"

            print(f" [Run] ROM for {token}...")
            sim = self._CreateSimulation("ROM", mu)
            sim.Run()

            np.save(snap_path, sim.GetSnapshotsMatrix())
            rom_data = sim.GetFinalData()
            with open(qoi_path, "w") as f:
                json.dump(rom_data, f, indent=4)
            
            # Load FOM for comparison
            with open(self.stage1_fom / f"qoi_{token}.json", "r") as f:
                fom_data = json.load(f)
            
            results.append({
                "mu": mu,
                "critical_factor_fom": fom_data["critical_factor"],
                "critical_factor_rom": rom_data["critical_factor"],
                "relative_l2_error_critical_factor_percent": 100.0 * abs(fom_data["critical_factor"] - rom_data["critical_factor"]) / fom_data["critical_factor"]
            })

        print(f"{'Case':<8} | {'Mu (c, phi)':<20} | {'FOM FoS':<10} | {'ROM FoS':<10} | {'Error %':<8}")
        print("-" * 60)
        for i, case in enumerate(results):
            mu_str = f"({case['mu'][0]:.0f}, {case['mu'][1]:.1f})"
            fom_val = case['critical_factor_fom']
            rom_val = case['critical_factor_rom']
            err = case['relative_l2_error_critical_factor_percent']
            print(f"Case_{i:<2} | {mu_str:<20} | {fom_val:<10.4f} | {rom_val:<10.4f} | {err:.4f}%")

        summary = {"per_case": results, "global_relative_l2_error_critical_factor_percent": np.mean([r["relative_l2_error_critical_factor_percent"] for r in results])}
        
        report_path = self.reports_root / "stage3_verification.json"
        with open(report_path, "w") as f:
            json.dump(summary, f, indent=4)
        
        print(">>> Stage 3 Complete.")
        return summary

    def stage4_rom_testing(self, mu_test, force_recompute_fom=False, force_recompute_rom=False, **kwargs):
        print(f"\n>>> Stage 4: ROM Testing (Evaluating {len(mu_test)} unseen cases)")
        results = []

        for mu in mu_test:
            token = self._MuToken(mu)
            snap_path = self.stage4_test / f"rom_{token}.npy"
            qoi_path = self.stage4_test / f"qoi_rom_{token}.json"
            fom_qoi_path = self.stage4_test / f"qoi_fom_{token}.json"

            # 1. Run ROM (Test)
            if qoi_path.exists() and not force_recompute_rom:
                print(f" [Skip] Found existing ROM for {token}")
                with open(qoi_path, "r") as f:
                    rom_data = json.load(f)
            else:
                print(f" [Run] ROM for {token}...")
                sim_rom = self._CreateSimulation("ROM", mu)
                sim_rom.Run()
                
                np.save(snap_path, sim_rom.GetSnapshotsMatrix())
                rom_data = sim_rom.GetFinalData()
                with open(qoi_path, "w") as f:
                    json.dump(rom_data, f, indent=4)

            # 2. Run FOM (Ground Truth for comparison)
            if fom_qoi_path.exists() and not force_recompute_fom:
                print(f" [Skip] Found existing FOM Ground Truth for {token}")
                with open(fom_qoi_path, "r") as f:
                    fom_data = json.load(f)
            else:
                print(f" [Run] FOM for {token} (Ground Truth)...")
                sim_fom = self._CreateSimulation("FOM", mu)
                sim_fom.Run()
                fom_data = sim_fom.GetFinalData()
                with open(fom_qoi_path, "w") as f:
                    json.dump(fom_data, f, indent=4)
            
            results.append({
                "mu": mu,
                "critical_factor_fom": fom_data["critical_factor"],
                "critical_factor_rom": rom_data["critical_factor"],
                "relative_l2_error_critical_factor_percent": 100.0 * abs(fom_data["critical_factor"] - rom_data["critical_factor"]) / fom_data["critical_factor"]
            })

        summary = {"per_case": results, "global_relative_l2_error_critical_factor_percent": np.mean([r["relative_l2_error_critical_factor_percent"] for r in results])}
        
        report_path = self.reports_root / "stage4_test.json"
        with open(report_path, "w") as f:
            json.dump(summary, f, indent=4)
        
        print(">>> Stage 4 Complete.")
        return summary

    def _CreateSimulation(self, sim_type, mu):
        # 1. Read and Update Base Parameters
        with open(self.project_parameters_name, "r") as f:
            params = KratosMultiphysics.Parameters(f.read())
        
        params = self.UpdateProjectParameters(params, mu)
        
        # 2. Add ROM Output (Snapshot creation)
        if not params.Has("output_processes"):
            params.AddEmptyValue("output_processes")
        
        if not params["output_processes"].Has("rom_output"):
            params["output_processes"].AddEmptyArray("rom_output")
        
        params["output_processes"]["rom_output"].Append(self._GetRomOutputProcessParams())
        
        # 3. Update Material File
        mat_file = params["solver_settings"]["material_import_settings"]["materials_filename"].GetString()
        self.UpdateMaterialParametersFile(mat_file, mu)

        return CPhiAnalysis(sim_type, params, self.CustomizeSimulation)

    def _GetRomOutputProcessParams(self):
        p = KratosMultiphysics.Parameters("""{
            "python_module" : "calculate_rom_basis_output_process",
            "kratos_module" : "KratosMultiphysics.RomApplication",
            "process_name"  : "CalculateRomBasisOutputProcess",
            "Parameters"    : {
                "model_part_name": "",
                "rom_manager" : true,
                "snapshots_control_type": "step",
                "snapshots_interval": 1,
                "nodal_unknowns": [],
                "rom_basis_output_format": "numpy",
                "rom_basis_output_name": "RomParameters",
                "rom_basis_output_folder": "rom_data",
                "svd_truncation_tolerance": 1e-3,
                "print_singular_values": false
            }
        }""")
        # Sync with manager settings
        rp = self.rom_params["ROM"]
        for key in ["model_part_name", "nodal_unknowns", "svd_truncation_tolerance", "snapshots_interval"]:
            if rp.Has(key):
                p["Parameters"].RemoveValue(key)
                p["Parameters"].AddValue(key, rp[key])
        return p

    def _CreateBasisOutputProcess(self):
        with open(self.project_parameters_name, "r") as f:
            params = KratosMultiphysics.Parameters(f.read())
        
        if not params.Has("output_processes"):
            params.AddEmptyValue("output_processes")
            
        if not params["output_processes"].Has("rom_output"):
            params["output_processes"].AddEmptyArray("rom_output")
            
        params["output_processes"]["rom_output"].Append(self._GetRomOutputProcessParams())
        
        model = KratosMultiphysics.Model()
        from KratosMultiphysics.GeoMechanicsApplication.geomechanics_analysis import GeoMechanicsAnalysis
        sim = self.CustomizeSimulation(GeoMechanicsAnalysis, model, params)
        sim.Initialize()
        for p in sim._GetListOfOutputProcesses():
            if isinstance(p, CalculateRomBasisOutputProcess):
                return p
        raise RuntimeError("CalculateRomBasisOutputProcess not found.")

    def DefaultUpdateProjectParameters(self, parameters, mu=None):
        return parameters

    def DefaultUpdateMaterialParametersFile(self, material_parameters_file_name=None, mu=None):
        pass

    def DefaultCustomizeSimulation(self, cls, global_model, parameters, mu=None):
        return cls(global_model, parameters)