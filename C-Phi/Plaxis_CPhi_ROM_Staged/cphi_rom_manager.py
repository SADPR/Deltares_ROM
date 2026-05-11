import json
import hashlib
import inspect
import types
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
    def __init__(
        self,
        simulation_type,
        base_parameters,
        customize_simulation,
        capture_nonconverged_snapshots=False,
        iteration_snapshots_per_solve_step=6,
    ):
        self.simulation_type = simulation_type
        self.base_parameters = base_parameters
        self.customize_simulation = customize_simulation
        self.capture_nonconverged_snapshots = capture_nonconverged_snapshots
        self.iteration_snapshots_per_solve_step = iteration_snapshots_per_solve_step
        self.critical_factor = float('nan')
        self.residual_norm = 0.0
        self.solutions = None
        self.converged_solutions = None
        self.nonconverged_solutions = None
        self.converged_times = []
        self.run_status = "not_run"
        self.exception_message = None
        self.last_attempted_time = float('nan')
        self.nonconverged_snapshots_data_list = []

    def _extract_snapshots_matrix(self, simulation):
        for process in simulation._GetListOfOutputProcesses():
            if isinstance(process, CalculateRomBasisOutputProcess):
                try:
                    return process._GetSnapshotsMatrix()
                except ValueError as exc:
                    # Non-converged C-Phi runs may stop before any snapshot is written.
                    # Keep the workflow alive and still report the converged FoS.
                    if "cannot be empty" in str(exc):
                        KratosMultiphysics.Logger.PrintWarning(
                            "CPhiAnalysis",
                            "No ROM snapshots were collected for this case. Returning an empty snapshot matrix."
                        )
                        return np.empty((0, 0))
                    raise
        raise Exception("CalculateRomBasisOutputProcess not found in simulation.")

    @staticmethod
    def _extract_nonconverged_snapshot_column(output_process):
        aux_data_array = []
        for snapshot_var in output_process.snapshot_variables_list:
            aux_data_array.append(
                np.array(
                    KratosMultiphysics.VariableUtils().GetSolutionStepValuesVector(
                        output_process.model_part.Nodes, snapshot_var, 0
                    ),
                    copy=False,
                )
            )
        return np.stack(aux_data_array, axis=1).reshape(-1, 1)

    def _select_iteration_indices(self, num_iterations):
        # Kratos stores [initial_state, iter_1, iter_2, ..., iter_last].
        # For ROM enrichment we keep Newton iteration states (iter_k), always
        # including the last one, and skip the initial state by default.
        if num_iterations <= 0:
            return np.array([], dtype=int)

        if num_iterations == 1:
            # No Newton iteration was performed; only initial state exists.
            return np.array([0], dtype=int)

        candidate_indices = np.arange(1, num_iterations, dtype=int)
        n_keep = int(self.iteration_snapshots_per_solve_step)
        if n_keep <= 0 or n_keep >= candidate_indices.size:
            return candidate_indices

        sampled_local = np.linspace(0, candidate_indices.size - 1, num=n_keep, dtype=int)
        sampled = candidate_indices[sampled_local]
        sampled[-1] = candidate_indices[-1]  # always keep last Newton state
        return np.unique(sampled)

    @staticmethod
    def _build_equation_id_lookup(dofs_array):
        eqid_by_node_var = {}
        for dof in dofs_array:
            node_id = int(dof.Id())
            var_name = dof.GetVariable().Name()
            eqid = int(dof.EquationId)
            eqid_by_node_var[(node_id, var_name)] = eqid
        return eqid_by_node_var

    def _nonconverged_matrix_to_snapshot_order(self, iter_matrix, dofs_array, output_process):
        eqid_by_node_var = self._build_equation_id_lookup(dofs_array)
        ordered_eqids = []
        n_rows = iter_matrix.shape[0]
        for node in output_process.model_part.Nodes:
            for snapshot_var in output_process.snapshot_variables_list:
                key = (int(node.Id), snapshot_var.Name())
                eqid = eqid_by_node_var.get(key, None)
                if eqid is None or eqid < 0 or eqid >= n_rows:
                    raise RuntimeError(f"Could not map DOF {(key[0], key[1])} into nonconverged solution matrix.")
                ordered_eqids.append(eqid)
        return iter_matrix[np.array(ordered_eqids, dtype=int), :]

    @staticmethod
    def _combine_snapshot_blocks(converged_snapshots, nonconverged_snapshots):
        if nonconverged_snapshots is None or nonconverged_snapshots.size == 0:
            return converged_snapshots
        if converged_snapshots is None or converged_snapshots.size == 0:
            return nonconverged_snapshots
        if converged_snapshots.shape[0] != nonconverged_snapshots.shape[0]:
            KratosMultiphysics.Logger.PrintWarning(
                "CPhiAnalysis",
                "Converged and nonconverged snapshot row sizes differ. Using only converged snapshots.",
            )
            return converged_snapshots
        return np.hstack([converged_snapshots, nonconverged_snapshots])

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

        run_exception = None
        original_finalize_solution_step = simulation.FinalizeSolutionStep
        solver = simulation._GetSolver()
        original_solve_solution_step = solver.SolveSolutionStep
        rom_output_process_cache = None

        def _tracked_finalize_solution_step(this):
            original_finalize_solution_step()
            try:
                converged_time = this._GetSolver().GetComputingModelPart().ProcessInfo[KratosMultiphysics.TIME]
            except Exception:
                converged_time = getattr(this, "time", float("nan"))
            self.converged_times.append(float(converged_time))

        def _tracked_solve_solution_step(this):
            nonlocal rom_output_process_cache
            strategy = getattr(this, "solving_strategy", None)
            if self.capture_nonconverged_snapshots and strategy is not None and hasattr(strategy, "SetUpNonconvergedSolutionsFlag"):
                strategy.SetUpNonconvergedSolutionsFlag(True)
            converged = original_solve_solution_step()
            if self.capture_nonconverged_snapshots:
                if rom_output_process_cache is None:
                    for process in simulation._GetListOfOutputProcesses():
                        if isinstance(process, CalculateRomBasisOutputProcess):
                            rom_output_process_cache = process
                            break
                if (
                    strategy is not None
                    and hasattr(strategy, "GetNonconvergedSolutions")
                    and rom_output_process_cache is not None
                ):
                    iter_matrix, dofs_array = strategy.GetNonconvergedSolutions()
                    iter_matrix = np.asarray(iter_matrix)
                    if iter_matrix.size > 0 and iter_matrix.shape[1] > 0:
                        try:
                            reordered = self._nonconverged_matrix_to_snapshot_order(
                                iter_matrix, dofs_array, rom_output_process_cache
                            )
                            selected_indices = self._select_iteration_indices(reordered.shape[1])
                            if selected_indices.size > 0:
                                self.nonconverged_snapshots_data_list.extend(
                                    [reordered[:, i:i + 1] for i in selected_indices]
                                )
                        except Exception as exc:
                            KratosMultiphysics.Logger.PrintWarning(
                                "CPhiAnalysis",
                                f"Could not map iteration snapshots from strategy ({exc}). "
                                "Falling back to end-of-solve snapshot only."
                            )
                            self.nonconverged_snapshots_data_list.append(
                                self._extract_nonconverged_snapshot_column(rom_output_process_cache)
                            )
            return converged

        simulation.FinalizeSolutionStep = types.MethodType(_tracked_finalize_solution_step, simulation)
        solver.SolveSolutionStep = types.MethodType(_tracked_solve_solution_step, solver)

        try:
            simulation.Run()
        except BaseException as exc:
            # Failure is expected in Strength Reduction (collapse = non-convergence)
            run_exception = exc
        finally:
            simulation.FinalizeSolutionStep = original_finalize_solution_step
            solver.SolveSolutionStep = original_solve_solution_step

        self.converged_solutions = self._extract_snapshots_matrix(simulation)
        if self.nonconverged_snapshots_data_list:
            self.nonconverged_solutions = np.hstack(self.nonconverged_snapshots_data_list)
        else:
            self.nonconverged_solutions = np.empty((0, 0))
        self.solutions = self._combine_snapshot_blocks(self.converged_solutions, self.nonconverged_solutions)

        try:
            self.last_attempted_time = float(
                simulation._GetSolver().GetComputingModelPart().ProcessInfo[KratosMultiphysics.TIME]
            )
        except Exception:
            self.last_attempted_time = float(getattr(simulation, "time", float("nan")))

        # Factor of Safety = 1.0 + last converged stage-2 time.
        # Do not use attempted TIME after a failed retry cycle.
        if self.converged_times:
            self.critical_factor = 1.0 + self.converged_times[-1]
        else:
            self.critical_factor = float("nan")

        if run_exception is None:
            self.run_status = "converged"
        else:
            self.run_status = "nonconverged_with_progress" if self.converged_times else "nonconverged_no_progress"
            self.exception_message = str(run_exception)

    def GetFinalData(self):
        critical_factor = None if np.isnan(self.critical_factor) else float(self.critical_factor)
        last_attempted_time = None if np.isnan(self.last_attempted_time) else float(self.last_attempted_time)
        attempted_critical_factor = None if last_attempted_time is None else 1.0 + last_attempted_time
        last_converged_time = None if not self.converged_times else float(self.converged_times[-1])
        converged_snapshot_columns = 0 if self.converged_solutions is None or self.converged_solutions.size == 0 else int(self.converged_solutions.shape[1])
        nonconverged_snapshot_columns = 0 if self.nonconverged_solutions is None or self.nonconverged_solutions.size == 0 else int(self.nonconverged_solutions.shape[1])
        return {
            "critical_factor": critical_factor,
            "critical_factor_attempted": attempted_critical_factor,
            "run_status": self.run_status,
            "last_converged_time": last_converged_time,
            "last_attempted_time": last_attempted_time,
            "converged_steps": len(self.converged_times),
            "num_converged_snapshots": converged_snapshot_columns,
            "num_nonconverged_snapshots": nonconverged_snapshot_columns,
            "exception": self.exception_message,
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
        capture_nonconverged_snapshots_for_fom=True,
        iteration_snapshots_per_solve_step=6,
    ):
        self.project_parameters_name = project_parameters_name
        self.mu_names = mu_names if mu_names is not None else ["cohesion", "friction_angle"]
        self.CustomizeSimulation = CustomizeSimulation or self.DefaultCustomizeSimulation
        self.UpdateProjectParameters = UpdateProjectParameters or self.DefaultUpdateProjectParameters
        self.UpdateMaterialParametersFile = UpdateMaterialParametersFile or self.DefaultUpdateMaterialParametersFile
        self.capture_nonconverged_snapshots_for_fom = capture_nonconverged_snapshots_for_fom
        self.iteration_snapshots_per_solve_step = iteration_snapshots_per_solve_step

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
            qoi = sim.GetFinalData()
            
            np.save(snap_path, sim.GetSnapshotsMatrix())
            with open(qoi_path, "w") as f:
                json.dump(qoi, f, indent=4)
            print(
                f"      snapshots: converged={qoi.get('num_converged_snapshots', 0)}, "
                f"nonconverged={qoi.get('num_nonconverged_snapshots', 0)}"
            )
        
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

            fom_cf = fom_data.get("critical_factor")
            rom_cf = rom_data.get("critical_factor")
            rel_err = self._RelativeErrorPercent(fom_cf, rom_cf)
            
            results.append({
                "mu": mu,
                "critical_factor_fom": fom_cf,
                "critical_factor_rom": rom_cf,
                "critical_factor_rom_attempted": rom_data.get("critical_factor_attempted"),
                "relative_l2_error_critical_factor_percent": rel_err,
                "rom_run_status": rom_data.get("run_status"),
                "rom_converged_steps": rom_data.get("converged_steps"),
                "rom_last_converged_time": rom_data.get("last_converged_time"),
            })

        print(f"{'Case':<8} | {'Mu (c, phi)':<20} | {'FOM FoS':<10} | {'ROM FoS':<10} | {'Error %':<8}")
        print("-" * 60)
        for i, case in enumerate(results):
            mu_str = f"({case['mu'][0]:.0f}, {case['mu'][1]:.1f})"
            fom_val = self._FormatFoS(case['critical_factor_fom'])
            rom_val = self._FormatFoS(case['critical_factor_rom'])
            err = case['relative_l2_error_critical_factor_percent']
            err_str = f"{err:.4f}%" if err is not None else "N/A"
            print(f"Case_{i:<2} | {mu_str:<20} | {fom_val:<10} | {rom_val:<10} | {err_str:<8}")

        valid_errors = [r["relative_l2_error_critical_factor_percent"] for r in results if r["relative_l2_error_critical_factor_percent"] is not None]
        summary = {
            "per_case": results,
            "global_relative_l2_error_critical_factor_percent": float(np.mean(valid_errors)) if valid_errors else None,
            "num_valid_error_cases": len(valid_errors),
            "num_failed_rom_cases": sum(1 for r in results if r["critical_factor_rom"] is None),
        }
        
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

            fom_cf = fom_data.get("critical_factor")
            rom_cf = rom_data.get("critical_factor")
            rel_err = self._RelativeErrorPercent(fom_cf, rom_cf)
            
            results.append({
                "mu": mu,
                "critical_factor_fom": fom_cf,
                "critical_factor_rom": rom_cf,
                "critical_factor_rom_attempted": rom_data.get("critical_factor_attempted"),
                "relative_l2_error_critical_factor_percent": rel_err,
                "rom_run_status": rom_data.get("run_status"),
                "rom_converged_steps": rom_data.get("converged_steps"),
                "rom_last_converged_time": rom_data.get("last_converged_time"),
            })

        valid_errors = [r["relative_l2_error_critical_factor_percent"] for r in results if r["relative_l2_error_critical_factor_percent"] is not None]
        summary = {
            "per_case": results,
            "global_relative_l2_error_critical_factor_percent": float(np.mean(valid_errors)) if valid_errors else None,
            "num_valid_error_cases": len(valid_errors),
            "num_failed_rom_cases": sum(1 for r in results if r["critical_factor_rom"] is None),
        }
        
        report_path = self.reports_root / "stage4_test.json"
        with open(report_path, "w") as f:
            json.dump(summary, f, indent=4)
        
        print(">>> Stage 4 Complete.")
        return summary

    def _CreateSimulation(self, sim_type, mu):
        # 1. Read and Update Base Parameters
        with open(self.project_parameters_name, "r") as f:
            params = KratosMultiphysics.Parameters(f.read())

        try:
            n_params = len(inspect.signature(self.UpdateProjectParameters).parameters)
        except (TypeError, ValueError):
            n_params = 2

        if n_params >= 3:
            params = self.UpdateProjectParameters(params, mu, sim_type)
        else:
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

        capture_nonconverged_snapshots = sim_type == "FOM" and self.capture_nonconverged_snapshots_for_fom
        return CPhiAnalysis(
            sim_type,
            params,
            self.CustomizeSimulation,
            capture_nonconverged_snapshots=capture_nonconverged_snapshots,
            iteration_snapshots_per_solve_step=self.iteration_snapshots_per_solve_step,
        )

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

    @staticmethod
    def _RelativeErrorPercent(reference_value, model_value):
        if reference_value is None or model_value is None:
            return None
        ref = float(reference_value)
        mod = float(model_value)
        if not np.isfinite(ref) or not np.isfinite(mod) or ref == 0.0:
            return None
        return 100.0 * abs(ref - mod) / abs(ref)

    @staticmethod
    def _FormatFoS(value):
        if value is None:
            return "N/A"
        return f"{value:.4f}"

    def DefaultUpdateMaterialParametersFile(self, material_parameters_file_name=None, mu=None):
        pass

    def DefaultCustomizeSimulation(self, cls, global_model, parameters, mu=None):
        return cls(global_model, parameters)
