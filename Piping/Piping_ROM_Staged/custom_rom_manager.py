import json
import importlib
from datetime import datetime
from pathlib import Path
import KratosMultiphysics
import KratosMultiphysics.GeoMechanicsApplication as KratosGeo
import numpy as np
from KratosMultiphysics.RomApplication.calculate_rom_basis_output_process import CalculateRomBasisOutputProcess
from KratosMultiphysics.RomApplication.rom_testing_utilities import SetUpSimulationInstance

def _build_head_samples(min_max_step):
    """Build [[head_0], [head_1], ...] from [h_min, h_max, step]."""
    if min_max_step is None or len(min_max_step) != 3:
        raise Exception('"min_max_step" must be [h_min, h_max, step].')

    h_min, h_max, step = min_max_step
    h_min = float(h_min)
    h_max = float(h_max)
    if step <= 0.0:
        raise Exception(f"Head step must be > 0. Got step={step}.")
    if h_max <= h_min:
        raise Exception(f"Head range must satisfy h_max > h_min. Got [{h_min}, {h_max}].")
    
    head_values = []
    head = h_min
    while head < h_max:
        head_values.append([head])
        head += step

    if not head_values:
        raise Exception("No head samples were generated from min_max_step.")

    return head_values


class PipingAnalysis:
    """
    Runs one piping case (one mu) while sweeping head values.
    Final QoI is taken from the last stable head (previous step).
    """

    def __init__(self, simulation_type, base_parameters, get_analysis_stage_class, customize_simulation, head_samples):
        self.simulation_type = simulation_type
        self.base_parameters = base_parameters
        self.get_analysis_stage_class = get_analysis_stage_class
        self.customize_simulation = customize_simulation
        self.head_samples = head_samples

        self.solutions = []
        self.residual_norms = []
        self.pipe_lengths = []

        self.length = 0.0
        self.residual_norm = 0.0
        self.critical_head = head_samples[0][0]

    def _clone_parameters_with_head(self, head):
        parameters = self.base_parameters.Clone()
        constraints = parameters["processes"]["constraints_process_list"]
        
        # Identify the head boundary. We look for 'WATER_PRESSURE' and 'is_fixed'
        # Usually index 0 or 1.
        target_idx = 0
        for i in range(constraints.size()):
            p = constraints[i]["Parameters"]
            if p["variable_name"].GetString() == "WATER_PRESSURE" and p["is_fixed"].GetBool():
                # We update the one that isn't the fixed 'seepage' face (usually 0.0)
                if p["reference_coordinate"].GetDouble() > 0.1 or "1" in p["model_part_name"].GetString():
                    target_idx = i
                    break

        constraints[target_idx]["Parameters"]["reference_coordinate"].SetDouble(float(head))
        return parameters

    def _extract_snapshots_matrix(self, simulation):
        for process in simulation._GetListOfOutputProcesses():
            if isinstance(process, CalculateRomBasisOutputProcess):
                return process._GetSnapshotsMatrix()
        return None # Return None if not found instead of raising exception

    def _get_pipe_length(self, simulation):
        mp = simulation._GetSolver().GetComputingModelPart()
        elements = [e for e in mp.Elements if e.Has(KratosGeo.PIPE_ELEMENT_LENGTH)]
        return sum(e.GetValue(KratosGeo.PIPE_ELEMENT_LENGTH) for e in elements if e.GetValue(KratosGeo.PIPE_ACTIVE))

    def _all_pipe_active(self, simulation):
        mp = simulation._GetSolver().GetComputingModelPart()
        elements = [e for e in mp.Elements if e.Has(KratosGeo.PIPE_ELEMENT_LENGTH)]
        if not elements: return False
        return all(e.GetValue(KratosGeo.PIPE_ACTIVE) for e in elements)

    def _run_single_head(self, head):
        parameters = self._clone_parameters_with_head(head)
        model = KratosMultiphysics.Model()
        
        if self.simulation_type == "FOM":
            analysis_stage_class = self.get_analysis_stage_class(parameters)
        else:
            # For ROM, SetUpSimulationInstance is the way to go
            analysis_stage_class = type(SetUpSimulationInstance(model, parameters))

        simulation = self.customize_simulation(analysis_stage_class, model, parameters, self.simulation_type)
        simulation.Run()
        
        snapshots = self._extract_snapshots_matrix(simulation)

        return {
            "all_pipe_active": self._all_pipe_active(simulation),
            "pipe_length": self._get_pipe_length(simulation),
            "snapshots_matrix": snapshots,
            "error_indicator_ok": simulation.IsErroIndicatorAcceptable(),
            "residual_norm": simulation.GetResidualNorm(),
        }

    def Run(self):
        for i, head_data in enumerate(self.head_samples):
            current_head = head_data[0]
            try:
                result = self._run_single_head(current_head)
            except Exception as e:
                import traceback
                print(f"  [Error] Head sweep crashed at H={current_head}")
                # traceback.print_exc() # Muted for production
                break

            self.solutions.append(result["snapshots_matrix"])
            self.residual_norms.append(result["residual_norm"])
            self.pipe_lengths.append(result["pipe_length"])

            limit_reached = result["all_pipe_active"] if self.simulation_type == "FOM" else (not result["error_indicator_ok"] or result["all_pipe_active"])
            
            if limit_reached:
                stable_idx = max(0, i - 1)
                self.length = self.pipe_lengths[stable_idx]
                self.residual_norm = self.residual_norms[stable_idx]
                self.critical_head = self.head_samples[stable_idx][0]
                return
        
        # If sweep ends without reaching limit
        if self.pipe_lengths:
            self.length = self.pipe_lengths[-1]
            self.residual_norm = self.residual_norms[-1]
            self.critical_head = self.head_samples[-1][0]
        else:
            self.length = 0.0
            self.residual_norm = 0.0
            self.critical_head = self.head_samples[0][0]

    def GetFinalData(self):
        return {
            "pipe_length": float(self.length),
            "residual_norm": float(self.residual_norm),
            "critical_head": float(self.critical_head),
        }

    def GetSnapshotsMatrix(self):
        if not self.solutions or self.solutions[0] is None:
            return None
        return np.block(self.solutions)


class RomManager:
    def __init__(self, project_parameters_name, general_rom_manager_parameters, CustomizeSimulation, UpdateMaterialParametersFile, mu_names=None):
        self.project_parameters_name = project_parameters_name
        self.general_rom_manager_parameters = general_rom_manager_parameters
        self.CustomizeSimulation = CustomizeSimulation
        self.UpdateMaterialParametersFile = UpdateMaterialParametersFile
        self.mu_names = mu_names if mu_names else ["permeability_xx", "d70"]

        self.storage_root = Path("rom_data")
        self.stage1_fom = self.storage_root / "stage1_fom"
        self.stage2_pod = self.storage_root / "stage2_pod"
        self.stage3_rom = self.storage_root / "stage3_rom"
        self.stage4_test = self.storage_root / "stage4_test"
        self.reports_root = self.storage_root / "reports"

        for d in [self.stage1_fom, self.stage2_pod, self.stage3_rom, self.stage4_test, self.reports_root]:
            d.mkdir(parents=True, exist_ok=True)

    def _MuToken(self, mu):
        return f"k{mu[0]:.1e}_d{mu[1]:.1e}"

    @staticmethod
    def _MuListsMatch(mu_a, mu_b):
        array_a = np.asarray(mu_a, dtype=float)
        array_b = np.asarray(mu_b, dtype=float)
        return array_a.shape == array_b.shape and np.array_equal(array_a, array_b)

    def stage1_fom_training(self, mu_train, min_max_step, force_recompute=False):
        print(f"\n>>> Stage 1: FOM Training ({len(mu_train)} cases)")
        for mu in mu_train:
            token = self._MuToken(mu)
            snap_path = self.stage1_fom / f"fom_{token}.npy"
            qoi_path = self.stage1_fom / f"qoi_{token}.json"

            if qoi_path.exists() and not force_recompute:
                print(f" [Skip] Existing FOM found for {token}")
                continue

            print(f" [Run] FOM for {token}...")
            sim = self._CreateSimulation("FOM", mu, min_max_step)
            sim.Run()
            snapshots = sim.GetSnapshotsMatrix()
            if snapshots is not None:
                np.save(snap_path, snapshots)
            
            with open(qoi_path, "w") as f:
                json.dump(sim.GetFinalData(), f, indent=4)
        print(">>> Stage 1 Complete.")

    def stage2_build_pod_basis(self, mu_train, load_basis_if_available=True):
        print("\n>>> Stage 2: Building POD Basis")
        basis_path = self.stage2_pod / "basis.npy"
        sigma_path = self.stage2_pod / "singular_values.npy"
        meta_path = self.stage2_pod / "basis_meta.json"
        svd_truncation_tolerance = self.general_rom_manager_parameters["ROM"]["svd_truncation_tolerance"].GetDouble()

        reuse_existing_basis = False
        if basis_path.exists() and sigma_path.exists() and meta_path.exists() and load_basis_if_available:
            with open(meta_path, "r") as f:
                basis_meta = json.load(f)

            same_mu_train = self._MuListsMatch(basis_meta.get("mu_train", []), mu_train)
            same_svd_tolerance = np.isclose(
                basis_meta.get("svd_truncation_tolerance", np.nan),
                svd_truncation_tolerance,
            )
            reuse_existing_basis = same_mu_train and same_svd_tolerance

            if not reuse_existing_basis:
                print(" [Rebuild] Existing POD basis metadata does not match current sampling.")

        if reuse_existing_basis:
            print(" [Skip] Loading existing POD basis")
            U = np.load(basis_path)
        else:
            snapshots = []
            for mu in mu_train:
                p = self.stage1_fom / f"fom_{self._MuToken(mu)}.npy"
                if not p.exists(): raise Exception(f"Missing snapshot: {p}")
                snapshots.append(np.load(p))
            
            X = np.block(snapshots)
            print(f" [Run] SVD on snapshots with shape {X.shape}...")
            basis_process = self._CreateBasisOutputProcess()
            U, S = basis_process._ComputeSVD(X)
            
            # 1. Save Basis Matrix
            basis_process._PrintRomBasis(U, S)
            
            # 2. MANUALLY Populate NodeIds.npy because dummy simulation doesn't do it right
            # We load the actual mesh to get the IDs in the correct order Kratos expects (sorted)
            print(" [Run] Recovering nodal mapping from mesh...")
            model = KratosMultiphysics.Model()
            mp = model.CreateModelPart("PorousDomain")
            KratosMultiphysics.ModelPartIO("mesh").ReadModelPart(mp)
            # Filter nodes that have the WATER_PRESSURE variable (those are in the basis)
            node_ids = sorted([node.Id for node in mp.Nodes])
            np.save(self.stage2_pod / "NodeIds.npy", np.array(node_ids, dtype=int))
            print(f" [Done] Saved {len(node_ids)} node IDs to NodeIds.npy")
            
            np.save(sigma_path, S)
            with open(meta_path, "w") as f:
                json.dump(
                    {
                        "created_at": datetime.now().isoformat(),
                        "mu_train": mu_train,
                        "svd_truncation_tolerance": svd_truncation_tolerance,
                        "basis_shape": list(U.shape),
                        "sigma_shape": list(S.shape),
                    },
                    f,
                    indent=4,
                )

        self._RegisterBasisInParameters()
        print(">>> Stage 2 Complete.")

    def stage3_rom_verification(self, mu_train, min_max_step, force_fom=False, force_rom=False):
        print(f"\n>>> Stage 3: ROM Verification ({len(mu_train)} cases)")
        return self._RunComparison("stage3_rom", mu_train, min_max_step, force_fom, force_rom)

    def stage4_rom_test(self, mu_test, min_max_step, force_fom=False, force_rom=False):
        print(f"\n>>> Stage 4: ROM Testing ({len(mu_test)} cases)")
        return self._RunComparison("stage4_test", mu_test, min_max_step, force_fom, force_rom)

    def _RunComparison(self, stage_name, mu_list, min_max_step, force_fom, force_rom):
        results = []
        target_dir = getattr(self, stage_name)
        for mu in mu_list:
            token = self._MuToken(mu)
            fom_qoi_path = target_dir / f"qoi_fom_{token}.json"
            rom_qoi_path = target_dir / f"qoi_rom_{token}.json"

            # FOM Ground Truth
            if not fom_qoi_path.exists() or force_fom:
                s1_qoi = self.stage1_fom / f"qoi_{token}.json"
                if s1_qoi.exists() and not force_fom:
                    with open(s1_qoi, "r") as f: fom_data = json.load(f)
                else:
                    print(f" [Run] FOM for {token}...")
                    sim = self._CreateSimulation("FOM", mu, min_max_step)
                    sim.Run()
                    fom_data = sim.GetFinalData()
                with open(fom_qoi_path, "w") as f: json.dump(fom_data, f, indent=4)
            else:
                with open(fom_qoi_path, "r") as f: fom_data = json.load(f)

            # ROM Simulation
            if not rom_qoi_path.exists() or force_rom:
                print(f" [Run] ROM for {token}...")
                sim = self._CreateSimulation("ROM", mu, min_max_step)
                sim.Run()
                rom_data = sim.GetFinalData()
                with open(rom_qoi_path, "w") as f: json.dump(rom_data, f, indent=4)
            else:
                with open(rom_qoi_path, "r") as f: rom_data = json.load(f)

            results.append({
                "mu": mu, "token": token,
                "critical_head_fom": fom_data["critical_head"],
                "critical_head_rom": rom_data["critical_head"],
                "pipe_length_fom": fom_data["pipe_length"],
                "pipe_length_rom": rom_data["pipe_length"],
                "error_head_percent": 100.0 * abs(fom_data["critical_head"]-rom_data["critical_head"])/max(1e-6, fom_data["critical_head"])
            })

        summary = {"per_case": results}
        with open(self.reports_root / f"{stage_name}_summary.json", "w") as f: json.dump(summary, f, indent=4)
        return summary

    def _CreateSimulation(self, sim_type, mu, min_max_step):
        with open(self.project_parameters_name, "r") as f:
            parameters = KratosMultiphysics.Parameters(f.read())
        
        # 1. Update Material Parameters
        self.UpdateMaterialParametersFile("MaterialParameters.json", mu)
        
        # 2. Inject Basis Creation Process (REQUIRED for extracting snapshots)
        if not parameters["output_processes"].Has("rom_output"):
            parameters["output_processes"].AddEmptyArray("rom_output")
        else:
            parameters["output_processes"].RemoveValue("rom_output")
            parameters["output_processes"].AddEmptyArray("rom_output")
        
        basis_params = KratosMultiphysics.Parameters(f"""{{
            "python_module" : "calculate_rom_basis_output_process",
            "kratos_module" : "KratosMultiphysics.RomApplication",
            "process_name"  : "CalculateRomBasisOutputProcess",
            "Parameters"    : {{
                "model_part_name": "PorousDomain",
                "rom_manager" : true,
                "snapshots_control_type": "step",
                "snapshots_interval": 1.0,
                "nodal_unknowns": ["WATER_PRESSURE"],
                "rom_basis_output_format": "numpy",
                "rom_basis_output_name": "basis",
                "rom_basis_output_folder": "rom_data/stage2_pod",
                "svd_truncation_tolerance": {self.general_rom_manager_parameters["ROM"]["svd_truncation_tolerance"].GetDouble()}
            }}
        }}""")
        parameters["output_processes"]["rom_output"].Append(basis_params)

        # 3. Handle VTK/GiD output based on manager flags
        save_vtk = self.general_rom_manager_parameters["save_vtk_output"].GetBool()
        if not save_vtk and parameters["output_processes"].Has("vtk_output"):
            parameters["output_processes"].RemoveValue("vtk_output")
            
        save_gid = self.general_rom_manager_parameters["save_gid_output"].GetBool()
        if not save_gid and parameters["output_processes"].Has("gid_output"):
            parameters["output_processes"].RemoveValue("gid_output")

        def get_analysis_stage_class(params):
            module_name = params["analysis_stage"].GetString()
            class_name = module_name.split(".")[-1]
            class_name = "".join(x.title() for x in class_name.split("_"))
            module = importlib.import_module(module_name)
            return getattr(module, class_name)

        return PipingAnalysis(sim_type, parameters, get_analysis_stage_class, self.CustomizeSimulation, _build_head_samples(min_max_step))

    def _CreateBasisOutputProcess(self):
        model = KratosMultiphysics.Model()
        model.CreateModelPart("PorousDomain") # Ensure root model part exists for process initialization
        params = KratosMultiphysics.Parameters(f"""{{
            "model_part_name": "PorousDomain",
            "rom_basis_output_format": "numpy",
            "rom_basis_output_name": "basis",
            "rom_basis_output_folder": "rom_data/stage2_pod",
            "svd_truncation_tolerance": {self.general_rom_manager_parameters["ROM"]["svd_truncation_tolerance"].GetDouble()},
            "nodal_unknowns": ["WATER_PRESSURE"]
        }}""")
        return CalculateRomBasisOutputProcess(model, params)

    def _RegisterBasisInParameters(self):
        # Creates a minimal RomParameters.json. 
        # The basis_process._PrintRomBasis() already created the real metadata in stage2_pod/basis.json
        data = {
            "rom_settings": { 
                "rom_bns_settings": {"monotonicity_preserving": False} 
            },
            "projection_strategy": "galerkin",
            "assembling_strategy": "global",
            "rom_basis_output_name": "basis",
            "rom_basis_output_folder": "rom_data/stage2_pod",
            "train_hrom": False, "run_hrom": False
        }
        with open("RomParameters.json", "w") as f: json.dump(data, f, indent=4)

    def plot_parameter_space(self, mu_train, mu_test):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,6))
        
        # Extract k and d
        train_k = [m[0] for m in mu_train]
        train_d = [m[1] for m in mu_test if False] # Just a spacer
        train_d = [m[1] for m in mu_train]
        
        test_k = [m[0] for m in mu_test]
        test_d = [m[1] for m in mu_test]
        
        plt.scatter(train_k, train_d, c='blue', label='Training (Stage 1)', alpha=0.6)
        plt.scatter(test_k, test_d, c='red', marker='x', s=100, label='Testing (Stage 4)')
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Permeability k [m^2]')
        plt.ylabel('Layer Thickness d [m]')
        plt.title('ROM Parameter space (Stage 4)')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend()
        
        plot_path = self.stage4_test / "parameter_space.png"
        plt.savefig(plot_path, dpi=150)
        print(f" >>> Stage 4 Parameter Plot saved to {plot_path}")
        plt.close()

    def load_qoi_vector(self, mu_list, case_tag, simulation_type, qoi_name):
        stage_dir = self.stage3_rom if case_tag == "train" else self.stage4_test
        suffix = "fom" if simulation_type == "FOM" else "rom"
        values = []
        for mu in mu_list:
            path = stage_dir / f"qoi_{suffix}_{self._MuToken(mu)}.json"
            if not path.exists() and case_tag == "train":
                path = self.stage1_fom / f"qoi_{self._MuToken(mu)}.json"
            with open(path, "r") as f:
                values.append(json.load(f)[qoi_name])
        return np.array(values)

    def PrintErrors(self):
        print("\n[Summary] ROM tests complete. Reports available in rom_data/reports/")
