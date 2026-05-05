import importlib
import json
import hashlib
from datetime import datetime
from pathlib import Path

import KratosMultiphysics
import KratosMultiphysics.GeoMechanicsApplication as KratosGeo
import numpy as np
from KratosMultiphysics.RomApplication.calculate_rom_basis_output_process import (
    CalculateRomBasisOutputProcess,
)
from KratosMultiphysics.RomApplication.rom_testing_utilities import SetUpSimulationInstance


def _build_head_samples(min_max_step):
    """Build [[head_0], [head_1], ...] from [h_min, h_max, step]."""
    if min_max_step is None or len(min_max_step) != 3:
        raise Exception('"min_max_step" must be [h_min, h_max, step].')

    h_min, h_max, step = min_max_step
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

    Stop criterion:
    - FOM: first head where all pipe elements are active
    - ROM: first head where ROM error indicator fails OR all pipe elements active

    Final QoI is taken from the last stable head (previous step).
    """

    def __init__(self, simulation_type, base_parameters, get_analysis_stage_class, customize_simulation, head_samples):
        if simulation_type not in ["FOM", "ROM"]:
            raise Exception(f'Unknown simulation_type "{simulation_type}". Use "FOM" or "ROM".')

        self.simulation_type = simulation_type
        self.base_parameters = base_parameters
        self.get_analysis_stage_class = get_analysis_stage_class
        self.customize_simulation = customize_simulation
        self.head_samples = head_samples

        self.solutions = []
        self.residual_norms = []
        self.pipe_lengths = []

        self._critical_head_metadata = None
        self.length = np.nan
        self.residual_norm = np.nan
        self.critical_head = np.nan

    def _clone_parameters_with_head(self, head):
        """Set the reference head in constraints_process_list and return cloned parameters."""
        parameters = self.base_parameters.Clone()

        constraints = parameters["processes"]["constraints_process_list"]
        if constraints.size() < 2:
            raise Exception("Expected at least 2 entries in processes.constraints_process_list.")

        if "Left" in constraints[0]["Parameters"]["model_part_name"].GetString():
            target_idx = 0
        else:
            target_idx = 1

        constraints[target_idx]["Parameters"]["reference_coordinate"].SetDouble(head)
        return parameters

    def _extract_snapshots_matrix(self, simulation):
        for process in simulation._GetListOfOutputProcesses():
            if isinstance(process, CalculateRomBasisOutputProcess):
                return process._GetSnapshotsMatrix()

        raise Exception(
            "CalculateRomBasisOutputProcess not found in output processes. "
            "ROM snapshots cannot be extracted."
        )

    @staticmethod
    def _get_pipe_model_part(simulation):
        for process in simulation._GetListOfOutputProcesses():
            if hasattr(process, "model_part"):
                model_part = process.model_part
                has_pipe_elements = any(
                    element.Has(KratosGeo.PIPE_ELEMENT_LENGTH) for element in model_part.Elements
                )
                if has_pipe_elements:
                    return model_part

        try:
            return simulation._GetSolver().GetComputingModelPart()
        except Exception:
            return None

    @classmethod
    def _pipe_elements(cls, simulation):
        model_part = cls._get_pipe_model_part(simulation)
        if model_part is None:
            return []
        return [element for element in model_part.Elements if element.Has(KratosGeo.PIPE_ELEMENT_LENGTH)]

    @classmethod
    def _all_pipe_active(cls, simulation):
        pipe_elements = cls._pipe_elements(simulation)
        if not pipe_elements:
            return False
        return all(element.GetValue(KratosGeo.PIPE_ACTIVE) for element in pipe_elements)

    @classmethod
    def _pipe_length(cls, simulation):
        elements = cls._pipe_elements(simulation)
        return sum(
            element.GetValue(KratosGeo.PIPE_ELEMENT_LENGTH)
            for element in elements
            if element.GetValue(KratosGeo.PIPE_ACTIVE)
        )

    def _run_single_head(self, head):
        parameters = self._clone_parameters_with_head(head)
        model = KratosMultiphysics.Model()

        if self.simulation_type == "FOM":
            analysis_stage_class = self.get_analysis_stage_class(parameters)
        else:
            analysis_stage_class = type(SetUpSimulationInstance(model, parameters))

        simulation = self.customize_simulation(
            analysis_stage_class,
            model,
            parameters,
            self.simulation_type,
        )
        simulation.Run()

        snapshots_matrix = self._extract_snapshots_matrix(simulation)
        residual_norm = simulation.GetResidualNorm()
        error_indicator_ok = simulation.IsErroIndicatorAcceptable()
        all_pipe_active = self._all_pipe_active(simulation)
        pipe_length = self._pipe_length(simulation)

        return {
            "all_pipe_active": all_pipe_active,
            "pipe_length": pipe_length,
            "snapshots_matrix": snapshots_matrix,
            "error_indicator_ok": error_indicator_ok,
            "residual_norm": residual_norm,
        }

    def _find_critical_state_linear_search(self):
        for i, head_data in enumerate(self.head_samples):
            current_head = head_data[0]
            result = self._run_single_head(current_head)

            self.solutions.append(result["snapshots_matrix"])
            self.residual_norms.append(result["residual_norm"])
            self.pipe_lengths.append(result["pipe_length"])

            if self.simulation_type == "FOM":
                reached_limit = result["all_pipe_active"]
            else:
                reached_limit = (not result["error_indicator_ok"]) or result["all_pipe_active"]

            if reached_limit:
                if i == 0:
                    raise Exception(
                        "First tested head already reaches the stop criterion. "
                        "Use a lower starting head or increase ROM training robustness."
                    )

                stable_idx = i - 1
                return {
                    "critical_head_metadata": self.head_samples[stable_idx],
                    "pipe_length": self.pipe_lengths[stable_idx],
                    "residual_norm": self.residual_norms[stable_idx],
                    "critical_head": self.head_samples[stable_idx][0],
                }

        return {
            "critical_head_metadata": None,
            "pipe_length": np.nan,
            "residual_norm": np.nan,
            "critical_head": np.nan,
        }

    def Run(self):
        result = self._find_critical_state_linear_search()
        self._critical_head_metadata = result["critical_head_metadata"]
        self.length = result["pipe_length"]
        self.residual_norm = result["residual_norm"]
        self.critical_head = result["critical_head"]

    def GetFinalData(self):
        return {
            "pipe_length": np.array(self.length),
            "residual_norm": np.array(self.residual_norm),
            "critical_head": self.critical_head,
        }

    def GetSnapshotsMatrix(self):
        if not self.solutions:
            raise Exception("No snapshots were collected in PipingAnalysis.")
        return np.block(self.solutions)


class RomManager:
    """
    Staged piping ROM manager (no SQL database).

    Main stages:
    - Stage 1: FOM training snapshots
    - Stage 2: POD basis
    - Stage 3: ROM verification on training set
    - Stage 4: ROM test on unseen set
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
        self.mu_names = mu_names if mu_names is not None else ["permeability_xx", "d70"]

        self._SetUpRomManagerParameters(general_rom_manager_parameters)

        self.CustomizeSimulation = CustomizeSimulation or self.DefaultCustomizeSimulation
        self.UpdateProjectParameters = UpdateProjectParameters or self.DefaultUpdateProjectParameters
        self.UpdateMaterialParametersFile = (
            UpdateMaterialParametersFile or self.DefaultUpdateMaterialParametersFile
        )

        self.SetupErrorsDictionaries()

        rom_folder = Path(self.general_rom_manager_parameters["ROM"]["rom_basis_output_folder"].GetString())
        self.storage_root = rom_folder / "staged_piping"
        self.fom_root = self.storage_root / "fom"
        self.rom_root = self.storage_root / "rom"
        self.pod_root = self.storage_root / "pod"
        self.qoi_root = self.storage_root / "qoi"
        self.reports_root = self.storage_root / "reports"
        self.case_meta_root = self.storage_root / "case_meta"
        self.index_root = self.storage_root / "index"
        self.case_index_file = self.index_root / "cases_index.json"

        for directory in [
            self.storage_root,
            self.fom_root,
            self.rom_root,
            self.pod_root,
            self.qoi_root,
            self.reports_root,
            self.case_meta_root,
            self.index_root,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        self._active_basis_token = None
        self._active_basis_version = None
        self._active_basis_mu_train = None

        self._project_parameters_signature = self._HashFileContent(self.project_parameters_name)
        self._general_rom_signature = self._PayloadDigest(
            self._ParametersToDict(self.general_rom_manager_parameters),
            short=False,
        )

    # ---------------------------------------------------------------------
    # Public staged API
    # ---------------------------------------------------------------------

    def stage1_fom_training(self, mu_train, min_max_step, force_recompute=False):
        self._CheckLinearOnly()
        self._CheckSupportedProjection()

        summary = self._RunPipingCases(
            simulation_type="FOM",
            mu_list=mu_train,
            case_tag="train",
            min_max_step=min_max_step,
            run_missing_only=not force_recompute,
            force_recompute=force_recompute,
            results_name="FOM_Train",
        )
        summary["stage"] = "stage1_fom_training"
        self._WriteReport("stage1_fom_training_summary.json", summary)
        return summary

    def stage2_build_pod_basis(self, mu_train, load_basis_if_available=True):
        self._CheckLinearOnly()
        self._CheckSupportedProjection()

        basis_path, sigma_path, meta_path = self._GetBasisFiles(mu_train)
        basis_token = self._BasisToken(mu_train)
        basis_exists = basis_path.exists() and sigma_path.exists()

        if basis_exists and load_basis_if_available:
            right_basis = np.load(basis_path)
            singular_values = np.load(sigma_path)
            if meta_path.exists():
                meta = self._ReadJson(meta_path)
                basis_version = meta.get("basis_version", meta.get("created_at"))
            else:
                basis_version = datetime.fromtimestamp(basis_path.stat().st_mtime).isoformat()
            action = "loaded_existing_basis"
        else:
            snapshots_matrix = self._StackSnapshots(mu_train, stage="fom", case_tag="train")
            basis_output_process = self.InitializeDummySimulationForBasisOutputProcess()
            right_basis, singular_values = basis_output_process._ComputeSVD(snapshots_matrix)

            np.save(basis_path, right_basis)
            np.save(sigma_path, singular_values)
            basis_version = datetime.now().isoformat()
            action = "created_basis"

        self._WriteJson(
            meta_path,
            {
                "created_at": datetime.now().isoformat(),
                "basis_version": basis_version,
                "mu_train": mu_train,
                "basis_token": basis_token,
                "basis_shape": list(right_basis.shape),
                "sigma_shape": list(singular_values.shape),
                "svd_truncation_tolerance": self.general_rom_manager_parameters["ROM"][
                    "svd_truncation_tolerance"
                ].GetDouble(),
            },
        )

        self._ActivateBasis(
            mu_train,
            right_basis,
            singular_values,
            basis_version,
            print_to_rom_parameters=True,
        )

        summary = {
            "stage": "stage2_build_pod_basis",
            "action": action,
            "mu_train": mu_train,
            "basis_token": basis_token,
            "basis_version": basis_version,
            "basis_file": str(basis_path),
            "sigma_file": str(sigma_path),
            "basis_shape": list(right_basis.shape),
            "sigma_shape": list(singular_values.shape),
        }
        self._WriteReport("stage2_pod_basis_summary.json", summary)
        return summary

    def stage3_rom_verification(
        self,
        mu_train,
        basis_mu_train=None,
        min_max_step=None,
        force_recompute_fom=False,
        force_recompute_rom=False,
    ):
        if basis_mu_train is None:
            basis_mu_train = mu_train

        summary = self._RunRomComparisonStage(
            stage_name="stage3_rom_verification",
            case_tag="train",
            mu_list=mu_train,
            basis_mu_train=basis_mu_train,
            min_max_step=min_max_step,
            force_recompute_fom=force_recompute_fom,
            force_recompute_rom=force_recompute_rom,
            fom_results_name="FOM_Verification",
            rom_results_name="ROM_Verification",
            report_name="stage3_rom_verification_summary.json",
        )
        self.ROMvsFOM["Verification"] = summary["global_relative_l2_error"]
        self.ROMvsFOM["Fit"] = summary["global_relative_l2_error"]
        return summary

    def stage4_rom_test(
        self,
        mu_test,
        basis_mu_train,
        min_max_step,
        force_recompute_fom=False,
        force_recompute_rom=False,
    ):
        summary = self._RunRomComparisonStage(
            stage_name="stage4_rom_test",
            case_tag="test",
            mu_list=mu_test,
            basis_mu_train=basis_mu_train,
            min_max_step=min_max_step,
            force_recompute_fom=force_recompute_fom,
            force_recompute_rom=force_recompute_rom,
            fom_results_name="FOM_Test",
            rom_results_name="ROM_Test",
            report_name="stage4_rom_test_summary.json",
        )
        self.ROMvsFOM["Test"] = summary["global_relative_l2_error"]
        return summary

    # Backward-compatible wrappers -------------------------------------------------

    def FitPiping(self, mu_train=[None], mu_validation=[None], min_max_step=[None]):
        self.stage1_fom_training(mu_train=mu_train, min_max_step=min_max_step, force_recompute=False)
        self.stage2_build_pod_basis(mu_train=mu_train, load_basis_if_available=True)
        self.stage3_rom_verification(
            mu_train=mu_train,
            basis_mu_train=mu_train,
            min_max_step=min_max_step,
            force_recompute_fom=False,
            force_recompute_rom=False,
        )

    def TestPiping(self, mu_test=[None], mu_train=[None], min_max_step=[None]):
        self.stage4_rom_test(
            mu_test=mu_test,
            basis_mu_train=mu_train,
            min_max_step=min_max_step,
            force_recompute_fom=False,
            force_recompute_rom=False,
        )

    # Simple data access for postprocess ------------------------------------------

    def load_qoi_vector(self, mu_list, case_tag, simulation_type, qoi_name):
        stage = self._NormalizeSnapshotStage(simulation_type)
        values = []

        for mu in mu_list:
            qoi_payload = self._LoadQoiPayload(
                stage=stage,
                case_tag=case_tag,
                mu=mu,
                expected_basis_token=self._active_basis_token if stage == "rom" else None,
                expected_basis_version=self._active_basis_version if stage == "rom" else None,
            )
            if qoi_payload is None:
                raise RuntimeError(
                    f"Missing QoI data for simulation_type='{simulation_type}', case_tag='{case_tag}', mu={mu}."
                )

            final_data = qoi_payload["final_data"]
            if qoi_name not in final_data:
                raise RuntimeError(
                    f"QoI '{qoi_name}' not found for simulation_type='{simulation_type}', case_tag='{case_tag}', mu={mu}."
                )
            values.append(float(final_data[qoi_name]))

        return np.array(values)

    def GenerateDatabaseSummary(self):
        summary = {
            "note": "This manager uses staged files, not SQL RomDatabase.",
            "storage_root": str(self.storage_root),
            "fom_root": str(self.fom_root),
            "rom_root": str(self.rom_root),
            "pod_root": str(self.pod_root),
            "qoi_root": str(self.qoi_root),
            "case_meta_root": str(self.case_meta_root),
            "case_index_file": str(self.case_index_file),
        }
        self._WriteReport("storage_summary.json", summary)

    def GenerateDatabaseCompleteDump(self):
        self.GenerateDatabaseSummary()

    def SetupErrorsDictionaries(self):
        self.ROMvsFOM = {
            "Fit": None,
            "Verification": None,
            "Test": None,
        }

    def PrintErrors(self):
        for key in ["Fit", "Verification", "Test"]:
            value = self.ROMvsFOM.get(key)
            if value is None:
                print(f"approximation error {key}: not computed")
            else:
                print(f"approximation error {key}: {value}")

    # ---------------------------------------------------------------------
    # Internal staged logic
    # ---------------------------------------------------------------------

    def _RunRomComparisonStage(
        self,
        stage_name,
        case_tag,
        mu_list,
        basis_mu_train,
        min_max_step,
        force_recompute_fom,
        force_recompute_rom,
        fom_results_name,
        rom_results_name,
        report_name,
    ):
        self._CheckLinearOnly()
        self._CheckSupportedProjection()
        self._EnsureBasisLoaded(basis_mu_train)

        fom_stage_summary = self._RunPipingCases(
            simulation_type="FOM",
            mu_list=mu_list,
            case_tag=case_tag,
            min_max_step=min_max_step,
            run_missing_only=not force_recompute_fom,
            force_recompute=force_recompute_fom,
            results_name=fom_results_name,
        )

        rom_stage_summary = self._RunPipingCases(
            simulation_type="ROM",
            mu_list=mu_list,
            case_tag=case_tag,
            min_max_step=min_max_step,
            run_missing_only=not force_recompute_rom,
            force_recompute=force_recompute_rom,
            results_name=rom_results_name,
        )

        per_case = []
        fom_critical_heads = []
        rom_critical_heads = []
        fom_pipe_lengths = []
        rom_pipe_lengths = []

        for mu in mu_list:
            fom_entry = self._FindCaseEntry(
                stage="fom",
                case_tag=case_tag,
                mu=mu,
            )
            rom_entry = self._FindCaseEntry(
                stage="rom",
                case_tag=case_tag,
                mu=mu,
                expected_basis_token=self._active_basis_token,
                expected_basis_version=self._active_basis_version,
            )

            if fom_entry is None or rom_entry is None:
                raise RuntimeError(
                    f"Missing or incompatible FOM/ROM QoI for mu={mu}, case_tag='{case_tag}'."
                )

            qoi_fom = self._ReadJson(Path(fom_entry["qoi_file"]))
            qoi_rom = self._ReadJson(Path(rom_entry["qoi_file"]))

            fom_data = qoi_fom["final_data"]
            rom_data = qoi_rom["final_data"]

            fom_critical_head = float(fom_data["critical_head"])
            rom_critical_head = float(rom_data["critical_head"])
            fom_pipe_length = float(fom_data["pipe_length"])
            rom_pipe_length = float(rom_data["pipe_length"])
            fom_residual_norm = float(fom_data["residual_norm"])
            rom_residual_norm = float(rom_data["residual_norm"])

            rel_head = self._RelativeScalarError(fom_critical_head, rom_critical_head)
            rel_pipe = self._RelativeScalarError(fom_pipe_length, rom_pipe_length)

            per_case.append(
                {
                    "mu": mu,
                    "case_id_fom": fom_entry.get("case_id"),
                    "case_id_rom": rom_entry.get("case_id"),
                    "relative_l2_error_critical_head": rel_head,
                    "relative_l2_error_critical_head_percent": None if rel_head is None else 100.0 * rel_head,
                    "relative_l2_error_pipe_length": rel_pipe,
                    "relative_l2_error_pipe_length_percent": None if rel_pipe is None else 100.0 * rel_pipe,
                    "critical_head_fom": fom_critical_head,
                    "critical_head_rom": rom_critical_head,
                    "pipe_length_fom": fom_pipe_length,
                    "pipe_length_rom": rom_pipe_length,
                    "residual_norm_fom": fom_residual_norm,
                    "residual_norm_rom": rom_residual_norm,
                    "qoi_fom_file": fom_entry["qoi_file"],
                    "qoi_rom_file": rom_entry["qoi_file"],
                }
            )

            fom_critical_heads.append(fom_critical_head)
            rom_critical_heads.append(rom_critical_head)
            fom_pipe_lengths.append(fom_pipe_length)
            rom_pipe_lengths.append(rom_pipe_length)

        global_rel_head = self._RelativeErrorFinite(fom_critical_heads, rom_critical_heads)
        global_rel_pipe = self._RelativeErrorFinite(fom_pipe_lengths, rom_pipe_lengths)

        summary = {
            "stage": stage_name,
            "case_tag": case_tag,
            "mu_list": mu_list,
            "basis_mu_train": basis_mu_train,
            "basis_token": self._active_basis_token,
            "basis_version": self._active_basis_version,
            "global_relative_l2_error": global_rel_head,
            "global_relative_l2_error_percent": None if global_rel_head is None else 100.0 * global_rel_head,
            "global_relative_l2_error_critical_head": global_rel_head,
            "global_relative_l2_error_critical_head_percent": None if global_rel_head is None else 100.0 * global_rel_head,
            "global_relative_l2_error_pipe_length": global_rel_pipe,
            "global_relative_l2_error_pipe_length_percent": None if global_rel_pipe is None else 100.0 * global_rel_pipe,
            "per_case": per_case,
            "fom_stage_summary": fom_stage_summary,
            "rom_stage_summary": rom_stage_summary,
            "error_definition": "relative_l2 = ||reference - approximation|| / ||reference|| on QoI vectors across mu cases",
        }

        self._WriteReport(report_name, summary)
        return summary

    def _RunPipingCases(
        self,
        simulation_type,
        mu_list,
        case_tag,
        min_max_step,
        run_missing_only,
        force_recompute,
        results_name,
    ):
        parameters = self._ReadProjectParameters()
        head_samples = _build_head_samples(min_max_step)
        stage = self._NormalizeSnapshotStage(simulation_type)

        if stage == "rom":
            self._SetRomFlagsForLinear()

        case_results = []
        for idx, mu in enumerate(mu_list):
            case_parameters = self._PrepareCaseParameters(
                project_parameters=parameters,
                mu=mu,
                results_name=results_name,
                case_idx=idx,
            )

            signature_payload = self._BuildCaseSignaturePayload(
                stage=stage,
                case_tag=case_tag,
                mu=mu,
                min_max_step=min_max_step,
                case_parameters=case_parameters,
            )
            case_id = self._CaseIdFromSignaturePayload(signature_payload)
            signature_digest = self._PayloadDigest(signature_payload, short=False)

            if run_missing_only and not force_recompute:
                cached_entry = self._FindCaseEntry(
                    stage=stage,
                    case_tag=case_tag,
                    mu=mu,
                    expected_basis_token=self._active_basis_token if stage == "rom" else None,
                    expected_basis_version=self._active_basis_version if stage == "rom" else None,
                    expected_signature_digest=signature_digest,
                )
                if cached_entry is not None:
                    case_results.append(
                        {
                            "mu": mu,
                            "case_id": cached_entry.get("case_id"),
                            "status": "loaded_existing",
                            "file": cached_entry["snapshot_file"],
                            "qoi_file": cached_entry["qoi_file"],
                        }
                    )
                    continue

            snapshot_path, qoi_path, case_meta_path = self._GetCasePathsById(
                stage=stage,
                case_tag=case_tag,
                case_id=case_id,
            )

            exists = snapshot_path.exists() and qoi_path.exists() and case_meta_path.exists()
            compatible = True
            if stage == "rom" and exists:
                compatible = self._IsRomSnapshotCompatible(
                    snapshot_path,
                    expected_basis_token=self._active_basis_token,
                    expected_basis_version=self._active_basis_version,
                ) and self._IsRomQoiCompatible(
                    qoi_path,
                    expected_basis_token=self._active_basis_token,
                    expected_basis_version=self._active_basis_version,
                )

            if exists and run_missing_only and not force_recompute and compatible:
                case_meta_payload = self._ReadJson(case_meta_path)
                self._RegisterCaseEntry(
                    case_id=case_id,
                    stage=stage,
                    case_tag=case_tag,
                    mu=mu,
                    snapshot_path=snapshot_path,
                    qoi_path=qoi_path,
                    case_meta_path=case_meta_path,
                    created_at=case_meta_payload.get("created_at", datetime.now().isoformat()),
                    basis_token=self._active_basis_token if stage == "rom" else None,
                    basis_version=self._active_basis_version if stage == "rom" else None,
                    signature_digest=signature_digest,
                )
                case_results.append(
                    {
                        "mu": mu,
                        "case_id": case_id,
                        "status": "loaded_existing",
                        "file": str(snapshot_path),
                    }
                )
                continue

            simulation = PipingAnalysis(
                simulation_type=simulation_type,
                base_parameters=case_parameters,
                get_analysis_stage_class=self._GetAnalysisStageClass,
                customize_simulation=self.CustomizeSimulation,
                head_samples=head_samples,
            )
            simulation.Run()

            snapshots = simulation.GetSnapshotsMatrix()
            final_data = simulation.GetFinalData()

            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(snapshot_path, snapshots)

            if stage == "rom":
                self._WriteRomSnapshotMetadata(
                    snapshot_path,
                    {
                        "created_at": datetime.now().isoformat(),
                        "mu": mu,
                        "projection_strategy": self.general_rom_manager_parameters[
                            "projection_strategy"
                        ].GetString(),
                        "basis_mu_train": self._active_basis_mu_train,
                        "basis_token": self._active_basis_token,
                        "basis_version": self._active_basis_version,
                        "shape": list(snapshots.shape),
                    },
                )

            qoi_payload = {
                "created_at": datetime.now().isoformat(),
                "mu": mu,
                "simulation_type": stage,
                "case_tag": case_tag,
                "case_id": case_id,
                "projection_strategy": (
                    self.general_rom_manager_parameters["projection_strategy"].GetString()
                    if stage == "rom"
                    else None
                ),
                "basis_token": self._active_basis_token if stage == "rom" else None,
                "basis_version": self._active_basis_version if stage == "rom" else None,
                "final_data": self._SerializeFinalData(final_data),
            }
            self._WriteJson(qoi_path, qoi_payload)

            created_at = datetime.now().isoformat()
            case_meta_payload = {
                "schema_version": 2,
                "created_at": created_at,
                "case_id": case_id,
                "stage": stage,
                "case_tag": case_tag,
                "mu": [float(value) for value in mu],
                "snapshot_file": str(snapshot_path),
                "qoi_file": str(qoi_path),
                "basis_token": self._active_basis_token if stage == "rom" else None,
                "basis_version": self._active_basis_version if stage == "rom" else None,
                "signature_digest": signature_digest,
                "signature_payload": signature_payload,
            }
            self._WriteJson(case_meta_path, case_meta_payload)

            self._RegisterCaseEntry(
                case_id=case_id,
                stage=stage,
                case_tag=case_tag,
                mu=mu,
                snapshot_path=snapshot_path,
                qoi_path=qoi_path,
                case_meta_path=case_meta_path,
                created_at=created_at,
                basis_token=self._active_basis_token if stage == "rom" else None,
                basis_version=self._active_basis_version if stage == "rom" else None,
                signature_digest=signature_digest,
            )

            if exists and not compatible and not force_recompute:
                status = "recomputed_incompatible_basis"
            else:
                status = "recomputed" if exists else "computed_new"

            case_results.append(
                {
                    "mu": mu,
                    "case_id": case_id,
                    "status": status,
                    "file": str(snapshot_path),
                    "qoi_file": str(qoi_path),
                    "shape": list(snapshots.shape),
                }
            )

        return {
            "simulation_type": stage,
            "case_tag": case_tag,
            "run_missing_only": run_missing_only,
            "force_recompute": force_recompute,
            "cases": case_results,
        }

    # ---------------------------------------------------------------------
    # Basis helpers
    # ---------------------------------------------------------------------

    def _EnsureBasisLoaded(self, mu_train):
        basis_path, sigma_path, meta_path = self._GetBasisFiles(mu_train)
        if not basis_path.exists() or not sigma_path.exists():
            raise RuntimeError("POD basis files not found. Run stage2_build_pod_basis first.")

        right_basis = np.load(basis_path)
        singular_values = np.load(sigma_path)

        basis_version = None
        if meta_path.exists():
            meta = self._ReadJson(meta_path)
            basis_version = meta.get("basis_version", meta.get("created_at"))
        if basis_version is None:
            basis_version = datetime.fromtimestamp(basis_path.stat().st_mtime).isoformat()

        self._ActivateBasis(
            mu_train,
            right_basis,
            singular_values,
            basis_version,
            print_to_rom_parameters=True,
        )
        return right_basis, singular_values

    def _ActivateBasis(self, mu_train, right_basis, singular_values, basis_version, print_to_rom_parameters):
        basis_token = self._BasisToken(mu_train)

        basis_changed = (
            basis_token != self._active_basis_token
            or basis_version != self._active_basis_version
        )

        if print_to_rom_parameters and basis_changed:
            self._PrintBasisToRomParameters(right_basis, singular_values)

        self._active_basis_token = basis_token
        self._active_basis_version = basis_version
        self._active_basis_mu_train = [list(mu) for mu in mu_train]

    def _PrintBasisToRomParameters(self, right_basis, singular_values):
        basis_output_process = self.InitializeDummySimulationForBasisOutputProcess()
        basis_output_process._PrintRomBasis(right_basis, singular_values)

    def _StackSnapshots(self, mu_list, stage, case_tag):
        blocks = []
        for mu in mu_list:
            snapshots = self._LoadSingleSnapshot(stage=stage, case_tag=case_tag, mu=mu)
            if snapshots is None:
                raise RuntimeError(
                    f"Missing snapshots for stage='{stage}', case_tag='{case_tag}', mu={mu}."
                )
            blocks.append(snapshots)
        return np.hstack(blocks)

    def _LoadSingleSnapshot(
        self,
        stage,
        case_tag,
        mu,
        expected_basis_token=None,
        expected_basis_version=None,
    ):
        entry = self._FindCaseEntry(
            stage=stage,
            case_tag=case_tag,
            mu=mu,
            expected_basis_token=expected_basis_token,
            expected_basis_version=expected_basis_version,
        )
        if entry is None:
            return None

        snapshot_path = Path(entry["snapshot_file"])
        if not snapshot_path.exists():
            return None

        if stage == "rom" and (expected_basis_token is not None or expected_basis_version is not None):
            if not self._IsRomSnapshotCompatible(
                snapshot_path,
                expected_basis_token=expected_basis_token,
                expected_basis_version=expected_basis_version,
            ):
                return None

        return np.load(snapshot_path)

    # ---------------------------------------------------------------------
    # QoI helpers
    # ---------------------------------------------------------------------

    def _SerializeFinalData(self, final_data):
        serialized = {}
        for key, value in final_data.items():
            arr = np.asarray(value)
            if arr.size == 1:
                serialized[key] = float(arr.reshape(-1)[0])
            else:
                serialized[key] = arr.tolist()
        return serialized

    def _LoadQoiPayload(
        self,
        stage,
        case_tag,
        mu,
        expected_basis_token=None,
        expected_basis_version=None,
    ):
        entry = self._FindCaseEntry(
            stage=stage,
            case_tag=case_tag,
            mu=mu,
            expected_basis_token=expected_basis_token,
            expected_basis_version=expected_basis_version,
        )
        if entry is None:
            return None

        qoi_path = Path(entry["qoi_file"])
        if not qoi_path.exists():
            return None

        payload = self._ReadJson(qoi_path)
        if stage == "rom" and (expected_basis_token is not None or expected_basis_version is not None):
            if not self._IsRomQoiCompatible(
                qoi_path,
                expected_basis_token=expected_basis_token,
                expected_basis_version=expected_basis_version,
            ):
                return None

        return payload

    def _IsRomQoiCompatible(self, qoi_path, expected_basis_token=None, expected_basis_version=None):
        payload = self._ReadJson(qoi_path)
        if expected_basis_token is not None and payload.get("basis_token") != expected_basis_token:
            return False
        if expected_basis_version is not None and payload.get("basis_version") != expected_basis_version:
            return False
        return True

    # ---------------------------------------------------------------------
    # File naming and metadata
    # ---------------------------------------------------------------------

    def _NormalizeSnapshotStage(self, simulation_type):
        if simulation_type in ["FOM", "fom"]:
            return "fom"
        if simulation_type in ["ROM", "rom"]:
            return "rom"
        raise RuntimeError(f"Unknown simulation_type/stage: {simulation_type}")

    def _HashFileContent(self, file_path):
        path = Path(file_path)
        if not path.is_file():
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _ParametersToDict(self, parameters):
        return json.loads(parameters.PrettyPrintJsonString())

    def _CanonicalizeForHash(self, obj):
        if isinstance(obj, dict):
            return {k: self._CanonicalizeForHash(v) for k, v in sorted(obj.items())}
        if isinstance(obj, (list, tuple)):
            return [self._CanonicalizeForHash(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return [self._CanonicalizeForHash(v) for v in obj.tolist()]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            val = float(obj)
            if np.isnan(val):
                return "NaN"
            if np.isposinf(val):
                return "Inf"
            if np.isneginf(val):
                return "-Inf"
            return float(f"{val:.16g}")
        return obj

    def _PayloadDigest(self, payload, short=True):
        canonical_payload = self._CanonicalizeForHash(payload)
        raw = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        if short:
            return digest[:16]
        return digest

    def _BuildCaseSignaturePayload(self, stage, case_tag, mu, min_max_step, case_parameters):
        material_file_name = case_parameters["solver_settings"]["material_import_settings"][
            "materials_filename"
        ].GetString()
        rom_error_indicator_tolerance = None
        if self.general_rom_manager_parameters.Has("rom_error_indicator_tolerance"):
            rom_error_indicator_tolerance = self.general_rom_manager_parameters[
                "rom_error_indicator_tolerance"
            ].GetDouble()

        return {
            "storage_schema_version": 2,
            "stage": stage,
            "case_tag": case_tag,
            "mu": [float(value) for value in mu],
            "head_sweep": [float(value) for value in min_max_step],
            "project_parameters_signature": self._project_parameters_signature,
            "general_rom_signature": self._general_rom_signature,
            "material_file": material_file_name,
            "material_signature": self._HashFileContent(material_file_name),
            "case_parameters_signature": self._PayloadDigest(
                self._ParametersToDict(case_parameters),
                short=False,
            ),
            "projection_strategy": self.general_rom_manager_parameters["projection_strategy"].GetString(),
            "type_of_decoder": self.general_rom_manager_parameters["type_of_decoder"].GetString(),
            "assembling_strategy": self.general_rom_manager_parameters["assembling_strategy"].GetString(),
            "rom_error_indicator_tolerance": rom_error_indicator_tolerance if stage == "rom" else None,
            "basis_token": self._active_basis_token if stage == "rom" else None,
            "basis_version": self._active_basis_version if stage == "rom" else None,
        }

    def _CaseIdFromSignaturePayload(self, signature_payload):
        return self._PayloadDigest(signature_payload, short=True)

    def _GetCasePathsById(self, stage, case_tag, case_id):
        if stage not in ("fom", "rom"):
            raise RuntimeError(f"Unknown snapshot stage: {stage}")

        root = self.fom_root if stage == "fom" else self.rom_root
        snapshot_path = root / case_tag / f"{stage}_c_{case_id}.npy"
        qoi_path = self.qoi_root / stage / case_tag / f"qoi_c_{case_id}.json"
        case_meta_path = self.case_meta_root / f"case_{case_id}.json"
        return snapshot_path, qoi_path, case_meta_path

    def _LoadCaseIndex(self):
        if not self.case_index_file.exists():
            return {"schema_version": 1, "cases": {}, "last_updated": None}
        return self._ReadJson(self.case_index_file)

    def _SaveCaseIndex(self, index_payload):
        self._WriteJson(self.case_index_file, index_payload)

    def _RegisterCaseEntry(
        self,
        case_id,
        stage,
        case_tag,
        mu,
        snapshot_path,
        qoi_path,
        case_meta_path,
        created_at,
        basis_token,
        basis_version,
        signature_digest,
    ):
        index_payload = self._LoadCaseIndex()
        index_payload.setdefault("cases", {})

        index_payload["cases"][case_id] = {
            "case_id": case_id,
            "stage": stage,
            "case_tag": case_tag,
            "mu": [float(value) for value in mu],
            "snapshot_file": str(snapshot_path),
            "qoi_file": str(qoi_path),
            "case_meta_file": str(case_meta_path),
            "basis_token": basis_token,
            "basis_version": basis_version,
            "signature_digest": signature_digest,
            "created_at": created_at,
        }
        index_payload["last_updated"] = datetime.now().isoformat()
        self._SaveCaseIndex(index_payload)

    def _MuMatches(self, mu_a, mu_b):
        if mu_a is None or mu_b is None:
            return False
        if len(mu_a) != len(mu_b):
            return False

        for a, b in zip(mu_a, mu_b):
            af = float(a)
            bf = float(b)
            tol = 1e-14 * max(1.0, abs(af), abs(bf))
            if abs(af - bf) > tol:
                return False
        return True

    def _FindCaseEntry(
        self,
        stage,
        case_tag,
        mu,
        expected_basis_token=None,
        expected_basis_version=None,
        expected_signature_digest=None,
    ):
        index_payload = self._LoadCaseIndex()
        cases = index_payload.get("cases", {})

        mu_values = [float(value) for value in mu]
        candidates = []
        for entry in cases.values():
            if entry.get("stage") != stage:
                continue
            if entry.get("case_tag") != case_tag:
                continue
            if not self._MuMatches(entry.get("mu"), mu_values):
                continue
            if expected_basis_token is not None and entry.get("basis_token") != expected_basis_token:
                continue
            if expected_basis_version is not None and entry.get("basis_version") != expected_basis_version:
                continue
            if expected_signature_digest is not None and entry.get("signature_digest") != expected_signature_digest:
                continue

            snapshot_path = Path(entry["snapshot_file"])
            qoi_path = Path(entry["qoi_file"])
            if snapshot_path.exists() and qoi_path.exists():
                candidates.append(entry)

        if candidates:
            candidates.sort(key=lambda item: item.get("created_at", ""))
            return candidates[-1]

        # Backward compatibility with old mu-based file names.
        legacy_snapshot = self._GetLegacySnapshotPath(stage, case_tag, mu)
        legacy_qoi = self._GetLegacyQoiPath(stage, case_tag, mu)
        if legacy_snapshot.exists() and legacy_qoi.exists():
            if expected_signature_digest is not None:
                return None
            if stage == "rom":
                if not self._IsRomSnapshotCompatible(
                    legacy_snapshot,
                    expected_basis_token=expected_basis_token,
                    expected_basis_version=expected_basis_version,
                ):
                    return None
                if not self._IsRomQoiCompatible(
                    legacy_qoi,
                    expected_basis_token=expected_basis_token,
                    expected_basis_version=expected_basis_version,
                ):
                    return None

            return {
                "case_id": None,
                "stage": stage,
                "case_tag": case_tag,
                "mu": mu_values,
                "snapshot_file": str(legacy_snapshot),
                "qoi_file": str(legacy_qoi),
                "case_meta_file": None,
                "basis_token": expected_basis_token,
                "basis_version": expected_basis_version,
                "signature_digest": None,
                "created_at": datetime.fromtimestamp(legacy_snapshot.stat().st_mtime).isoformat(),
            }

        return None

    def _GetLegacySnapshotPath(self, stage, case_tag, mu):
        root = self.fom_root if stage == "fom" else self.rom_root
        case_dir = root / case_tag

        if stage == "fom":
            file_name = f"fom_{self._MuToken(mu)}.npy"
        else:
            proj = self.general_rom_manager_parameters["projection_strategy"].GetString()
            file_name = f"rom_{proj}_{self._MuToken(mu)}.npy"

        return case_dir / file_name

    def _GetLegacyQoiPath(self, stage, case_tag, mu):
        case_dir = self.qoi_root / stage / case_tag
        return case_dir / f"qoi_{self._MuToken(mu)}.json"

    def _GetBasisFiles(self, mu_train):
        token = self._MuListToken(mu_train)
        basis_path = self.pod_root / f"right_basis_train_{token}.npy"
        sigma_path = self.pod_root / f"singular_values_train_{token}.npy"
        meta_path = self.pod_root / f"basis_meta_train_{token}.json"
        return basis_path, sigma_path, meta_path

    def _BasisToken(self, mu_train):
        return self._MuListToken(mu_train)

    def _GetRomSnapshotMetaPath(self, rom_snapshot_path):
        return rom_snapshot_path.with_name(f"{rom_snapshot_path.name}.meta.json")

    def _WriteRomSnapshotMetadata(self, rom_snapshot_path, payload):
        self._WriteJson(self._GetRomSnapshotMetaPath(rom_snapshot_path), payload)

    def _ReadRomSnapshotMetadata(self, rom_snapshot_path):
        meta_path = self._GetRomSnapshotMetaPath(rom_snapshot_path)
        if not meta_path.exists():
            return None
        return self._ReadJson(meta_path)

    def _IsRomSnapshotCompatible(self, rom_snapshot_path, expected_basis_token=None, expected_basis_version=None):
        meta = self._ReadRomSnapshotMetadata(rom_snapshot_path)
        if meta is None:
            return False
        if expected_basis_token is not None and meta.get("basis_token") != expected_basis_token:
            return False
        if expected_basis_version is not None and meta.get("basis_version") != expected_basis_version:
            return False
        return True

    def _FloatToken(self, value):
        txt = f"{float(value):.12g}"
        txt = txt.replace("-", "m").replace("+", "").replace(".", "p")
        return txt

    def _MuToken(self, mu):
        values = list(mu)
        if self.mu_names and len(values) == len(self.mu_names):
            return "__".join(f"{name}_{self._FloatToken(value)}" for name, value in zip(self.mu_names, values))
        return "__".join(f"mu{i + 1}_{self._FloatToken(value)}" for i, value in enumerate(values))

    def _MuListToken(self, mu_list):
        """Short deterministic token for a parameter list (avoids long file names)."""
        normalized = [[float(value) for value in mu] for mu in mu_list]
        payload = json.dumps(normalized, separators=(",", ":"), ensure_ascii=True)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        return f"n{len(normalized)}_{digest}"

    # ---------------------------------------------------------------------
    # Kratos integration helpers
    # ---------------------------------------------------------------------

    def _ReadProjectParameters(self):
        with open(self.project_parameters_name, "r", encoding="utf-8") as parameter_file:
            return KratosMultiphysics.Parameters(parameter_file.read())

    def _PrepareCaseParameters(self, project_parameters, mu, results_name, case_idx):
        parameters = self.UpdateProjectParameters(project_parameters.Clone(), mu)
        parameters = self._ApplyKratosDeltaresCompatibilityFixes(parameters)
        parameters = self._AddBasisCreationToProjectParameters(parameters)
        parameters = self._StoreResultsByName(parameters, results_name, mu, case_idx)

        materials_file_name = parameters["solver_settings"]["material_import_settings"][
            "materials_filename"
        ].GetString()
        self.UpdateMaterialParametersFile(materials_file_name, mu)

        return parameters

    def InitializeDummySimulationForBasisOutputProcess(self):
        parameters = self._ReadProjectParameters()
        parameters = self._ApplyKratosDeltaresCompatibilityFixes(parameters)
        parameters = self._AddBasisCreationToProjectParameters(parameters)
        parameters = self._StoreNoResults(parameters)

        model = KratosMultiphysics.Model()
        analysis_stage_class = self._GetAnalysisStageClass(parameters)
        simulation = self.CustomizeSimulation(analysis_stage_class, model, parameters)
        simulation.Initialize()

        for process in simulation._GetListOfOutputProcesses():
            if isinstance(process, CalculateRomBasisOutputProcess):
                return process

        raise RuntimeError(
            "CalculateRomBasisOutputProcess not found in dummy simulation. "
            "Basis creation cannot continue."
        )

    def _GetAnalysisStageClass(self, parameters):
        analysis_stage_module_name = parameters["analysis_stage"].GetString()
        analysis_stage_class_name = analysis_stage_module_name.split(".")[-1]
        analysis_stage_class_name = "".join(x.title() for x in analysis_stage_class_name.split("_"))

        analysis_stage_module = importlib.import_module(analysis_stage_module_name)
        return getattr(analysis_stage_module, analysis_stage_class_name)

    def _AddBasisCreationToProjectParameters(self, parameters):
        if not parameters["output_processes"].Has("rom_output"):
            parameters["output_processes"].AddEmptyArray("rom_output")
        else:
            parameters["output_processes"].RemoveValue("rom_output")
            parameters["output_processes"].AddEmptyArray("rom_output")

        parameters["output_processes"]["rom_output"].Append(self._SetUpRomBasisParameters())
        return parameters

    def _StoreResultsByName(self, parameters, results_name, mu, idx):
        output_name_mode = self.general_rom_manager_parameters["output_name"].GetString()
        case_name = ", ".join(map(str, mu)) if output_name_mode == "mu" else str(idx)

        if self.general_rom_manager_parameters["save_gid_output"].GetBool():
            if parameters["output_processes"].Has("gid_output"):
                parameters["output_processes"]["gid_output"][0]["Parameters"]["output_name"].SetString(
                    "Results/" + results_name + "_" + case_name
                )
        else:
            if parameters["output_processes"].Has("gid_output"):
                parameters["output_processes"].RemoveValue("gid_output")

        if self.general_rom_manager_parameters["save_vtk_output"].GetBool():
            if parameters["output_processes"].Has("vtk_output"):
                parameters["output_processes"]["vtk_output"][0]["Parameters"]["output_path"].SetString(
                    "Results/vtk_output_" + results_name + "_" + case_name
                )
        else:
            if parameters["output_processes"].Has("vtk_output"):
                parameters["output_processes"].RemoveValue("vtk_output")

        return parameters

    def _StoreNoResults(self, parameters):
        if parameters["output_processes"].Has("gid_output"):
            parameters["output_processes"].RemoveValue("gid_output")
        if parameters["output_processes"].Has("vtk_output"):
            parameters["output_processes"].RemoveValue("vtk_output")
        return parameters

    def _SetRomFlagsForLinear(self):
        strategy = self.general_rom_manager_parameters["projection_strategy"].GetString()
        if strategy != "galerkin":
            raise RuntimeError(
                "Staged piping manager currently supports only Galerkin projection. "
                f"Received '{strategy}'."
            )

        parameters_file_folder = self.general_rom_manager_parameters["ROM"]["rom_basis_output_folder"].GetString()
        parameters_file_name = self.general_rom_manager_parameters["ROM"]["rom_basis_output_name"].GetString()
        parameters_file_path = Path(parameters_file_folder) / Path(parameters_file_name).with_suffix(".json")

        if not parameters_file_path.exists():
            raise RuntimeError(f"RomParameters file not found: {parameters_file_path}")

        with parameters_file_path.open("r+", encoding="utf-8") as parameter_file:
            data = json.load(parameter_file)
            data.setdefault("rom_settings", {})
            data["projection_strategy"] = "galerkin"
            data["assembling_strategy"] = self.general_rom_manager_parameters[
                "assembling_strategy"
            ].GetString()
            data["train_hrom"] = False
            data["run_hrom"] = False
            data["rom_settings"]["rom_bns_settings"] = self._SetGalerkinBnSParameters()

            parameter_file.seek(0)
            json.dump(data, parameter_file, indent=4)
            parameter_file.truncate()

    def _SetGalerkinBnSParameters(self):
        defaults = {"monotonicity_preserving": False}
        if self.general_rom_manager_parameters["ROM"].Has("galerkin_rom_bns_settings"):
            rom_params = self.general_rom_manager_parameters["ROM"]["galerkin_rom_bns_settings"]
            if rom_params.Has("monotonicity_preserving"):
                defaults["monotonicity_preserving"] = rom_params[
                    "monotonicity_preserving"
                ].GetBool()
        return defaults

    def _SetUpRomBasisParameters(self):
        defaults = self._GetDefaultRomBasisOutputParameters()
        defaults["Parameters"]["rom_manager"].SetBool(True)

        rom_params = self.general_rom_manager_parameters["ROM"]
        keys_to_copy = [
            "svd_truncation_tolerance",
            "model_part_name",
            "rom_basis_output_format",
            "rom_basis_output_name",
            "rom_basis_output_folder",
            "nodal_unknowns",
            "snapshots_interval",
            "print_singular_values",
        ]

        for key in keys_to_copy:
            if key in rom_params.keys():
                defaults["Parameters"][key] = rom_params[key]

        return defaults

    def _GetDefaultRomBasisOutputParameters(self):
        return KratosMultiphysics.Parameters(
            """{
            "python_module" : "calculate_rom_basis_output_process",
            "kratos_module" : "KratosMultiphysics.RomApplication",
            "process_name"  : "CalculateRomBasisOutputProcess",
            "help"          : "Writes ROM basis from snapshots",
            "Parameters"    : {
                "model_part_name": "",
                "rom_manager" : false,
                "snapshots_control_type": "step",
                "snapshots_interval": 1.0,
                "nodal_unknowns": [],
                "rom_basis_output_format": "json",
                "rom_basis_output_name": "RomParameters",
                "rom_basis_output_folder": "rom_data",
                "svd_truncation_tolerance": 1e-3,
                "print_singular_values": false
            }
        }"""
        )

    # ---------------------------------------------------------------------
    # Validation/setup/defaults
    # ---------------------------------------------------------------------

    def _CheckLinearOnly(self):
        decoder_type = self.general_rom_manager_parameters["type_of_decoder"].GetString()
        if decoder_type != "linear":
            raise RuntimeError(
                "Staged piping manager supports only linear decoder. "
                f"Current type_of_decoder='{decoder_type}'."
            )

    def _CheckSupportedProjection(self):
        projection_strategy = self.general_rom_manager_parameters["projection_strategy"].GetString()
        if projection_strategy != "galerkin":
            raise RuntimeError(
                "Staged piping manager supports only Galerkin projection. "
                f"Current projection_strategy='{projection_strategy}'."
            )

    def _SetUpRomManagerParameters(self, input_parameters):
        if input_parameters is None:
            input_parameters = KratosMultiphysics.Parameters()

        self.general_rom_manager_parameters = input_parameters

        default_settings = KratosMultiphysics.Parameters(
            """{
            "rom_stages_to_train" : ["ROM"],
            "rom_stages_to_test" : ["ROM"],
            "paralellism" : null,
            "projection_strategy": "galerkin",
            "type_of_decoder" : "linear",
            "assembling_strategy": "global",
            "save_gid_output": false,
            "save_vtk_output": false,
            "output_name": "id",
            "rom_error_indicator_tolerance": 1e-5,
            "ROM":{
                "svd_truncation_tolerance": 1e-5,
                "model_part_name": "PorousDomain",
                "nodal_unknowns": ["WATER_PRESSURE"],
                "rom_basis_output_format": "numpy",
                "rom_basis_output_name": "RomParameters",
                "rom_basis_output_folder": "rom_data",
                "snapshots_control_type": "step",
                "snapshots_interval": 1,
                "print_singular_values": false,
                "galerkin_rom_bns_settings": {
                    "monotonicity_preserving": false
                }
            }
        }"""
        )

        self.general_rom_manager_parameters.RecursivelyValidateAndAssignDefaults(default_settings)

    def _ApplyKratosDeltaresCompatibilityFixes(self, parameters):
        if parameters.Has("solver_settings"):
            solver_settings = parameters["solver_settings"]
            if solver_settings.Has("nodal_smoothing"):
                solver_settings.RemoveValue("nodal_smoothing")
        return parameters

    def DefaultUpdateProjectParameters(self, parameters, mu=None):
        return self._ApplyKratosDeltaresCompatibilityFixes(parameters)

    def DefaultUpdateMaterialParametersFile(self, material_parameters_file_name=None, mu=None):
        pass

    def DefaultCustomizeSimulation(self, cls, global_model, parameters, mu=None):
        class DefaultCustomSimulation(cls):
            def __init__(self, model, project_parameters):
                super().__init__(model, project_parameters)

        return DefaultCustomSimulation(global_model, parameters)

    # ---------------------------------------------------------------------
    # Small utilities
    # ---------------------------------------------------------------------

    def _RelativeError(self, reference, approximation):
        ref_norm = np.linalg.norm(reference)
        if ref_norm <= 0.0:
            return None
        return float(np.linalg.norm(reference - approximation) / ref_norm)

    def _RelativeErrorFinite(self, reference, approximation):
        ref = np.asarray(reference, dtype=float).reshape(-1)
        approx = np.asarray(approximation, dtype=float).reshape(-1)

        if ref.shape != approx.shape:
            raise RuntimeError(
                f"Cannot compare arrays with different shapes: {ref.shape} vs {approx.shape}."
            )

        finite_mask = np.isfinite(ref) & np.isfinite(approx)
        if not np.any(finite_mask):
            return None

        return self._RelativeError(ref[finite_mask], approx[finite_mask])

    def _RelativeScalarError(self, reference, approximation):
        ref = float(reference)
        approx = float(approximation)

        if not np.isfinite(ref) or not np.isfinite(approx):
            return None

        ref_abs = abs(ref)
        if ref_abs <= 0.0:
            return None

        return float(abs(ref - approx) / ref_abs)

    def _WriteJson(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file_pointer:
            json.dump(payload, file_pointer, indent=4)

    def _ReadJson(self, path):
        with path.open("r", encoding="utf-8") as file_pointer:
            return json.load(file_pointer)

    def _WriteReport(self, name, payload):
        self._WriteJson(self.reports_root / name, payload)
