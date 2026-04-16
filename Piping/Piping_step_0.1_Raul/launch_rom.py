import KratosMultiphysics
from custom_rom_manager import RomManager
import json
import numpy as np
from pathlib import Path

import time
from matplotlib import pyplot as plt

FIGURES_DIR = Path("figures")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def CustomizeSimulation(cls, global_model, parameters, type_of_simulation='FOM'):

    class CustomSimulation(cls):

        def __init__(self, model,project_parameters, type_of_simulation = type_of_simulation):
            super().__init__(model,project_parameters)
            self.type_of_simulation  = type_of_simulation
            self.ErroIndicator = True
            self.residual_norms = []
            """
            Customize as needed
            """

        def ModifyInitialGeometry(self):
            super().ModifyInitialGeometry()
            """
            Customize as needed
            """

        def InitializeSolutionStep(self):
            super().InitializeSolutionStep()
            """
            Customize as needed
            """


        def CustomMethod(self):
            """
            Customize as needed
            """
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
            if self.type_of_simulation=='ROM':
                if self.ResidualNorm > 1e-5: #TODO how to setup a robut threshold?
                    self.ErroIndicator = False

    return CustomSimulation(global_model, parameters, type_of_simulation)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #




#taken from the GeoMechanics app tests folder
def get_pipe_length(simulation):
    """
    Gets the length of all active pipe elemnets
    :param simulation:
    :return: pipe_length :
    """
    model_part = simulation._list_of_output_processes[0].model_part
    elements = model_part.Elements
    return sum([element.GetValue(KratosMultiphysics.GeoMechanicsApplication.PIPE_ELEMENT_LENGTH) for element in elements if element.GetValue(KratosMultiphysics.GeoMechanicsApplication.PIPE_ACTIVE)])




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def UpdateMaterialParametersFile(materials_file_name, mu=None):
    """
    Customize ProjectParameters here for imposing different conditions to the simulations as needed
   """
    premeability__xx = mu[0]
    d70 = mu[1]
    with open(materials_file_name, 'r') as parameter_file:
        parameters = json.load(parameter_file)
        parameters['properties'][0]['Material']['Variables']['PERMEABILITY_XX'] = premeability__xx
        parameters['properties'][1]['Material']['Variables']['PERMEABILITY_XX'] = premeability__xx
        parameters['properties'][3]['Material']['Variables']['PIPE_D_70'] = d70
    with open(materials_file_name, 'w') as parameter_file:
        json.dump(parameters, parameter_file, indent=4)



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #




def GetRomManagerParameters():
    """
    This function allows to easily modify all the parameters for the ROM simulation.
    The returned KratosParameter object is seamlessly used inside the RomManager.
    """
    general_rom_manager_parameters = KratosMultiphysics.Parameters("""{
            "rom_stages_to_train" : ["ROM"],             // ["ROM","HROM"]
            "rom_stages_to_test" : ["ROM"],                   //  ["ROM","HROM"]
            "paralellism" : null,                        // null, TODO: add "compss"
            "projection_strategy": "galerkin",           // "lspg", "galerkin", "petrov_galerkin"
            "save_gid_output": false,                     // false, true #if true, it must exits previously in the ProjectParameters.json
            "save_vtk_output": false,                    // false, true #if true, it must exits previously in the ProjectParameters.json
            "output_name": "id",                         // "id" , "mu"
            "assembling_strategy": "global",
            "ROM":{
                "svd_truncation_tolerance": 0,
                "model_part_name": "PorousDomain",              // This changes depending on the simulation: Structure, FluidModelPart, ThermalPart #TODO: Idenfity it automatically
                "nodal_unknowns": ["WATER_PRESSURE"]           // Main unknowns. Snapshots are taken from these
            }
        }""")

    return general_rom_manager_parameters


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def _figure_path(filename):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR / filename


def _relative_l2_error_percent(reference, approximation):
    reference = np.asarray(reference, dtype=float)
    approximation = np.asarray(approximation, dtype=float)
    ref_norm = np.linalg.norm(reference)
    if ref_norm == 0.0:
        return np.nan
    return 100.0 * np.linalg.norm(reference - approximation) / ref_norm





def plot_pipe_length_plus_residual(
    fom,
    rom,
    fom2,
    rom2,
    stage_label="Unknown stage",
    qoi_label="Critical Head",
    output_png_name="error_indicator_rom.png",
):
    fom = np.squeeze(np.asarray(fom, dtype=float))
    rom = np.squeeze(np.asarray(rom, dtype=float))
    fom2 = np.squeeze(np.asarray(fom2, dtype=float))
    rom2 = np.squeeze(np.asarray(rom2, dtype=float))

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
    ax2.plot(case_indices, fom2, "bo-", label="Residual Norm FOM")
    ax2.plot(case_indices, rom2, "ro-", label="Residual Norm ROM")
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



def get_percentual_min_max(val, percentace):
    return val-val*(percentace/100), val + val*(percentace/100)


def get_multiple_params(number_of_total_samples, seed1, seed2):

    permeability_min, permeability_max = get_percentual_min_max(5E-12, 80) #+- 80%
    np.random.seed(seed1)
    row1 = np.random.uniform(permeability_min, permeability_max, size=number_of_total_samples)

    d70_min, d70_max = get_percentual_min_max(0.0001, 50) #+- 50%
    np.random.seed(seed2)
    row2 = np.random.uniform(d70_min, d70_max,  size=number_of_total_samples)

    # Combine the rows vertically to create a column vector
    mu = np.vstack((row1, row2))

    return (mu.T).tolist()



def plot_pipe_length_ROMs(
    fom,
    rom,
    name_1="FOM",
    name_2="ROM",
    stage_label="Unknown stage",
    qoi_label="Pipe Length",
    output_png_name="comparison.png",
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

    # print(np.where(error!=0)[0])



def plot_mu_values(mu_train,mu_test):
    mu_train_a = np.zeros(len(mu_train))
    mu_train_m = np.zeros(len(mu_train))
    mu_test_a  = np.zeros(len(mu_test))
    mu_test_m  = np.zeros(len(mu_test))
    for i in range(len(mu_train)):
        mu_train_a[i] = mu_train[i][0]
        mu_train_m[i] = mu_train[i][1]
    for i in range(len(mu_test)):
        mu_test_a[i] = mu_test[i][0]
        mu_test_m[i] = mu_test[i][1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(mu_train_m, mu_train_a, 'bs', label="Train Values")
    ax.plot(mu_test_m, mu_test_a, 'ro', label="Test Values")
    ax.set_title('Mu Values')
    ax.set_ylabel(r'$d_{70}$')
    ax.set_xlabel(r'Permeability XX')
    ax.grid(True)
    ax.legend(bbox_to_anchor=(.85, 1.03, 1., .102), loc='upper left', borderaxespad=0.)
    fig.tight_layout()
    fig.savefig(_figure_path('sampling.png'), dpi=200)
    plt.show()
    plt.close(fig)


def get_qoi_vector_in_mu_order(rom_manager, mu_list, table_name, qoi_name):
    """
    Retrieve QoI values in the same order as mu_list.
    """
    values = []
    for mu in mu_list:
        value = rom_manager.data_base.get_snapshots_matrix_from_database(
            [mu], table_name=table_name, QoI=qoi_name
        )
        if value is None:
            raise RuntimeError(
                f"Missing QoI '{qoi_name}' in table '{table_name}' for mu={mu}."
            )
        values.append(float(np.squeeze(np.asarray(value))))
    return np.array(values)


if __name__ == "__main__":

    general_rom_manager_parameters = GetRomManagerParameters()
    project_parameters_name = "ProjectParameters.json"
    rom_manager = RomManager(project_parameters_name=project_parameters_name,general_rom_manager_parameters=general_rom_manager_parameters,UpdateMaterialParametersFile=UpdateMaterialParametersFile, CustomizeSimulation=CustomizeSimulation)




    #mu_train = [[1.5337712840261257e-12, 0.0001]]



    mu_train = get_multiple_params(15, 42, 72)
    mu_test = get_multiple_params(6, 44, 74)

    plot_mu_values(mu_train,mu_test)




    rom_manager.FitPiping(mu_train, min_max_step=[3.0,10.0,0.1])

    #mu_train = mu_train[:12]


    QoI_FOM = get_qoi_vector_in_mu_order(rom_manager, mu_train, table_name='QoI_FOM', qoi_name='pipe_length')
    QoI_ROM = get_qoi_vector_in_mu_order(rom_manager, mu_train, table_name='QoI_ROM', qoi_name='pipe_length')
    plot_pipe_length_ROMs(
        QoI_FOM,
        QoI_ROM,
        stage_label="Verification (train set)",
        qoi_label="Pipe Length",
        output_png_name="verification_pipe_length.png",
    )


    QoI_FOM = get_qoi_vector_in_mu_order(rom_manager, mu_train, table_name='QoI_FOM', qoi_name='critical_head')
    QoI_ROM = get_qoi_vector_in_mu_order(rom_manager, mu_train, table_name='QoI_ROM', qoi_name='critical_head')
    plot_pipe_length_ROMs(
        QoI_FOM,
        QoI_ROM,
        stage_label="Verification (train set)",
        qoi_label="Critical Head",
        output_png_name="verification_critical_head.png",
    )


    QoI_FOM = get_qoi_vector_in_mu_order(rom_manager, mu_train, table_name='QoI_FOM', qoi_name='critical_head')
    QoI_ROM = get_qoi_vector_in_mu_order(rom_manager, mu_train, table_name='QoI_ROM', qoi_name='critical_head')
    QoI_FOM_2 = get_qoi_vector_in_mu_order(rom_manager, mu_train, table_name='QoI_FOM', qoi_name='residual_norm')
    QoI_ROM_2 = get_qoi_vector_in_mu_order(rom_manager, mu_train, table_name='QoI_ROM', qoi_name='residual_norm')

    plot_pipe_length_plus_residual(
        QoI_FOM,
        QoI_ROM,
        QoI_FOM_2,
        QoI_ROM_2,
        stage_label="Verification (train set)",
        qoi_label="Critical Head",
        output_png_name="verification_critical_head_with_residual.png",
    )



    rom_manager.TestPiping(mu_test, min_max_step=[3.0,10.0,0.1])


    QoI_FOM = get_qoi_vector_in_mu_order(rom_manager, mu_test, table_name='QoI_FOM', qoi_name='critical_head')
    QoI_ROM = get_qoi_vector_in_mu_order(rom_manager, mu_test, table_name='QoI_ROM', qoi_name='critical_head')
    QoI_FOM_2 = get_qoi_vector_in_mu_order(rom_manager, mu_test, table_name='QoI_FOM', qoi_name='residual_norm')
    QoI_ROM_2 = get_qoi_vector_in_mu_order(rom_manager, mu_test, table_name='QoI_ROM', qoi_name='residual_norm')


    plot_pipe_length_plus_residual(
        QoI_FOM,
        QoI_ROM,
        QoI_FOM_2,
        QoI_ROM_2,
        stage_label="Test (unseen set)",
        qoi_label="Critical Head",
        output_png_name="test_critical_head_with_residual.png",
    )
