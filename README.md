# Deltares ROM Projects

This repository is dedicated to keeping track of problems and simulations solved using the **Kratos Multiphysics `RomApplication`** in conjunction with **Deltares**.

## 🛠 Required Kratos Branch

To run the simulations in this repository (e.g., the Piping cases), you must compile Kratos from the following specialized branch:
[Kratos_Deltares_ROM](https://github.com/KratosMultiphysics/Kratos/tree/Kratos_Deltares_ROM)

### Why is this branch necessary?
The residual error indicator logic used in these projects requires accurate **Residual Norm** calculations. In the standard Kratos `master` repository, the calculation of the Right-Hand Side (RHS) was disabled in certain solvers to improve performance.

As noted by Pooyan in the source code:
> **NOTE:** The following part will be commented because it is time consuming and there is no obvious reason to be here. If someone needs this part please notify the community via mailing list before uncommenting it.

```cpp
{
    // TSparseSpace::SetToZero(mb);
    // p_builder_and_solver->BuildRHS(p_scheme, r_model_part, mb);
}
```

The `Kratos_Deltares_ROM` branch re-enables these calculations, which are currently not part of the master repository but are essential for retrieving the residuals needed for our ROM stability assessments.

## 📂 Project Structure

- **[Piping/Piping_step_0.1](Piping/Piping_step_0.1/)**: Initial implementation of the piping ROM workflow.