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

### 🌊 Piping Projects
- **[Piping/Piping_ROM_JSON](Piping/Piping_ROM_JSON/)**: Legacy implementation using hash-based JSON storage.
- **[Piping/Piping_ROM_Manager_SQL](Piping/Piping_ROM_Manager_SQL/)**: Version utilizing the SQL-based ROM Manager (Raul's modifications).
- **[Piping/Piping_ROM_Staged](Piping/Piping_ROM_Staged/)**: New transparent architecture with stage-based directory storage and human-readable filenames.

### 🏔 C-Phi Projects
- **[C-Phi/Plaxis_CPhi_ROM_Staged](C-Phi/Plaxis_CPhi_ROM_Staged/)**: ROM implementation for Strength Reduction using the Staged architecture.
- **[C-Phi/Plaxis_CPhi_Benchmark](C-Phi/Plaxis_CPhi_Benchmark/)**: Advanced benchmark using a PLAXIS mesh and U_Pw solver.
- **[C-Phi/C-Phi_reduction_process](C-Phi/C-Phi_reduction_process/)**: Basic verification test for the $c-\phi$ reduction algorithm.


<!-- 
NOTE ON PLAXIS BENCHMARK:
Stage 2 in this folder originally required 'example64.dll' (a Windows-only UDSM). 
To ensure compatibility with Linux/Kratos_Deltares_ROM, the material law was swapped 
to the native Kratos 'GeoMohrCoulombWithTensionCutOff2D' while preserving 
identical physical parameters. The DLL is only required for strict 1:1 numerical 
verification against PLAXIS.
-->


## 📚 Learning Resources

If you are new to the Kratos ROM application, you can follow these comprehensive resources:

- **Step-by-Step Tutorial**: [Kratos ROM Tutorial Repository](https://github.com/SADPR/Kratos_ROM_Tutorial)
- **Video Series**: [Kratos ROM Tutorial YouTube Playlist](https://youtube.com/playlist?list=PLJZAo1kyATsUA8U4hex36P4HLAr0KJWH_&si=sOpA-_IfUJdfTpMV)