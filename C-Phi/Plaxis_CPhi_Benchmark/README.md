# Advanced Plaxis C-Phi Reduction Benchmark

This directory contains an advanced hydro-mechanical benchmark for the **Strength Reduction Method (SRM)**, also known as the $c-\phi$ reduction process, used in geotechnical engineering to find the Factor of Safety (FoS) of a slope or embankment.

## 🌊 Problem Description

Unlike the basic C-Phi test in the main Kratos repository, this model is highly realistic:
*   **Mesh**: The geometry (`plxMesh.mdpa`) originates from PLAXIS 2D and contains a dense domain of 1184 nodes and 553 elements.
*   **Hydraulics**: It utilizes the `U_Pw` solver (Displacement + Pore Water Pressure) and features a defined **Phreatic Line** (water table).
*   **Constitutive Law**: It relies on a Mohr-Coulomb material model with tension cut-off to simulate soil failure.

### ⏱ Execution Stages
The analysis is automated by `Kratos_stages.py` and split into two steps:
1.  **Stage 1 (Initialization)**: The soil model settles under its own weight (Gravity load = -9.81 $m/s^2$) via a Quasi-Static solver. Initial geostatic stresses and steady-state pore pressures are computed.
2.  **Stage 2 (C-Phi Reduction)**: The `ApplyCPhiReductionProcess` gradually weakens the soil's cohesion ($c$) and friction angle ($\phi$) using a mathematical reduction factor. The Newton-Raphson solver iterates until it can no longer find a stable equilibrium. The point of non-convergence indicates a massive structural failure (landslide), yielding the actual Factor of Safety.

---

## 🛠 Fixes & Modifications from the Original Project

### 1. The UDSM Library (Windows vs Linux)
The original `MaterialParameters_stage2.json` file was configured to execute the soil failure model using an external **User Defined Soil Model (UDSM)** file named `example64.dll`. 
*   **The Issue**: A `.dll` is a compiled Windows binary. Attempting to run this on a Linux system throws a fatal `Cannot load the specified UDSM` error.
*   **The Fix**: The configuration was modified to bypass the `.dll` and use Kratos's native implementation of the exact same algorithm: `GeoMohrCoulombWithTensionCutOff2D`. The generic `UMAT_PARAMETERS` array (which passed raw numbers like `[5.38e6, 0.3, 10000.0, 35.0, 0.0, 0.0]` to the DLL) was rewritten using standard Kratos variables:
    *   `YOUNG_MODULUS`: 5.384615e6
    *   `GEO_COHESION`: 10000.0
    *   `GEO_FRICTION_ANGLE`: 35.0

### 2. Is the DLL strictly necessary?
**No. It is only required for strict cross-software verification.**

Using `example64.dll` forces Kratos to calculate the soil stresses using the exact external code that PLAXIS uses. This is done to achieve a *1-to-1 numerical benchmark* ensuring Kratos produces identical outputs to commercial software. 

Since Kratos has its own mathematically equivalent native `Mohr-Coulomb` formulation, bypassing the DLL does not change the physics of the problem; it simply processes the mathematics using Kratos's internal C++ code rather than PLAXIS's Fortran code.

If strict verification against PLAXIS is required in the future, the code must be provided with the Linux-compiled version of the library (`.so` format) instead of the Windows `.dll`.
