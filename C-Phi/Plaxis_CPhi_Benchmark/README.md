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

---

## 🧠 Understanding the C-Phi Reduction Algorithm

The **Strength Reduction Method** algorithm calculates the Factor of Safety (FoS) of a structure by systematically weakening the soil until numerical convergence fails (representing a physical landslide or collapse).

### The Math
Under the Mohr-Coulomb failure model, soil resists collapsing due to two primary strengths:
1. **Cohesion ($c$)**
2. **Friction Angle ($\phi$)**

The reduction is driven by a multiplier called the **Reduction Factor ($R$)**, where the **Factor of Safety ($FoS$)** is the inverse: 
$$ FoS = \frac{1}{R} $$

### Step-by-Step Logic
Assume the soil has an initial real cohesion of $c_0 = 10,000$ Pa and friction angle of $\phi_0 = 35^\circ$.

1. **Iteration 0 (Baseline)**: $R = 1.0$. The simulated soil retains 100% of its strength. The system is naturally stable under gravity.
2. **Iteration 1**: The solver drops the reduction factor (e.g., $R = 0.9$). 
   Kratos injects *fictitious, weakened* values into the finite element model:
   $$ c_{new} = c_0 \times R = 10,000 \times 0.9 = 9,000 \text{ Pa} $$
   $$ \tan(\phi_{new}) = \tan(\phi_0) \times R $$
   The solver checks if the gravity load can still be supported by this $9,000$ Pa soil. If it converges, the structure is still mathematically "safe".
3. **Iteration N (Failure)**: The solver continues iterating downward ($R = 0.8, 0.7, \dots$). 
   Eventually, it reaches a threshold where the material is too weak:
   $$ c_{new} = c_0 \times R = 10,000 \times 0.4 = 4,000 \text{ Pa} $$
   At this point, the mathematical equilibrium matrices diverge (Error/No Convergence). Physically, a slip surface has formed, and the displacements run off to infinity.
4. **The Verdict**: The very last stable $R$ value before the collapse is retrieved. If the structure collapsed at $R=0.4$ but was completely stable at $R=0.5$, then the soil is handling the loads at exactly *half* of its actual strength capacity. Therefore, the engineer determines:
   $$ FoS = \frac{1}{0.5} = 2.0 $$

### 💡 Real-World Analogy: Why do we reduce strength?
A common question arises: *Does soil naturally lose half its strength like this in real life?*

**No, it doesn't.** In reality, disasters usually occur because **loads increase** (e.g., torrential rains, floods, new constructions), not because the inherent friction of the soil magically halves. 

However, in geotechnical engineering, simulating a massive dam as you continuously add physical loads (water, weight, earthquakes) is a computational nightmare because stress distribution paths change chaotically.

Thus, the **C-Phi Reduction is a mathematical trick** to find our "margin of error". 
Imagine you build a brick wall weighing 100 kg, and it rests on a wooden table. To find the table's Factor of Safety, you have two options:
1. **The Real-Life Approach**: Keep adding books on top of the wall until the table breaks at 150 kg. (You discover you had a 50 kg margin).
2. **The C-Phi Approach**: Leave the wall's weight locked at 100 kg, but use math to virtually degrade the table's wood into cardboard, and then into paper, until the 100 kg crushes it. 

If the math shows the "virtual table" had to be made 1.5 times weaker before it broke under the normal 100 kg load, you conclude your actual wooden table is 1.5 times stronger than it strictly needs to be. This is exactly what Kratos and PLAXIS do to the soil's $c$ and $\phi$ parameters.

