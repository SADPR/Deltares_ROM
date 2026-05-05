# Plaxis C-Phi Reduced Order Model (ROM) 🚀

This repository contains the implementation of a **Reduced Order Model (ROM)** for the Plaxis C-Phi Strength Reduction benchmark. The goal is to build a high-performance surrogate model capable of predicting the **Factor of Safety (FoS)** of a soil slope in a fraction of the time required by a Full Order Model (FOM) simulation.

---

## 📂 Project Structure & Directory Guide

The core project logic is split into human-readable scripts and machine-readable data.

### 📜 Master Scripts
- **`launch_cphi_rom.py`**: The main entry point. Use this to run the different stages (0 to 3) of the ROM lifecycle.
- **`cphi_rom_manager.py`**: The engine driving the automation. It handles the Kratos execution, snapshot extraction, and file management.
- **`Kratos_stages.py`**: The original FOM runner used for baseline validation.

### 📁 Data Storage (`rom_data/`)
The project uses a **Transparent Architecture** where data is organized into stages:

- **`stage1_fom/`**: Contains the "Ground Truth" snapshots (`fom_c...npy`) and Quantities of Interest (`qoi_...json`).
- **`stage2_pod/`**: Contains the **POD Basis**. After running SVD on the FOM data, the master shape vectors (`basis.npy`), metadata (`basis.json`), and mapping (`NodeIds.npy`) are stored here.
- **`stage3_ver/`**: Results from the **ROM Verification** execution (running on training parameters).
- **`stage4_test/`**: Results for the **ROM Testing** (running on unseen parameters).
- **`reports/`**: Human-readable JSON summaries of each stage (errors, SVD decay, etc.).

### Filename Convention
Files are named using a `_MuToken` based on the physics parameters:
`fom_c<cohesion>_phi<friction>.npy`

---

## 🛠 How to Run

Because the project utilizes an external **User Defined Soil Model (UDSM)** shared library (`example64.so`), you must use the `LD_PRELOAD` shim to handle a missing symbol (`show_extra_info_`).

### 1. Initialize the Shim (Once per session)
```bash
cat > /tmp/show_extra_stub.c <<'EOC'
void show_extra_info_(void) {}
EOC
gcc -shared -fPIC -o /tmp/libshow_extra_stub.so /tmp/show_extra_stub.c
```

### 2. Execute the ROM pipeline
```bash
LD_PRELOAD=/tmp/libshow_extra_stub.so \
PYTHONPATH=/home/kratos/Kratos_Deltares/bin/Release:${PYTHONPATH} \
python3 launch_cphi_rom.py
```

---

## 🧠 The ROM Lifecycle (Stages)

We use a staged approach to transform raw FEA data into a fast surrogate:

| Stage | Name | Description |
| :--- | :--- | :--- |
| **0** | **Sampling** | Visualizes the training grid (e.g., 3x3) across the Cohesion and Friction Angle space. |
| **1** | **FOM Training** | Runs the expensive Full Order Model cases to harvest the physical "vocabulary" of the slope failure. |
| **2** | **POD Basis** | Compressed the collected snapshots using SVD to find the dominant failure modes. |
| **3** | **Verification** | Runs the ROM using the new basis and compares its FoS prediction against the FOM truth. |

---

## 🔬 Technical Background

### Why train on a "Process"?
Unlike simple structural problems, slope failure is **path-dependent**. We don't just care about the start and end; we care about the progressive development of the slip surface. By training the ROM on the entire $c-\phi$ reduction steps, we teach it the "evolution" of the failure, allowing it to predict the exact divergence point much more accurately.

### The ROM Error Indicator
The ROM is mathematically "stiff"—it wants to converge even when the physics say it shouldn't. We use a **Residual-based Error Indicator**. As soon as the ROM tries to project a state where the internal forces are widely unbalanced (high residual), the manager detects this spike and "trips" the simulation early, recording that specific point as the **Factor of Safety**.
