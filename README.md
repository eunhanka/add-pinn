# ADD-PINN: Adaptive Domain Decomposition Physics-Informed Neural Networks for Traffic State Estimation

> Reference implementation for the manuscript "Adaptive Domain Decomposition Physics-Informed Neural Networks for Traffic State Estimation with Sparse Sensor Data".
> Submitted to Transportation Research Part C: Emerging Technologies, 2026.
> DOI: [to be added upon publication]

![status](https://img.shields.io/badge/status-under%20review-orange)
![license](https://img.shields.io/badge/license-MIT-blue)
![python](https://img.shields.io/badge/python-%E2%89%A53.9-blue)

---

## Overview

Traffic state estimation (TSE) on freeway corridors is a sparse-data, shock-laden state reconstruction problem: the macroscopic traffic state evolves under a hyperbolic conservation law (the LWR equation, rho_t plus q(rho)_x equals 0), but available sensors observe speed only at a handful of fixed locations. Standard physics-informed neural networks (PINNs) are well known to struggle when the underlying solution contains shocks, since a single global network must balance smooth-region accuracy against discontinuity sharpness.

ADD-PINN addresses this with a two-stage residual-guided framework. Stage 1 trains a single coarse PINN on the full domain to obtain an initial estimate and a global PDE-residual map. Stage 2 inspects this residual map, decides whether decomposition is warranted via a data-driven shock indicator, selects split positions from valleys of the smoothed residual profile, and fine-tunes child networks coupled by Rankine-Hugoniot interface conditions. Child networks are warm-started from the coarse parent for computational efficiency.

## Contributions

The four contributions evaluated in the manuscript are:

- C1: Two-stage residual-guided adaptive domain decomposition. ADD-PINN automatically detects subdomain count and split positions from the spatial residual profile of a coarse PINN solution, removing the need for manual specification of subdomain partitions.
- C2: Spatial decomposition is most effective for fixed-sensor TSE. Three independent arguments (shockwave geometry, data coverage asymmetry, residual anisotropy) together with a 120-run ablation study establish spatial-only decomposition as the preferred strategy in this regime.
- C3: Coarse-to-fine training with warm-start. Child networks inherit the parent's learned weights for fine-tuning, providing computational efficiency over training from scratch (2.4x speedup vs XPINN on I-24).
- C4: Large-scale validation on real freeway data. Evaluation across five days of I-24 MOTION (1,500 runs) and an NGSIM I-80 supplementary set, with controlled direction ablations and full statistical analysis.

---

## Repository structure

add-pinn-release/
- README.md                  this file
- LICENSE                    MIT
- CITATION.cff               citation metadata
- requirements.txt           Python dependencies
- .gitignore
- src/                       core library
  - __init__.py
  - model.py                 AdaStdpinnLWR plus baseline classes
  - utils.py                 data loading, metrics, sensor utilities
- experiments/               reproducible run scripts
  - run_all.py                       main I-24 sweep (B1 through B6)
  - run_domain_direction_ablation.py spatial / temporal / space-time ablation
  - run_ngsim_quick.py               NGSIM main runs
  - run_ngsim_no_force.py            NGSIM no-force-decomp control
  - gen_figure_data.py               extracts inputs for plot scripts
  - reextract_residuals.py           refreshes residual-profile data
  - plot_figures.py                  legacy figure orchestrator
  - statistical_analysis.py          paired t-tests with Holm-Bonferroni correction
  - verify_b6_b8.py                  sanity check for ADD-PINN vs XPINN
- paper/
  - tables_latex.py          generates LaTeX tables from results/tables/
- data/
  - README.md                data download / preparation instructions
- results/
  - tables/                  16 CSV files, all reported tables
  - figures/
    - output_final/          paper figures (PDF + PNG)
    - plot_motivation.py
    - plot_error_maps_final.py
    - plot_error_maps_2x2.py
    - plot_speed_fields_final.py
    - plot_single_panel_7s.py
    - plot_residual_profile_final.py
    - plot_sensor_sensitivity_final.py

---

## Installation

Tested with Python 3.9 to 3.13, PyTorch 2.0+ (CUDA 11.8 / 12.1 verified, CPU fallback supported).

git clone https://github.com/eunhanka/add-pinn.git
cd add-pinn-release
python -m venv .venv
source .venv/bin/activate          (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

requirements.txt lists torch>=2.0, numpy, pandas, scipy, matplotlib, and pyDOE.

---

## Data preparation

The repository does not redistribute raw datasets. See data/README.md for licensing-compliant download instructions for both data sources:

- I-24 MOTION: five days (20221121, 20221122, 20221123, 20221129, 20221202), aggregated to a 100 by 7,200 grid (Delta x approximately 211 ft, Delta t equals 2 s, 4-hour observation window 06:00-10:00 CST).
- NGSIM I-80: supplementary benchmark, configured as in Huang and Agarwal (2023).

After download, place files at:

data/i24/<YYYYMMDD>.csv
data/ngsim/ngsim_data.csv

---

## Reproducing the paper results

### Step 1: Prepare data
Follow data/README.md.

### Step 2: Main I-24 sweep
cd experiments
python run_all.py

Resume-safe: re-running skips configurations already present in results/tables/per_seed_all.csv. Full run is 6 methods times 5 sensor counts times 5 days times 10 seeds equals 1,500 experiments.

### Step 3: Domain-direction ablation (Section 5.2)
python run_domain_direction_ablation.py

Compares spatial-only (default ADD-PINN), temporal-only, and space-time decomposition variants on 20221121 and 20221122. Total: 2 days times 3 sensor counts times 2 ablation methods times 10 seeds equals 120 runs.

### Step 4: NGSIM supplementary (Section 5.5)
python run_ngsim_quick.py        (main NGSIM runs with force_decomp=True)
python run_ngsim_no_force.py     (default adaptive mode with force_decomp=False)

### Step 5: Generate figures
cd ../results/figures
python plot_motivation.py
python plot_error_maps_final.py
python plot_error_maps_2x2.py
python plot_speed_fields_final.py
python plot_single_panel_7s.py
python plot_residual_profile_final.py
python plot_sensor_sensitivity_final.py

All scripts write to results/figures/output_final/.

### Step 6: LaTeX tables
cd ../../paper
python tables_latex.py

### Step 7: Statistical analysis
cd ../experiments
python statistical_analysis.py

Performs paired t-tests with Holm-Bonferroni correction, Wilcoxon signed-rank tests, sign tests, Cohen's d, and 95 percent confidence intervals.

---

## Methods (paper nomenclature)

The paper reports 6 methods (B1 through B6). The development codebase uses an extended set of identifiers (B2 through B8) with several development-only variants. The mapping below shows the correspondence between the codebase identifiers and the published baseline IDs.

| Codebase ID | Paper ID | Method                | Description                               |
| ----------- | -------- | --------------------- | ----------------------------------------- |
| B2          | B1       | Simple NN             | data fit only, no PDE constraint          |
| B3          | B2       | Vanilla PINN          | fixed weights, no decomposition           |
| B4          | B3       | PINN + RAR            | residual-adaptive refinement              |
| B5          | B4       | PINN + viscosity      | Huang and Agarwal (2023)                  |
| B8          | B5       | XPINN                 | space-time decomposition                  |
| B7          | B6       | ADD-PINN (ours)       | two-stage adaptive DD with R-H interfaces |

---

## Citation

@article{add-pinn-2026,
  title   = {Adaptive Domain Decomposition Physics-Informed Neural Networks for Traffic State Estimation with Sparse Sensor Data},
  author  = {Ka, Eunhan and Leclercq, Ludovic and Ukkusuri, Satish V.},
  journal = {Transportation Research Part C: Emerging Technologies},
  year    = {2026},
  note    = {Manuscript under review}
}

A finalized BibTeX entry will replace this block upon acceptance.

---

## License

Released under the MIT License. See LICENSE.

---

## Contact

Eunhan Ka (corresponding author): kae@purdue.edu

Lyles School of Civil and Construction Engineering, Purdue University

ORCID: https://orcid.org/0000-0003-0954-8075
