# Datasets

This directory contains the aggregated traffic speed data used to train and evaluate ADD-PINN. Both datasets are derived from publicly accessible sources; the raw trajectory data are not redistributed here.

---

## Directory layout

data/
- i24/
  - 20221121.csv  (4.2-mile section of I-24, 06:00-10:00 CST, accident scenario)
  - 20221122.csv  (nominal weekday)
  - 20221123.csv  (pre-Thanksgiving congestion)
  - 20221129.csv  (nominal weekday)
  - 20221202.csv  (nominal weekday)
- ngsim/
  - ngsim_data.csv  (I-80 benchmark slice as in Huang and Agarwal, 2023)

---

## File schema

All CSV files share the following column convention used by `src/utils.py:load_dataset`:

| column  | type   | meaning                                          |
| ------- | ------ | ------------------------------------------------ |
| `t`     | int    | discrete time bin index                          |
| `x`     | int    | discrete cell index along the corridor           |
| `speed` | float  | aggregated mean speed in **feet per second (ft/s)** |

Note that the speed column is in ft/s, not mph. The loader at `src/utils.py:load_dataset` (line 34) converts ft/s to mph by multiplying by 0.681818. Speeds are subsequently normalized to [0, 1] via min-max scaling on the full aggregated field, and the free-flow speed is estimated as the 95th percentile of the aggregated speed field.

Domain dimensions used by the experiments:

| dataset | spatial extent | temporal extent | grid (n_x x n_t) | dx       | dt    |
| ------- | -------------- | --------------- | ---------------- | -------- | ----- |
| I-24    | 21,120 ft (4 mi) | 14,400 s (4 h) | 100 x 7,200      | ~211 ft  | 2 s   |
| NGSIM   | 1,600 ft       | 900 s           | varies            | varies   | varies |

---

## I-24 MOTION

This release uses five days of aggregated speed data from the I-24 MOTION testbed:

> Gloudemans, D., Wang, Y., Ji, J., Zachar, G., Barbour, W., Hall, E., Cebelak, M., Smith, L., and Work, D. B. (2023). I-24 MOTION: An instrument for freeway traffic science. Transportation Research Part C: Emerging Technologies, 155, 104311.

The aggregated CSV files in `data/i24/` were derived from the I-24 MOTION raw vehicle trajectory data with explicit acknowledgement of the original program. We thank the I-24 MOTION team and the Tennessee Department of Transportation for providing access to the trajectory dataset that made this work possible.

If you use the data files in this directory, please cite the original I-24 MOTION paper above in addition to citing this work.

Raw trajectory data is distributed by the I-24 MOTION program through their official channels. Refer to the program homepage for the current data access policy.

---

## NGSIM I-80

The NGSIM benchmark slice was prepared in the configuration of:

> Huang, A. J. and Agarwal, S. (2023). On the limitations of physics-informed deep learning: Illustrations using first-order hyperbolic conservation law-based traffic flow models. IEEE Open Journal of Intelligent Transportation Systems, 4, 279-293.

NGSIM raw trajectories are publicly available from the U.S. Department of Transportation FHWA Next Generation Simulation portal.

---

## Aggregation procedure

The mapping from raw vehicle trajectories to the aggregated speed grid follows a standard binning procedure: for each spatiotemporal cell (x_i, t_j) of size dx by dt, the speed value is the mean of all vehicle speeds whose trajectory passes through that cell during that time bin, averaged across lanes. Cells with no vehicle crossings are filled by linear interpolation along the time axis.

The aggregation script itself is not included in this release because it relies on a specific raw-data ingestion pipeline tied to the I-24 MOTION raw trajectory format. Researchers interested in regenerating the aggregated CSVs from raw data are referred to standard binning procedures used in the macroscopic traffic flow literature, for example as documented in Treiber and Kesting (2013, "Traffic Flow Dynamics") or Seo et al. (2017). The bin sizes used in this work are dx = 211 ft, dt = 2 s for I-24 and as documented above for NGSIM.

---

## Reproducibility note

The aggregated CSVs in this directory are deterministic given the bin sizes documented above. With matched seeds (`SEEDS = [42, 123, 456, 789, 1024, 2048, 3000, 4096, 5555, 7777]` in `experiments/run_all.py`) and matched hardware/CUDA versions, reported metrics are typically reproducible to within seed-to-seed standard deviation. Bit-for-bit reproduction is not guaranteed across different GPU models or CUDA versions due to non-deterministic floating-point operations in cuDNN.
