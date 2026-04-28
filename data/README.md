# Datasets

The ADD-PINN release does not redistribute raw traffic data. Both datasets are
publicly available from their original providers under their own terms of use.
Download them and place them in the layout described below.

---

## Expected layout

```
data/
├── i24/
│   ├── 20221121.csv
│   ├── 20221122.csv
│   ├── 20221123.csv
│   ├── 20221129.csv
│   └── 20221202.csv
└── ngsim/
    └── ngsim_data.csv
```

All files are CSV with three columns:

| column        | type   | meaning                              |
| ------------- | ------ | ------------------------------------ |
| `time_index`  | int    | discrete time bin index (Δt = 2 s for I-24, 0.1 s for NGSIM) |
| `space_index` | int    | discrete cell index along the corridor |
| `speed_mph`   | float  | aggregated mean speed in mph         |

Speeds are normalized to [0, 1] by min-max scaling on the full aggregated
field at runtime in `src/utils.py:load_dataset`.

---

## I-24 MOTION

> Gloudemans, D., Wang, Y., Ji, J., Zachár, G., Barbour, W., Hall, E., Cebelak,
> M., Smith, L., & Work, D. B. (2023). *I-24 MOTION: An instrument for freeway
> traffic science.* Transportation Research Part C: Emerging Technologies,
> 155, 104311.

This release uses five days of aggregated speed data on a 4.2-mile section of
I-24 in Tennessee:

| Date         | Notes                                        |
| ------------ | -------------------------------------------- |
| 2022-11-21   | accident scenario (used for primary results) |
| 2022-11-22   | nominal weekday                              |
| 2022-11-23   | pre-Thanksgiving congestion                  |
| 2022-11-29   | nominal weekday                              |
| 2022-12-02   | nominal weekday                              |

Each file covers the 06:00–10:00 CST observation window and is aggregated to
a 100-cell × 7,200-step grid (Δx ≈ 0.04 mi, Δt = 2 s, total 4 hours, 4 mi).

**Access:** trajectory data is distributed by the I-24 MOTION program. See
the program homepage for the current data access policy and request portal.
*[Insert official I-24 MOTION data portal URL here at submission time.]*
After receiving access to raw trajectories, aggregate to the
(time_index, space_index, speed_mph) format above.

---

## NGSIM I-80

> U.S. Department of Transportation, Federal Highway Administration (FHWA).
> *Next Generation Simulation (NGSIM)* Vehicle Trajectories and Supporting
> Data. Public dataset.

The NGSIM slice used here matches the configuration of:

> Huang, A. J., & Agarwal, S. (2023). *On the limitations of physics-informed
> deep learning: Illustrations using first-order hyperbolic conservation
> law-based traffic flow models.* IEEE Open Journal of Intelligent
> Transportation Systems, 4, 279–293.

**Access:** NGSIM is freely available from the FHWA Next Generation Simulation
public portal. Download the I-80 trajectory subset and aggregate it to the
expected CSV format above.

---

## Reproducibility note

The aggregation step (raw trajectories to time-space speed grid) is deterministic given the bin sizes documented above. With matched seeds (SEEDS = [42, 123, 456, 789, 1024, 2048, 3000, 4096, 5555, 7777] in experiments/run_all.py) and matched hardware/CUDA versions, reported metrics are typically reproducible to within seed-to-seed standard deviation. Bit-for-bit reproduction is not guaranteed across different GPU models or CUDA versions due to non-deterministic floating-point operations in cuDNN.
