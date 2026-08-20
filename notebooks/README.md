# notebooks/

Three executed notebooks, committed **with their outputs** so the ReadTheDocs build
renders them without running anything (`nbsphinx_execute = "never"`).

| notebook | subject | priors |
|---|---|---|
| `01_encoded_tire_force.ipynb` | The ablation ladder on direct tire-force data: what each prior buys in accuracy *and* in physical violations, plus the soft-penalty control and a data-efficiency curve. | P1–P4 |
| `02_relaxation_graining_tire_cell.ipynb` | Transient behaviour ($\tau = \sigma/v$) against GRU and Neural ODE controls, then the thermal/wear/graining condition model and its structural guarantees. | P5, P7 |
| `03_four_wheel_physics_supervision.ipynb` | One shared tire model identified from IMU signals only through exact Newton–Euler equations, versus a per-wheel ablation. | P6 |

Run them from this directory (they insert the repository root on `sys.path`), or
re-execute all three with:

```bash
jupyter nbconvert --execute --inplace notebooks/*.ipynb
```

Every notebook seeds with `set_seed(0)` and uses in-repo synthetic data, so the numbers
are reproducible without any download. The graining results in notebook 2 are
**synthetic and weakly supervised** — a demonstrator of the model structure, not
validated real motorsport graining.
