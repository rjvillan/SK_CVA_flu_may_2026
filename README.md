# Influenza SLIR calibration — South Korea and Community of Valencia, 2016–2017

Code repository for the paper:

> **Kyeongah Nah, Jonggul Lee, Almudena Sánchez, and Rafael J.
Villanueva, Epidemic dynamics and parameter identifiability of the 2016–2017 influenza A(H3N2) season in South Korea and the Community of Valencia: a model-based comparison**

## Overview

This repository contains the Python code used to calibrate a SLIR (Susceptible–Latent–Infectious–Recovered) epidemiological model against influenza surveillance data from two regions:

- **CVA** — Community of Valencia, Spain (33 weekly data points, season 2016–2017)
- **SK** — South Korea (53 weekly data points, season 2016–2017)

Calibration is performed with a hybrid algorithm: **Particle Swarm Optimisation (PSO)** followed by **Nelder–Mead** local refinement. Multiple independent runs are executed and the best solution across all runs is selected. A parameter identifiability analysis is also included.

## Data sources

### Community of Valencia (CVA)

Observed data are weekly reported influenza cases per 100,000 inhabitants, extracted from:

> *Prevención y vigilancia de la gripe en la Comunitat Valenciana. Temporada 2016-2017.*  
> Informes de Salud nº 150. Generalitat Valenciana, Conselleria de Sanitat Universal i Salut Pública, 2017. ISSN 1139-6873.

The original report is included in this repository as **`IS-150.pdf`** because it is no longer accessible through the official health surveillance portal.

### South Korea (SK)

Observed data are weekly ILI (influenza-like illness) rates per 1,000 sentinel visits, from epidemiological weeks 36/2016 to 35/2017, as reported by the Korea Disease Control and Prevention Agency (KDCA) sentinel surveillance system.

## The SLIR model

The model tracks five compartments: S (susceptible), L (latent/exposed), I (infectious), R (recovered), and C (cumulative infectious). It includes demographic turnover (births/deaths).

**ODE system (daily units):**

```
dS/dt = μ − (β·I/N + δ)·S
dL/dt = β·S·I/N − (ε + δ)·L
dI/dt = ε·L − (γ + δ)·I
dR/dt = γ·I − δ·R
dC/dt = ε·L
```

**Parameters calibrated** (all specified in weekly units in `cfg_parametros.yaml`):

| Parameter | Description |
|-----------|-------------|
| `beta` | transmission rate (1/week) |
| `epsilon` | L→I progression rate (1/week) |
| `gamma` | I→R recovery rate (1/week) |
| `s0` | initial susceptible fraction |
| `l0` | initial latent fraction |
| `i0` | initial infectious fraction |
| `p` | reporting fraction (CVA) or ILI/infectious ratio (SK) |
| `a` | additive baseline of reported cases |

**Objective function:** RMSE between model output and observed data, plus proportional penalties for epidemiologically implausible R₀ or attack rate.

## Repository structure

```
SK_CVA_flu/
├── CVA/                         # Community of Valencia calibration
│   ├── 0.-Configuracion/
│   │   ├── cfg_calibrado.yaml   # PSO/NM settings
│   │   └── cfg_parametros.yaml  # parameter bounds
│   ├── 1.-Configurar_parametros.py     # interactive parameter setup
│   ├── 2.-Lanzar_calibrado_hibrido.py  # main launcher (PSO + NM)
│   ├── PSO/
│   │   ├── Modulo_MO_rPSO.py   # PSO algorithm
│   │   ├── aux_PSO.py           # PSO utilities (Pareto, boundary handling…)
│   │   └── aux_externos.py      # ID generation and guess writing
│   ├── NM/
│   │   └── Modulo_NM.py         # Nelder-Mead refinement module
│   ├── SIMULADOR/
│   │   └── gripe_slir.py        # SLIR model + error function
│   ├── RESULTADOS/              # calibration output (created at runtime)
│   └── 1.-Visualizacion_resultados/
│       ├── 1.-Mejor_ajuste_determinista.py  # best-fit export to Excel
│       ├── 2.-Identificabilidad.py          # identifiability analysis
└── SK/                          # South Korea calibration (same structure)
IS-150.pdf                       # CVA surveillance report (data source)
```

## Requirements

Python 3.9+ and the following packages:

```
numpy
scipy
pyyaml
pandas
openpyxl
matplotlib
```

Install with:

```bash
pip install numpy scipy pyyaml pandas openpyxl matplotlib
```

## How to run

All commands are run from inside either `CVA/` or `SK/`. The workflow is identical for both regions.

### Step 1 — Configure the search bounds (optional)

The file `0.-Configuracion/cfg_parametros.yaml` already contains the bounds used in the paper. To modify them interactively:

```bash
cd SK_CVA_flu/CVA
python 1.-Configurar_parametros.py
```

### Step 2 — Configure the calibration settings (optional)

Edit `0.-Configuracion/cfg_calibrado.yaml` to change the number of PSO particles, the maximum number of evaluations, or to enable/disable Nelder–Mead refinement.

### Step 3 — Run the calibration

```bash
cd SK_CVA_flu/CVA
python 2.-Lanzar_calibrado_hibrido.py --runs 5
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--runs N` | 5 | number of independent runs |
| `--paralelo` | off | run all N runs simultaneously |

Each run launches three concurrent processes: the SLIR simulator, the PSO optimiser, and the Nelder–Mead refiner. All inter-process communication happens through YAML files and a SQLite database; no network access is required.

Results are saved to `RESULTADOS/`:

- `evaluaciones_global.db` — all evaluated parameter sets and their errors
- `resumen_run_XX.yaml` — summary of each run (best error, parameters, evaluation count)
- `MEJORES/` — best parameter sets found during each run

### Step 4 — Visualise results

From inside the `1.-Visualizacion_resultados/` directory:

```bash
# Export best fit to Excel (parameters, compartments, observed vs model)
python 1.-Mejor_ajuste_determinista.py

# Identifiability plots and Spearman correlation heatmap
python 2.-Identificabilidad.py
```

## Calibration algorithm

The hybrid PSO + Nelder-Mead procedure works as follows:

1. **PSO** initialises a swarm of 30 particles using a Halton low-discrepancy sequence for better initial coverage of the parameter space.
2. During PSO, **Nelder-Mead** runs concurrently: whenever PSO finds a new global best, NM takes it as a starting point and performs a local refinement, depositing the improved solution back into PSO's guess queue.
3. After PSO reaches the evaluation budget, NM performs one final adjustment from the overall best point.
4. This cycle is repeated for N independent runs (different random seeds). The best solution across all runs is reported.

The simulator runs as a separate process and communicates with PSO/NM exclusively through files, making the architecture modular: any model can be plugged in by replacing `SIMULADOR/gripe_slir.py` while keeping PSO, NM, and the launcher unchanged.

## Adapting to a new model or dataset

1. Replace `SIMULADOR/gripe_slir.py` with your own simulator. The interface contract is:
   - **Input:** a YAML file with keys `Identificador`, `Semilla`, and one key per parameter.
   - **Output:** a YAML file with key `Errores: [value]` and an entry in `evaluaciones.db`.
2. Update `cfg_parametros.yaml` with your parameter names and bounds.
3. Set the simulator name in `cfg_calibrado.yaml` (`simulador: your_simulator_name`).

## License

The code is released under the MIT License. The data contained in `IS-150.pdf` is a public health report published by the Generalitat Valenciana and is reproduced here for reproducibility purposes only; it remains the property of its authors.
