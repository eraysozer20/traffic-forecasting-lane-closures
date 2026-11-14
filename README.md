# Traffic Forecasting Under Lane Closure Conditions

Code for my master's thesis (Eray Sözer, CSE). The thesis is about traffic speed forecasting on freeway sensor networks when lanes are closed. It introduces **ResCAL**, a residual correction model that sits on top of a base forecaster (STGCN, Graph WaveNet or LSTM) and uses lane-closure information to fix the base model's errors near work zones.

The datasets, trained models and results are **not** in this repository because of their size. The scripts below rebuild the inputs from public Caltrans PeMS data.

## Repository layout

```
data_pipeline/   Builds the model inputs from raw Caltrans PeMS data
  ├─ download_occ_pems.py, extract_station_meta.py, extract_lane_closures.py   download from PeMS
  ├─ match_closure_with_sensors.py, grade_closures.py                          link closures to sensors
  ├─ create_DESPT_input.py, find_unseen_window.py, cut_save_data.py, h5_2_csv.py
  ├─ fetch_station_latlon.py, build_osrm_adj.py, build_STGCN_input.py          road-distance adjacency
  ├─ analyze_*.py, compare_speed_closure.py, visualize*.py, report.py          analysis and plots
  ├─ calibration/                                                              SUMO simulation calibration (district 08)
  ├─ run_pipeline.py / run_pipeline.ps1 / run_all*.ps1                         run the steps for every district
  └─ order.txt                                                                 order of the pipeline steps

rescal/          Models, training and experiments
  ├─ ResCAL.py, MyResCAL.py                       ResCAL residual correction model
  ├─ STGCN_WZ.py, GraphWaveNet_WZ.py, LSTM_WZ.py  closure-aware base models ("_NO" = without closure input)
  ├─ main3.py                                     main experiment: baselines + ResCAL ablations with k-fold CV
  ├─ main3_*.py, main_*.py, main2.py              earlier and alternative experiment scripts
  ├─ utils/                                       CLI args, dataset configs, data loading, windowing, metrics, residuals
  ├─ clean/                                       refactored training code (data / models / training)
  ├─ contrast_*.py, closure_contrast.py           contrastive pre-training experiments for closure behaviour
  ├─ create_table.py, table2.py, table3.py, ab_table.py   result tables
  └─ *.sh                                         SLURM job scripts for the GPU cluster

trl_dag/         Earlier exploration: temporal representation learner + dynamic graph generator
notebooks/       pems03_merge.ipynb (data merging for district 03)
```

## Data pipeline

The steps are listed in `data_pipeline/order.txt`. In short:

1. Download station data and lane-closure records from PeMS (needs a free PeMS account).
2. Match closures to sensors and pick evaluation windows.
3. Build speed and closure tensors (`.h5` / `.csv`) and road-distance adjacency matrices with OSRM.
4. Copy the outputs to `rescal/data/PEMSXX/`.

The PeMS download scripts read credentials from environment variables:

```bash
export PEMS_USERNAME="you@example.com"
export PEMS_PASSWORD="..."
```

Some scripts still contain absolute Windows/WSL paths from my machine. Change them to point to your own data folder.

## Running the experiments

```bash
cd rescal
python main3.py --enable-cuda \
  --dataset_choice PEMSD5 \
  --base_model lstm \
  --model_variant wz \
  --epochs 20 --num_folds 5
```

`--base_model` can be `stgcn`, `gwnet` or `lstm`. `--dataset_choice` can be `Tyson`, `PEMSD3`–`PEMSD12` or their `SUB` subsets. Per-dataset settings are in `utils/configs.py`. `run_main3_ablation.sh` runs the full ablation as a SLURM array job.

## Requirements

Python 3.9+, PyTorch, NumPy, pandas, SciPy, h5py, matplotlib and tqdm. The data pipeline also uses `requests`, `playwright` (PeMS download), OSRM (road distances) and SUMO / `sumolib` (calibration).

## Acknowledgements

The STGCN-WZ base model and the Tyson's Corner dataset come from the original *Predicting-taffic-speed-based-on-graph-neural-networks* project, a 2021 master's thesis supervised by Dr. David Lattanzi and funded by VDOT. The ResCAL model, the closure-aware GWNet/LSTM variants, the PeMS lane-closure pipeline and the experiment code are my own work.
