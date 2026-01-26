# regDL-TCIP

Regional Deep Learning (regDL) for Tropical Cyclone Intensity Projection (TCIP).

## Project description
regDL-TCIP provides a full, reproducible pipeline for tropical cyclone intensity projection using
regional deep learning models. It covers CMIP6/WRF preprocessing, optional idealized WRF extraction,
dataset canonicalization and splitting, model configuration, and training with registry-driven
pipelines. A SLURM wrapper orchestrates the whole workflow so you can change a run by editing one
file instead of touching every script.

## Control flow (important)
The **wrapper controls everything**. If you keep the same workflow, you only need to edit the wrapper.
The information flow is:

```
wrapper -> cli -> registry -> module
```

- The wrapper passes parameters to the CLI.
- The CLI selects the right registry entry.
- The registry dispatches to the correct module implementation.

## Project layout (what each file/dir does)
- `README.md` : project overview (this file)
- `FRAMEWORK.txt` : concise project layout reference
- `cli/` : entry points for the main steps
  - `cli/build_model.py` : build model config + dummy checkpoint + YAML history
  - `cli/feed_data.py` : canonicalize data + split into train/val/test
  - `cli/train.py` : training entry (selects pipeline via registry)
- `configs/` : outputs for history/configs (YAML/JSON)
  - `configs/model/` : model history YAMLs (e.g., `AFNO_v1.yaml`)
- `module/` : core package (registries + implementations)
  - `module/data_canonicalizer/` : canonicalizers + splitters + data registry
    - `module/data_canonicalizer/canonicalizers/` : dataset readers/normalizers
    - `module/data_canonicalizer/splitters/` : train/val/test split strategies
    - `module/data_canonicalizer/registry.py` : dataset type -> canonicalizer/splitter
  - `module/models/` : architectures, builders, and workflow helpers
    - `module/models/architectures/` : model definitions (e.g., AFNO-TCP)
    - `module/models/blocks.py` : AFNO/BC blocks + `BLOCK_REGISTRY`
    - `module/models/builders/` : config builders for architectures
    - `module/models/registry.py` : model registry + YAML writer
    - `module/models/workflow.py` : checkpoint + config writing helpers
  - `module/training/` : training loop, datasets, losses, and pipelines
    - `module/training/datasets.py` : windowing + dataset loaders
    - `module/training/losses.py` : weighted loss construction
    - `module/training/masks.py` : BC rim utilities
    - `module/training/pipelines/` : training pipelines (e.g., `P1-TC_AFNO`)
    - `module/training/registry.py` : pipeline registry
- `preprocess/` : raw preprocessing scripts (reference implementations)
  - `preprocess/WRF/` : WRF/CMIP6 preprocessing steps
    - `preprocess/WRF/CMIP6-WRF_step1.py`
    - `preprocess/WRF/CMIP6-WRF_step2.py`
    - `preprocess/WRF/WRF-IDEALIZED.py`
- `wrapper/` : SLURM job wrappers
  - `wrapper/job_br200.sh` : main SLURM job wrapper (add `job2`, `job3`, ... for other workflows)

## Parameters used in `wrapper/job_br200.sh` (checked)
This section documents every parameter used in the wrapper, its meaning, and the current values from the file.

### Run flags (1 = run, 0 = skip)
- `CMIP6_WRF=(1 1)` : toggle CMIP6/WRF preprocessing steps; `[0]=step1 (crop), [1]=step2 (npy)`
- `WRF_IDEALIZED=(1)` : toggle the idealized WRF pipeline
- `TRAINING=(1 1 1)` : toggle training steps; `[0]=build, [1]=feed, [2]=train`

### SLURM resources (top of the wrapper)
- `#SBATCH -N 1` : number of nodes
- `#SBATCH -t 10:00:00` : wall time limit
- `#SBATCH -J TCNN-ctl` : job name
- `#SBATCH -p gpu` : partition/queue
- `#SBATCH --gpus-per-node=4` : GPUs per node
- `#SBATCH --ntasks-per-node=4` : tasks per node (MPI ranks)
- `#SBATCH -A r00043` : account/project ID
- `#SBATCH --mem=200G` : total memory per node

### Global paths
- `ROOT` : repo root (derived from wrapper location)
- `PREPROCESS_DIR` (`/N/slate/kmluong/PROJECT2`) : external CMIP6/WRF preprocessing directory that contains `step1.py` and `step2.py`

### CMIP6/WRF preprocessing inputs/outputs
- `track_file` (`/N/project/hurricane-deep-learning/data/cmip6/ssp245_2080_2100_track.txt`) : ASCII track file used to center/crop around TC tracks
- `raw_data_dir` (`/N/project/hurricane-deep-learning/data/cmip6/ssp245_2080_2100/`) : directory of raw WRF output files (e.g., `raw_wrfout_*`)
- `level1_dir` (`/N/slate/kmluong/PROJECT2/level_1_data_ssp245`) : output directory for cropped NetCDF files (step1)
- `level2_dir` (`/N/slate/kmluong/PROJECT2/level_2_data_ssp245`) : output directory for NumPy arrays (step2)

### CMIP6/WRF preprocessing parameters
- `imsize_x` (`100`) : crop width (grid points in x)
- `imsize_y` (`100`) : crop height (grid points in y)
- `frames` (`5`) : number of consecutive time frames for each sample (step2)
- `var_levels` (`U10m V10m SST LANDMASK U28 V28 U05 V05 T23 QVAPOR10 PHB10`) : variable-level codes to extract
- `prefix` (`wrf_tropical_cyclone_track_${frames}_dataset`, expands to `wrf_tropical_cyclone_track_5_dataset`) : output filename prefix for step2

### CMIP6/WRF step1 (crop) CLI params
Called as `python step1.py` in `PREPROCESS_DIR`.
- `--track_file` (`${track_file}`) : TC track file used to locate storm centers
- `--data_dir` (`${raw_data_dir}`) : directory of raw WRF outputs
- `--workdir` (`${level1_dir}`) : output directory for cropped NetCDFs
- `--imsize_x` (`${imsize_x}`) : crop width (grid points)
- `--imsize_y` (`${imsize_y}`) : crop height (grid points)

### CMIP6/WRF step2 (npy) CLI params
Called as `python step2.py` in `PREPROCESS_DIR`.
- `--indir` (`${level1_dir}`) : input directory of step1 NetCDF files
- `--outdir` (`${level2_dir}`) : output directory for NumPy arrays
- `--frames` (`${frames}`) : number of frames per sample
- `--var_levels` (`${var_levels[@]}`) : variable-level list to extract
- `--prefix` (`${prefix}`) : output filename prefix

### WRF idealized pipeline params (`preprocess/WRF/WRF-IDEALIZED.py`)
- `--imsize_variables` (`${imsize_x} ${imsize_y}`) : spatial crop size `[width height]`
- `--root` (`/N/slate/kmluong/PROJECT2/WRF`) : output root directory
- `--wrf_base` (`/N/project/Typhoon-deep-learning/data/tc-wrf/`) : base directory containing idealized WRF experiment folders
- `--var_levels` (`U10m V10m SST LANDMASK U14 V14 U03 V03 T12 QVAPOR05 PHB05`) : variable-level list to extract
- `--experiment_wrf` (`exp_02km_m01 exp_02km_m02 exp_02km_m03 exp_02km_m04 exp_02km_m05 exp_02km_m06 exp_02km_m07 exp_02km_m08 exp_02km_m09 exp_02km_m10`) : experiment folder names to process
- `--X_resolution` (`d01`) : resolution token in filenames

### Training inputs/outputs (used by build/feed/train)
- `DATA_PATH` (`/N/slate/kmluong/PROJECT2/level_2_data/wrf_tropical_cyclone_track_5_dataset_X.npy`) : input dataset `.npy` (X array) for training/feeding
- `TMP_DIR` (`/N/slate/kmluong/PROJECT2/tmp`) : temp directory for train/val/test split arrays
- `CKPT_DIR` (`/N/slate/kmluong/PROJECT2/checkpoints`) : checkpoint output directory
- `MODEL_NAME` (`AFNO-TCP.pt`) : base checkpoint filename
- `MODEL_CFG` (`${CKPT_DIR}/AFNO_config1.json`) : JSON model config path
- `MODEL_YAML` (`${ROOT}/configs/model/AFNO_v1.yaml`) : YAML history file path (written by build step)
- `WRF_DIR` (`/N/slate/kmluong/PROJECT2/WRF/wrf_data`) : directory of WRF experiment `.npy` files for data_mode=1

### Build model params (wrapper -> `cli.build_model`)
- `--architecture` (`afno_v1`) : model architecture key (registry choice)
- `--output` (`${MODEL_YAML}`) : output YAML history path
- `--num_vars` (`11`) : number of input/output variables (channels)
- `--num_times` (`3`) : number of input time steps
- `--height` (`100`) : spatial grid height
- `--width` (`100`) : spatial grid width
- `--num_blocks` (`6`) : number of AFNO blocks
- `--film_zdim` (`128`) : FiLM conditioning embedding dim
- `--e_channels` (`256`) : expressive channels
- `--hidden_factor` (`8`) : hidden expansion factor
- `--mlp_expansion_ratio` (`4`) : MLP expansion ratio
- `--stem_channels` (`256`) : input CNN expressive power
- `--checkpoint_dir` (`${CKPT_DIR}`) : checkpoint output directory
- `--checkpoint_name` (`${MODEL_NAME}`) : checkpoint filename
- `--model_config_path` (`${MODEL_CFG}`) : JSON config output path

### Feed data params (wrapper -> `cli.feed_data`)
- `--data_mode` (`1`) : data loading mode (0 = single `.npy`, 1 = WRF exp files)
- `--data_path` (`${DATA_PATH}`) : path to the input `.npy` dataset (mode 0 or X array)
- `--wrf_dir` (`${WRF_DIR}`) : WRF experiment directory (mode 1)
- `--x_resolution` (`d01`) : resolution token in filenames
- `--imsize` (`100`) : spatial size token in filenames
- `--exp_split` (`12348910+57+6`) : compact exp split string `train+val+test`
- `--train_frac` (`0.7`) : training split fraction
- `--val_frac` (`0.2`) : validation split fraction
- `--test_frac` (`0.1`) : test split fraction
- `--num_segments` (`1`) : number of val/test segments (0 disables segmented split)
- `--step_in` (`3`) : number of input frames used per sample
- `--temp` (`${TMP_DIR}`) : temp directory for split arrays

### Train params (wrapper -> `cli.train`)
- `srun --ntasks=4 --gpus=4 --gpus-per-task=1` : run training across 4 tasks/GPUs
- `--trainer` (`afno_tcp_v1`) : training pipeline key (registry choice)
- `--step_in` (`3`) : number of input frames used per sample
- `--batch_size` (`8`) : batch size
- `--num_workers` (`0`) : dataloader worker count
- `--pin_memory` (`true`) : pin CPU memory in dataloader
- `--num_vars` (`11`) : number of input/output variables
- `--num_times` (`3`) : number of input time steps
- `--height` (`100`) : spatial grid height
- `--width` (`100`) : spatial grid width
- `--num_blocks` (`8`) : number of AFNO blocks or hidden layers
- `--rim` (`1`) : boundary pixels enforced from target
- `--learning_rate` (`1e-4`) : optimizer learning rate
- `--weight_decay` (`1e-4`) : optimizer weight decay
- `--num_epochs` (`300`) : number of training epochs
- `--film_zdim` (`64`) : FiLM conditioning embedding dim
- `--checkpoint_dir` (`${CKPT_DIR}`) : checkpoint output directory
- `--checkpoint_name` (`trained-${MODEL_NAME}`) : checkpoint filename
- `--e_channels` (`256`) : expressive channels
- `--hidden_factor` (`8`) : hidden expansion factor
- `--mlp_expansion_ratio` (`4`) : MLP expansion ratio
- `--stem_channels` (`256`) : input CNN expressive power
- `--L1_weight` (`1.0`) : weight for L1 loss
- `--L2_weight` (`0.0`) : weight for L2 loss
- `--Center_weight` (`0.0`) : weight for center loss
- `--Extreme_weight` (`0.5`) : weight for extreme loss
- `--HighFreq_weight` (`1.0`) : weight for high-frequency loss
- `--loss_gamma_center` (`1.0`) : center emphasis strength
- `--loss_center_width` (`0.25`) : center Gaussian sigma
- `--loss_center_width_mode` (`ratio`) : interpret center width as ratio or pixels
- `--loss_gamma_extreme` (`1.0`) : extreme emphasis strength
- `--loss_extreme_mode` (`zscore`) : extreme scoring mode
- `--loss_extreme_q` (`0.70`) : extreme percentile threshold
- `--loss_extreme_scale` (`1.0`) : extreme shaping scale
- `--loss_normalize_weights` (`true`) : normalize weights to mean 1
- `--loss_eps` (`1e-6`) : loss numerical stability epsilon
- `--high_freq_component_loss` (`false`) : use magnitude-spectrum loss instead of residual-spectrum loss
- `--high_freq_cutoff_ratio` (`0.5`) : high-frequency cutoff ratio in [0, 1]
- `--init_model_path` (`${CKPT_DIR}/${MODEL_NAME}`) : path to dummy/initial model state_dict
- `--model_config_path` (`${MODEL_CFG}`) : model config JSON path
- `--temp` (`${TMP_DIR}`) : temp directory for split arrays

## Notes
- If the workflow stays the same, only adjust the wrapper; the CLI/registry/module chain stays intact.
- Use `FRAMEWORK.txt` as the quick reference map.
