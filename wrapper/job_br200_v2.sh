#!/bin/bash -l
# ==============================================================================
# WRF-centric batch wrapper (CMIP6 still supported)
# - All user-selectable variables are grouped up top.
# - WRF idealized workflow wired (step1 crop → optional step2 chunk → feed_data).
# ==============================================================================

# SLURM -------------------------------------------------------------
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -J TCNN-wrf
#SBATCH -p gpu
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH -A r00043
#SBATCH --mem=200G

module load python/gpu/3.10.10
set -x

ROOT="/N/u/kmluong/BigRed200/regDL-TCIP"   # repo root
WORKDIR_BASE="/N/slate/kmluong/regDL-TCIP"          # base working directory; datasource subdir will be appended
ideal_wrf_base="/N/project/Typhoon-deep-learning/data/tc-wrf/"   # idealized input base dir, no need for track file
CMIP6_BASE_DIR="/N/project/hurricane-deep-learning/data/cmip6"   # contains *_track.txt and matched raw dirs
CMIP6_TRACK_GLOB="*_track.txt"                                    # basename <name>_track.txt => raw dir <name>
cd "$ROOT"

# ------------------------------------------------------------------------------
# Selectable flags (1=run, 0=skip)
# ------------------------------------------------------------------------------
DATASOURCE="CMIP6"  # choose data source workflow: CMIP6 or IDEALIZED
CMIP6_WRF=(0 1)         # auto-off by DATASOURCE (manual on, auto off)
WRF_IDEALIZED=(0 0)     # auto-off by DATASOURCE (manual on, auto off)
TRAINING=(1 1 1)       # [build model config][run feed_data split][run training loop]
datasource_key="${DATASOURCE^^}"
case "${datasource_key}" in
    CMIP6|IDEALIZED)
        WORKDIR="${WORKDIR_BASE}/${datasource_key}"
        ;;
    *)
        echo "ERROR: DATASOURCE must be CMIP6 or IDEALIZED (got '${DATASOURCE}')" >&2
        exit 1
        ;;
esac

# ------------------------------------------------------------------------------
# Common scratch / checkpoints / model config
# ------------------------------------------------------------------------------
TMP_DIR="$WORKDIR/tmp"                  # temp output for train/val/test splits
CKPT_DIR="$WORKDIR/checkpoints"         # checkpoint output directory
PREPROCESS_DIR="${ROOT}/preprocess"           # base dir for cmip6 step1/2 scripts
MODEL_NAME="AFNO-TCP.pt"                                 # base checkpoint name
MODEL_CFG="${CKPT_DIR}/AFNO_config1.json"                # build_model output config (json)
MODEL_YAML="${ROOT}/configs/model/AFNO_v1.yaml"           # build_model output yaml

# ------------------------------------------------------------------------------
# Model build params (cli.build_model)
# ------------------------------------------------------------------------------
BUILD_ARCH="afno_no_bc"          # architecture id for cli.build_model / cli.train (e.g., afno_v1, afno_no_bc)
MODEL_STEP_IN=3               # shared temporal length for feed_data/build_model/train
BUILD_NUM_VARS=11             # channel count in X (auto-overridden if AUTO_NUM_VARS_FROM_DATA=true) ###AUTO OVERRIDEN###
BUILD_NUM_TIMES="${MODEL_STEP_IN}"  # frames used during training
BUILD_HEIGHT=100              # spatial height
BUILD_WIDTH=100               # spatial width
BUILD_NUM_BLOCKS=6            # AFNO depth for init config
BUILD_FILM_ZDIM=128           # FiLM latent size
BUILD_E_CHANNELS=256          # encoder stem channels
BUILD_HIDDEN_FACTOR=8         # AFNO hidden factor
BUILD_MLP_EXPANSION_RATIO=4   # MLP expansion
BUILD_STEM_CHANNELS=256       # stem width

# ------------------------------------------------------------------------------
# Training hyperparameters (cli.train)
# ------------------------------------------------------------------------------
TRAIN_TRAINER="afno_tcp_v1"        # trainer registry key
TRAIN_STEP_IN="${MODEL_STEP_IN}"   # input frames per sample
TRAIN_BATCH_SIZE=8                 # per-GPU batch size
TRAIN_NUM_WORKERS=0                # 0 on GPU nodes to avoid fork+CUDA issues
TRAIN_PIN_MEMORY=true              # pin memory for data loader
TRAIN_NUM_VARS=11                  # channel count in X (auto-overridden if AUTO_NUM_VARS_FROM_DATA=true)
TRAIN_NUM_TIMES="${MODEL_STEP_IN}" # frames per sample
TRAIN_HEIGHT=100                   # spatial height
TRAIN_WIDTH=100                    # spatial width
TRAIN_NUM_BLOCKS=8                 # AFNO depth during training
TRAIN_RIM=1                        # rim padding flag
TRAIN_LR=1e-4                      # learning rate
TRAIN_WD=1e-4                      # weight decay
TRAIN_EPOCHS=300                   # total epochs
TRAIN_FILM_ZDIM=64                 # FiLM latent size used in trainer
TRAIN_E_CHANNELS=256               # encoder stem channels
TRAIN_HIDDEN_FACTOR=8              # AFNO hidden factor
TRAIN_MLP_EXP_RATIO=4              # MLP expansion
TRAIN_STEM_CHANNELS=256            # stem width
TRAIN_L1=1.0                       # L1 loss weight
TRAIN_L2=0.0                       # L2 loss weight
TRAIN_CENTER_WEIGHT=0.0            # center loss weight
TRAIN_EXTREME_WEIGHT=0.5           # extreme loss weight
TRAIN_HIGHFREQ_WEIGHT=1.0          # high-frequency loss weight
TRAIN_GAMMA_CENTER=1.0             # center gamma
TRAIN_CENTER_WIDTH=0.25            # center width
TRAIN_CENTER_WIDTH_MODE="ratio"    # center width mode
TRAIN_GAMMA_EXTREME=1.0            # extreme gamma
TRAIN_EXTREME_MODE="zscore"        # extreme mode
TRAIN_EXTREME_Q=0.70               # extreme quantile
TRAIN_EXTREME_SCALE=1.0            # extreme scale
TRAIN_NORMALIZE_WEIGHTS=true       # normalize loss weights
TRAIN_EPS=1e-6                     # loss epsilon
TRAIN_HIGH_FREQ_COMPONENT_LOSS=false  # enable high-freq component loss
TRAIN_HIGH_FREQ_CUTOFF_RATIO=0.5   # high-freq cutoff ratio
AUTO_NUM_VARS_FROM_DATA=true       # infer channel count from feed source and override BUILD_NUM_VARS/TRAIN_NUM_VARS

# ------------------------------------------------------------------------------
# CMIP6/WRF preprocessing inputs/outputs
# ------------------------------------------------------------------------------
level1_dir="$WORKDIR/level_1_data"                              # CMIP6 step1 output dir
level2_dir="$WORKDIR/level_2_data"                              # CMIP6 step2 output dir

# ------------------------------------------------------------------------------
# Preprocess parameters
# ------------------------------------------------------------------------------
imsize_x=100                 # CMIP6 crop width
imsize_y=100                 # CMIP6 crop height
frames=5                     # sequence length for CMIP6 step2
# CMIP6 variables used in step2
var_levels=(
  U10m V10m SST LANDMASK
  U28 V28 U05 V05
  T23 QVAPOR10 PHB10 PSFC
)
cmip6_num_vars=${#var_levels[@]}  # auto count for output naming
prefix="wrf_tropical_cyclone_track_${frames}f_${cmip6_num_vars}v_dataset"  # CMIP6 step2 output prefix

# ------------------------------------------------------------------------------
# WRF idealized inputs/outputs
# ------------------------------------------------------------------------------
ideal_wrf_root="$WORKDIR"                       # IDEALIZED already has its own WORKDIR branch
ideal_x_res="d01"                                                 # resolution string in filename
ideal_frames=480                     # frame length used in idealized step2
# Idealized variables to extract in step1
ideal_var_levels=(
  U10m V10m SST LANDMASK
  U14 V14 U03 V03
  T12 QVAPOR05 PHB05 PSFC
)
ideal_num_vars=${#ideal_var_levels[@]}  # auto count for idealized output naming
Idealized experiment folders to process
ideal_experiments=(
    exp_02km_m01 exp_02km_m02 exp_02km_m03 exp_02km_m04 exp_02km_m05
    exp_02km_m06 exp_02km_m07 exp_02km_m08 exp_02km_m09 exp_02km_m10
)
#ideal_experiments=(exp_02km_m10)
ideal_step1_out="${ideal_wrf_root}/level_1_data"                  # idealized step1 output dir (.nc)
ideal_step2_out="${ideal_wrf_root}/level_2_data"                  # idealized step2 output dir (.npy)
ideal_chunk_prefix="wrf_idealized_track_${ideal_frames}f_${ideal_num_vars}v_dataset"  # idealized step2 prefix
ideal_chunk_X="${ideal_step2_out}/${ideal_chunk_prefix}_X.npy"      # idealized step2 X output
ideal_chunk_ID="${ideal_step2_out}/${ideal_chunk_prefix}_exp_ids.npy"  # idealized step2 sample exp ids

# Split config by experiment IDs in idealized mode (works with step2 exp_ids output)
WRF_EXP_SPLIT="12348910+57+6"

# ------------------------------------------------------------------------------
# feed_data split parameters
# ------------------------------------------------------------------------------
FEED_TRAIN_FRAC=0.7          # fraction for train split (data_mode=0)
FEED_VAL_FRAC=0.2            # fraction for val split (data_mode=0)
FEED_TEST_FRAC=0.1           # fraction for test split (data_mode=0)
FEED_NUM_SEGMENTS=1          # segmented split count (data_mode=0)
FEED_STEP_IN="${MODEL_STEP_IN}"  # number of input frames in feed_data

# ------------------------------------------------------------------------------
# Training input selection (auto-switch based on DATASOURCE)
# ------------------------------------------------------------------------------
case "${datasource_key}" in
    CMIP6)
        #CMIP6_WRF=(1 1)      # run CMIP6 crop + CMIP6 npy
        WRF_IDEALIZED=(0 0)  # skip idealized pipeline
        FEED_MODE=0          # use npy_single
        FEED_DATA="${level2_dir}/${prefix}_X.npy"  # CMIP6 step2 output
        FEED_EXP_ARGS=()
        ;;
    IDEALIZED)
        CMIP6_WRF=(0 0)      # skip CMIP6 pipeline
        #WRF_IDEALIZED=(1 1)  # run idealized step1 + step2
        FEED_MODE=0          # use chunked idealized npy
        FEED_DATA="${ideal_chunk_X}"
        FEED_EXP_ARGS=(
          --splitter wrf_experiment
          --exp_split "${WRF_EXP_SPLIT}"
          --exp_ids_path "${ideal_chunk_ID}"
        )
        ;;
    *)
        echo "ERROR: DATASOURCE must be CMIP6 or IDEALIZED (got '${DATASOURCE}')" >&2
        exit 1
        ;;
esac

infer_num_vars_from_feed_source() {
    if [ "${FEED_MODE}" -eq 0 ]; then
        if [ ! -f "${FEED_DATA}" ]; then
            echo ""
            return 0
        fi
        python - "${FEED_DATA}" <<'PY'
import numpy as np
import sys

path = sys.argv[1]
arr = np.load(path, mmap_mode="r")
if arr.ndim != 5:
    raise SystemExit(f"Expected [N, T, H, W, C] for {path}, got {arr.shape}")
print(int(arr.shape[-1]))
PY
        return
    fi

    shopt -s nullglob
    wrf_files=( "${ideal_step1_out}"/x_"${ideal_x_res}"_*_exp_02km_m*.npy )
    shopt -u nullglob
    if [ "${#wrf_files[@]}" -eq 0 ]; then
        echo ""
        return 0
    fi

    python - "${wrf_files[0]}" <<'PY'
import numpy as np
import sys

path = sys.argv[1]
arr = np.load(path, mmap_mode="r")
if arr.ndim != 4:
    raise SystemExit(f"Expected [T, C, H, W] for {path}, got {arr.shape}")
print(int(arr.shape[1]))
PY
}

# ------------------------------------------------------------------------------
# CMIP6/WRF preprocessing
# ------------------------------------------------------------------------------
if [ "${CMIP6_WRF[0]}" -eq 1 ]; then
    mkdir -p "$level1_dir"

    shopt -s nullglob
    cmip6_tracks=( "$CMIP6_BASE_DIR"/$CMIP6_TRACK_GLOB )
    shopt -u nullglob

    if [ "${#cmip6_tracks[@]}" -eq 0 ]; then
        echo "ERROR: no track files matched ${CMIP6_BASE_DIR}/${CMIP6_TRACK_GLOB}" >&2
        exit 1
    fi

    cmip6_pairs=0
    for track_file in "${cmip6_tracks[@]}"; do
        track_base="$(basename "$track_file")"
        raw_name="${track_base%_track.txt}"
        raw_data_dir="${CMIP6_BASE_DIR}/${raw_name}"

        if [ ! -d "$raw_data_dir" ]; then
            echo "WARN: skip ${track_file}; missing raw dir ${raw_data_dir}" >&2
            continue
        fi

        echo "CMIP6 step1 pair: track=${track_file} data_dir=${raw_data_dir}"
        python "$PREPROCESS_DIR/WRF/CMIP6-WRF_step1.py" \
            --track_file "$track_file" \
            --data_dir   "$raw_data_dir" \
            --workdir    "$level1_dir" \
            --imsize_x   "$imsize_x" \
            --imsize_y   "$imsize_y"
        cmip6_pairs=$((cmip6_pairs + 1))
    done

    if [ "$cmip6_pairs" -eq 0 ]; then
        echo "ERROR: no valid CMIP6 track/raw-dir pairs found under ${CMIP6_BASE_DIR}" >&2
        exit 1
    fi
fi

if [ "${CMIP6_WRF[1]}" -eq 1 ]; then
    shopt -s nullglob
    level1_files=( "$level1_dir"/*.nc )
    shopt -u nullglob
    if [ "${#level1_files[@]}" -eq 0 ]; then
        echo "ERROR: no .nc files found in ${level1_dir}; skip CMIP6 step2" >&2
        exit 1
    fi

    python "$PREPROCESS_DIR/WRF/CMIP6-WRF_step2.py" \
        --indir       "$level1_dir" \
        --outdir      "$level2_dir" \
        --frames      "$frames" \
        --var_levels  "${var_levels[@]}" \
        --prefix      "$prefix"
fi

# ------------------------------------------------------------------------------
# WRF idealized pipeline
# ------------------------------------------------------------------------------
if [ "${WRF_IDEALIZED[0]}" -eq 1 ]; then
    python "$PREPROCESS_DIR/WRF/WRF-IDEALIZED_step1.py" \
        --data_dir "${ideal_wrf_base}" \
        --workdir "${ideal_step1_out}" \
        --imsize_x "${imsize_x}" \
        --imsize_y "${imsize_y}" \
        --experiment_wrf "${ideal_experiments[@]}" \
        --x_resolution "${ideal_x_res}"
fi

if [ "${WRF_IDEALIZED[1]}" -eq 1 ]; then
    shopt -s nullglob
    ideal_level1_files=( "$ideal_step1_out"/*.nc )
    shopt -u nullglob
    if [ "${#ideal_level1_files[@]}" -eq 0 ]; then
        echo "ERROR: no .nc files found in ${ideal_step1_out}; skip IDEALIZED step2" >&2
        exit 1
    fi

    python "$PREPROCESS_DIR/WRF/WRF-IDEALIZED_step2.py" \
        --indir "${ideal_step1_out}" \
        --outdir "${ideal_step2_out}" \
        --frames "${ideal_frames}" \
        --var_levels "${ideal_var_levels[@]}" \
        --prefix "${ideal_chunk_prefix}"
fi

# ------------------------------------------------------------------------------
# Training pipeline
# ------------------------------------------------------------------------------
if [ "${AUTO_NUM_VARS_FROM_DATA}" = true ]; then
    inferred_num_vars="$(infer_num_vars_from_feed_source)"
    if [ -n "${inferred_num_vars}" ]; then
        echo "Auto-detected num_vars=${inferred_num_vars} from feed source."
        BUILD_NUM_VARS="${inferred_num_vars}"
        TRAIN_NUM_VARS="${inferred_num_vars}"
    else
        echo "WARN: could not infer num_vars from feed source; keeping BUILD_NUM_VARS=${BUILD_NUM_VARS}, TRAIN_NUM_VARS=${TRAIN_NUM_VARS}" >&2
    fi
fi

if [ "${TRAINING[0]}" -eq 1 ]; then
    python -m cli.build_model \
      --architecture "${BUILD_ARCH}" \
      --output "${MODEL_YAML}" \
      --num_vars "${BUILD_NUM_VARS}" \
      --num_times "${BUILD_NUM_TIMES}" \
      --height "${BUILD_HEIGHT}" \
      --width "${BUILD_WIDTH}" \
      --num_blocks "${BUILD_NUM_BLOCKS}" \
      --film_zdim "${BUILD_FILM_ZDIM}" \
      --e_channels "${BUILD_E_CHANNELS}" \
      --hidden_factor "${BUILD_HIDDEN_FACTOR}" \
      --mlp_expansion_ratio "${BUILD_MLP_EXPANSION_RATIO}" \
      --stem_channels "${BUILD_STEM_CHANNELS}" \
      --checkpoint_dir "${CKPT_DIR}" \
      --checkpoint_name "${MODEL_NAME}" \
      --model_config_path "${MODEL_CFG}"
fi

if [ "${TRAINING[1]}" -eq 1 ]; then
    python -m cli.feed_data \
      --data_mode "${FEED_MODE}" \
      --data_path "${FEED_DATA}" \
      --wrf_dir "${ideal_step1_out}" \
      --x_resolution "${ideal_x_res}" \
      --imsize "${imsize_x}" \
      --train_frac "${FEED_TRAIN_FRAC}" \
      --val_frac "${FEED_VAL_FRAC}" \
      --test_frac "${FEED_TEST_FRAC}" \
      --num_segments "${FEED_NUM_SEGMENTS}" \
      --step_in "${FEED_STEP_IN}" \
      --temp "${TMP_DIR}" \
      "${FEED_EXP_ARGS[@]}"
fi

if [ "${TRAINING[2]}" -eq 1 ]; then
    srun --ntasks=4 --gpus=4 --gpus-per-task=1 python -m cli.train \
      --trainer "${TRAIN_TRAINER}" \
      --architecture "${BUILD_ARCH}" \
      --step_in "${TRAIN_STEP_IN}" \
      --batch_size "${TRAIN_BATCH_SIZE}" \
      --num_workers "${TRAIN_NUM_WORKERS}" \
      --pin_memory "${TRAIN_PIN_MEMORY}" \
      --num_vars "${TRAIN_NUM_VARS}" \
      --num_times "${TRAIN_NUM_TIMES}" \
      --height "${TRAIN_HEIGHT}" \
      --width "${TRAIN_WIDTH}" \
      --num_blocks "${TRAIN_NUM_BLOCKS}" \
      --rim "${TRAIN_RIM}" \
      --learning_rate "${TRAIN_LR}" \
      --weight_decay "${TRAIN_WD}" \
      --num_epochs "${TRAIN_EPOCHS}" \
      --film_zdim "${TRAIN_FILM_ZDIM}" \
      --checkpoint_dir "${CKPT_DIR}" \
      --checkpoint_name "trained-${MODEL_NAME}" \
      --e_channels "${TRAIN_E_CHANNELS}" \
      --hidden_factor "${TRAIN_HIDDEN_FACTOR}" \
      --mlp_expansion_ratio "${TRAIN_MLP_EXP_RATIO}" \
      --stem_channels "${TRAIN_STEM_CHANNELS}" \
      --L1_weight "${TRAIN_L1}" \
      --L2_weight "${TRAIN_L2}" \
      --Center_weight "${TRAIN_CENTER_WEIGHT}" \
      --Extreme_weight "${TRAIN_EXTREME_WEIGHT}" \
      --HighFreq_weight "${TRAIN_HIGHFREQ_WEIGHT}" \
      --loss_gamma_center "${TRAIN_GAMMA_CENTER}" \
      --loss_center_width "${TRAIN_CENTER_WIDTH}" \
      --loss_center_width_mode "${TRAIN_CENTER_WIDTH_MODE}" \
      --loss_gamma_extreme "${TRAIN_GAMMA_EXTREME}" \
      --loss_extreme_mode "${TRAIN_EXTREME_MODE}" \
      --loss_extreme_q "${TRAIN_EXTREME_Q}" \
      --loss_extreme_scale "${TRAIN_EXTREME_SCALE}" \
      --loss_normalize_weights "${TRAIN_NORMALIZE_WEIGHTS}" \
      --loss_eps "${TRAIN_EPS}" \
      --high_freq_component_loss "${TRAIN_HIGH_FREQ_COMPONENT_LOSS}" \
      --high_freq_cutoff_ratio "${TRAIN_HIGH_FREQ_CUTOFF_RATIO}" \
      --init_model_path "${CKPT_DIR}/${MODEL_NAME}" \
      --model_config_path "${MODEL_CFG}" \
      --temp "${TMP_DIR}"
fi

echo "All requested steps completed."
