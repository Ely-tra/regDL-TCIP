#!/bin/bash -l
# ==============================================================================
#
# Description:
#   Unified SLURM batch for CMIP6/WRF preprocessing, optional idealized WRF
#   steps, and training with canonicalizer-based data feeding.
#
# ==============================================================================

#SBATCH -N 1
#SBATCH -t 10:00:00
#SBATCH -J TCNN-ctl
#SBATCH -p gpu
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH -A r00043
#SBATCH --mem=200G

module load python/gpu/3.10.10
set -x

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREPROCESS_DIR="/N/slate/kmluong/PROJECT2"
cd "$ROOT"

# ------------------------------------------------------------------------------
# Flags (1=run, 0=skip)
# ------------------------------------------------------------------------------
CMIP6_WRF=(1 1)        # [0]=step1 (crop), [1]=step2 (npy)
WRF_IDEALIZED=(1)      # [0]=idealized pipeline (placeholder)
TRAINING=(1 1 1)       # [0]=build, [1]=feed, [2]=train

# ------------------------------------------------------------------------------
# CMIP6/WRF preprocessing inputs/outputs
# ------------------------------------------------------------------------------
track_file='/N/project/hurricane-deep-learning/data/cmip6/ssp245_2080_2100_track.txt'
raw_data_dir='/N/project/hurricane-deep-learning/data/cmip6/ssp245_2080_2100/'
level1_dir='/N/slate/kmluong/PROJECT2/level_1_data_ssp245'
level2_dir='/N/slate/kmluong/PROJECT2/level_2_data_ssp245'

# ------------------------------------------------------------------------------
# Preprocess parameters
# ------------------------------------------------------------------------------
imsize_x=100
imsize_y=100
frames=5
var_levels=(
  U10m V10m SST LANDMASK
  U28 V28 U05 V05
  T23 QVAPOR10 PHB10
)
prefix="wrf_tropical_cyclone_track_${frames}_dataset"

# ------------------------------------------------------------------------------
# WRF idealized inputs/outputs
# ------------------------------------------------------------------------------
ideal_wrf_base="/N/project/Typhoon-deep-learning/data/tc-wrf/"
ideal_wrf_root="/N/slate/kmluong/PROJECT2/WRF"
ideal_x_res="d01"
ideal_var_levels=(
  U10m V10m SST LANDMASK
  U14 V14 U03 V03
  T12 QVAPOR05 PHB05
)
ideal_experiments=(
  exp_02km_m01 exp_02km_m02 exp_02km_m03 exp_02km_m04 exp_02km_m05
  exp_02km_m06 exp_02km_m07 exp_02km_m08 exp_02km_m09 exp_02km_m10
)

# ------------------------------------------------------------------------------
# Training inputs/outputs
# ------------------------------------------------------------------------------
DATA_PATH="/N/slate/kmluong/PROJECT2/level_2_data/wrf_tropical_cyclone_track_5_dataset_X.npy"
TMP_DIR="/N/slate/kmluong/PROJECT2/tmp"
CKPT_DIR="/N/slate/kmluong/PROJECT2/checkpoints"
MODEL_NAME="AFNO-TCP.pt"
MODEL_CFG="${CKPT_DIR}/AFNO_config1.json"
MODEL_YAML="${ROOT}/configs/model/AFNO_v1.yaml"
WRF_DIR="/N/slate/kmluong/PROJECT2/WRF/wrf_data"

# ------------------------------------------------------------------------------
# CMIP6/WRF preprocessing
# ------------------------------------------------------------------------------
if [ "${CMIP6_WRF[0]}" -eq 1 ]; then
    pushd "$PREPROCESS_DIR" >/dev/null
    python step1.py \
        --track_file "$track_file" \
        --data_dir   "$raw_data_dir" \
        --workdir    "$level1_dir" \
        --imsize_x   $imsize_x \
        --imsize_y   $imsize_y
    popd >/dev/null
fi

if [ "${CMIP6_WRF[1]}" -eq 1 ]; then
    pushd "$PREPROCESS_DIR" >/dev/null
    python step2.py \
        --indir       "$level1_dir" \
        --outdir      "$level2_dir" \
        --frames      $frames \
        --var_levels  "${var_levels[@]}" \
        --prefix      "$prefix"
    popd >/dev/null
fi

# ------------------------------------------------------------------------------
# WRF idealized pipeline placeholder
# ------------------------------------------------------------------------------
if [ "${WRF_IDEALIZED[0]}" -eq 1 ]; then
    python "${ROOT}/preprocess/WRF/WRF-IDEALIZED.py" \
        --imsize_variables "${imsize_x}" "${imsize_y}" \
        --root "${ideal_wrf_root}" \
        --wrf_base "${ideal_wrf_base}" \
        --var_levels "${ideal_var_levels[@]}" \
        --experiment_wrf "${ideal_experiments[@]}" \
        --X_resolution "${ideal_x_res}"
fi

# ------------------------------------------------------------------------------
# Training pipeline
# ------------------------------------------------------------------------------
if [ "${TRAINING[0]}" -eq 1 ]; then
    python -m cli.build_model \
      --architecture afno_v1 \
      --output "${MODEL_YAML}" \
      --num_vars 11 \
      --num_times 3 \
      --height 100 \
      --width 100 \
      --num_blocks 6 \
      --film_zdim 128 \
      --e_channels 256 \
      --hidden_factor 8 \
      --mlp_expansion_ratio 4 \
      --stem_channels 256 \
      --checkpoint_dir "${CKPT_DIR}" \
      --checkpoint_name "${MODEL_NAME}" \
      --model_config_path "${MODEL_CFG}"
fi

if [ "${TRAINING[1]}" -eq 1 ]; then
    python -m cli.feed_data \
      --data_mode 1 \
      --data_path "${DATA_PATH}" \
      --wrf_dir "${WRF_DIR}" \
      --x_resolution d01 \
      --imsize 100 \
      --exp_split 12348910+57+6 \
      --train_frac 0.7 \
      --val_frac 0.2 \
      --test_frac 0.1 \
      --num_segments 1 \
      --step_in 3 \
      --temp "${TMP_DIR}"
fi

if [ "${TRAINING[2]}" -eq 1 ]; then
    srun --ntasks=4 --gpus=4 --gpus-per-task=1 python -m cli.train \
      --trainer afno_tcp_v1 \
      --step_in 3 \
      --batch_size 8 \
      --num_workers 0 \
      --pin_memory true \
      --num_vars 11 \
      --num_times 3 \
      --height 100 \
      --width 100 \
      --num_blocks 8 \
      --rim 1 \
      --learning_rate 1e-4 \
      --weight_decay 1e-4 \
      --num_epochs 300 \
      --film_zdim 64 \
      --checkpoint_dir "${CKPT_DIR}" \
      --checkpoint_name "trained-${MODEL_NAME}" \
      --e_channels 256 \
      --hidden_factor 8 \
      --mlp_expansion_ratio 4 \
      --stem_channels 256 \
      --L1_weight 1.0 \
      --L2_weight 0.0 \
      --Center_weight 0.0 \
      --Extreme_weight 0.5 \
      --HighFreq_weight 1.0 \
      --loss_gamma_center 1.0 \
      --loss_center_width 0.25 \
      --loss_center_width_mode ratio \
      --loss_gamma_extreme 1.0 \
      --loss_extreme_mode zscore \
      --loss_extreme_q 0.70 \
      --loss_extreme_scale 1.0 \
      --loss_normalize_weights true \
      --loss_eps 1e-6 \
      --high_freq_component_loss false \
      --high_freq_cutoff_ratio 0.5 \
      --init_model_path "${CKPT_DIR}/${MODEL_NAME}" \
      --model_config_path "${MODEL_CFG}" \
      --temp "${TMP_DIR}"
fi

echo "All requested steps completed."
