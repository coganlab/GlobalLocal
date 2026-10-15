#!/bin/bash
# Submit the coding-axes analysis (docs/analysis_plans.md, subspace plan §3.1-3.4, §4):
# per-electrode congruency and switch-type effects in every A4 decoding window on
# shared split halves, the cosine between them, its ceiling, shuffle null and
# participant bootstrap; the subspace test (principal angles, alignment index);
# the condition trajectories; and the figures. Run from dcc_scripts/subspace/.
#
# Usage:
#   bash submit_coding_axes_dcc.sh                         # real data
#   REMOVE_MEAN=1 bash submit_coding_axes_dcc.sh           # overall-activity control (§3.3)
#   ROIS=occ bash submit_coding_axes_dcc.sh                # region control (§3.3)
#   RT_CONTROL=match bash submit_coding_axes_dcc.sh        # RT control (§3.3): adjust | match | random
#   bash submit_coding_axes_all.sh                         # the main run and every control
#   DATA_SOURCE=synthetic N_SPLITS=20 N_PERM=50 N_BOOT=200 bash submit_coding_axes_dcc.sh  # dry run

# ---------------------------------------------------------------------------
# Epochs file (high-gamma, rescaled). Match one you actually have on disk.
# ---------------------------------------------------------------------------
EPOCHS_ROOT_FILE="Stimulus_-1.0to1.5sec_decFactor_8_outliers_10_drop_and_nan_thresh_perc_5.0_70.0-150.0_Hz_padLength_1.5s_filterbank_hilbert_stat_func_ttest_zmax_20"

# Data source: 'real' loads epoched data; 'synthetic' validates the pipeline.
DATA_SOURCE=${DATA_SOURCE:-real}

# ---------------------------------------------------------------------------
# Window grid: WINDOW_SIZE-sample windows every STEP_SIZE samples tiling
# [WINDOWS_TMIN, WINDOWS_TMAX] s. 64 / 16 at 256 Hz from 0 s are the A4 windows.
# ---------------------------------------------------------------------------
WINDOWS_TMIN=${WINDOWS_TMIN:-0.0}
WINDOWS_TMAX=${WINDOWS_TMAX:-1.5}
WINDOW_SIZE=${WINDOW_SIZE:-64}
STEP_SIZE=${STEP_SIZE:-16}

REMOVE_MEAN=${REMOVE_MEAN:-0}                 # 1 = A4's remove_mean control
RT_CONTROL=${RT_CONTROL:-none}                # none | adjust | match | random
RT_MATCH_BINS=${RT_MATCH_BINS:-10}            # quantile RT bins (match / random)
ELECTRODES=${ELECTRODES:-all}                 # 'all' or 'sig'
# sbatch --export splits on commas, so a comma-containing ROIS reaches the job
# through `export` + --export=ALL instead (as in the segregation submit script).
export ROIS=${ROIS:-lpfc}

N_SPLITS=${N_SPLITS:-200}                     # split halves, real labels
N_PERM=${N_PERM:-1000}                        # shuffles per factor
N_NULL_SPLITS=${N_NULL_SPLITS:-20}            # split halves per shuffle
N_BOOT=${N_BOOT:-2000}                        # participant resamples
ALPHA=${ALPHA:-0.05}
SEED=${SEED:-0}

K_MAX=${K_MAX:-3}                             # subspace dimensions compared / tested
# Trajectory axis windows (s), 'tmin:tmax'; the first is the main figure. Holds
# commas, so it reaches the job through `export` + --export=ALL, as ROIS does.
export TRAJ_AXES=${TRAJ_AXES:-0.0:1.5,0.1:0.5,0.6:1.4}
TRAJ_FIRST=${TRAJ_FIRST:-congruency}          # axis kept whole: congruency | switch
mkdir -p out

echo "Submitting coding axes (source=$DATA_SOURCE, windows=[$WINDOWS_TMIN,$WINDOWS_TMAX]s w$WINDOW_SIZE s$STEP_SIZE, remove_mean=$REMOVE_MEAN, rt_control=$RT_CONTROL, rois=$ROIS, splits=$N_SPLITS, perm=${N_PERM}x$N_NULL_SPLITS, boot=$N_BOOT, seed=$SEED, k_max=$K_MAX, traj_axes=$TRAJ_AXES, traj_first=$TRAJ_FIRST)"
sbatch --job-name="coding_axes_${DATA_SOURCE}" \
    --export=ALL,EPOCHS_ROOT_FILE="$EPOCHS_ROOT_FILE",DATA_SOURCE="$DATA_SOURCE",WINDOWS_TMIN="$WINDOWS_TMIN",WINDOWS_TMAX="$WINDOWS_TMAX",WINDOW_SIZE="$WINDOW_SIZE",STEP_SIZE="$STEP_SIZE",REMOVE_MEAN="$REMOVE_MEAN",RT_CONTROL="$RT_CONTROL",RT_MATCH_BINS="$RT_MATCH_BINS",ELECTRODES="$ELECTRODES",N_SPLITS="$N_SPLITS",N_PERM="$N_PERM",N_NULL_SPLITS="$N_NULL_SPLITS",N_BOOT="$N_BOOT",ALPHA="$ALPHA",SEED="$SEED",K_MAX="$K_MAX",TRAJ_FIRST="$TRAJ_FIRST" \
    sbatch_coding_axes_dcc.sh
