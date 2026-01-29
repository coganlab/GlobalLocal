#!/bin/bash
#SBATCH --output=out/aligned_svm_ncv/slurm_%j.out
#SBATCH -e out/aligned_svm_ncv/slurm_%j.err
#SBATCH -p common,scavenger,coganlab-gpu
#SBATCH -c 50
#SBATCH --mem=200G
#SBATCH --time=24:00:00

default_subj=("D0057")
default_region="lpfc"
default_conditions=("stimulus_ci75" "stimulus_ci25")
default_time_start="-1.0"
default_time_end="1.5"
default_window_width="0.5"
default_time_step="0.5"
default_out_fig="cond_timecourse_sig.png"
default_out_csv="cond_timecourse_sig_summary.csv"
default_sig_pairs_dir="/hpc/home/$USER/coganlab/$USER/GlobalLocal/src/analysis/pac/sig_pairs"

# Parse arguments
if [ $# -ge 1 ]; then
    IFS=' ' read -r -a subj <<< "${1}"
else
    subj=("${default_subj[@]}")
fi

region=${2:-$default_region}

if [ $# -ge 3 ]; then
    IFS=' ' read -r -a conditions <<< "${3}"
else
    conditions=("${default_conditions[@]}")
fi

time_start=${4:-$default_time_start}
time_end=${5:-$default_time_end}
window_width=${6:-$default_window_width}
time_step=${7:-$default_time_step}
sig_pairs_dir=${8:-$default_sig_pairs_dir}

out_fig=${9:-$default_out_fig}
out_csv=${10:-$default_out_csv}


echo "=========================================="
echo "Plot Conditions Timecourse (sig pairs only)"
echo "=========================================="
echo "Subjects: ${subj[*]}"
echo "Region: $region"
echo "Conditions: ${conditions[*]}"
echo "Time window: start=$time_start end=$time_end width=$window_width step=$time_step"
echo "Sig pairs dir: $sig_pairs_dir"
echo "Output figure: $out_fig"
echo "Output CSV: $out_csv"
if [ -n "$verbose_flag" ]; then
    echo "Verbose: enabled"
fi
echo "=========================================="

# Relax common ulimits to avoid "Too many open files" etc.
# These are per-job settings; remove or reduce if your cluster forbids 'unlimited'.
ulimit -n 65536 || true
ulimit -v unlimited || true
ulimit -s unlimited || true

# Activate conda environment robustly
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -d "$HOME/miniconda3/etc/profile.d" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -d "$HOME/anaconda3/etc/profile.d" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Warning: conda not found in PATH or common locations. Proceeding without conda activation."
fi

# activate environment (change name if needed)
conda activate ieeg

# Path to the plotting script (adjust if necessary)
PY_SCRIPT="/hpc/home/$USER/coganlab/$USER/GlobalLocal/src/analysis/pac/plot_timeline.py"

# Build python command
cmd=(python "$PY_SCRIPT")
cmd+=( --subj "${subj[@]}" )
cmd+=( --region "$region" )
cmd+=( --conditions "${conditions[@]}" )
cmd+=( --time_start "$time_start" --time_end "$time_end" --window_width "$window_width" --time_step "$time_step" )
cmd+=( --sig_pairs_dir "$sig_pairs_dir" )
cmd+=( --out_fig "$out_fig" --out_csv "$out_csv" )
if [ -n "$verbose_flag" ]; then
    cmd+=( --verbose )
fi

echo ""
echo "Executing command:"
printf '%q ' "${cmd[@]}"
echo -e "\n"

# Run
"${cmd[@]}"