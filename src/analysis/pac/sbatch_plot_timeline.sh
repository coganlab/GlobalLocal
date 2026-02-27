#!/bin/bash
#SBATCH --output=out/aligned_svm_ncv/slurm_%j.out
#SBATCH -e out/aligned_svm_ncv/slurm_%j.err
#SBATCH -p common,scavenger,coganlab-gpu
#SBATCH -c 50
#SBATCH --mem=200G
#SBATCH --time=24:00:00

default_subj=("D0121")
default_region="lpfc"
default_conditions=("stimulus_c75" "stimulus_c25" "stimulus_i75" "stimulus_i25")
default_time_start="-1.0"
default_time_end="1.5"
default_window_width="0.5"
default_time_step="0.5"
default_out_fig="cond_timecourse_sig.png"
default_out_csv="cond_timecourse_sig_summary.csv"
#default_sig_pairs_dir="/hpc/home/$USER/coganlab/$USER/GlobalLocal/src/analysis/pac/sig_pairs"


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

echo "Subjects: ${subj[*]}"
echo "Region: $region"
echo "Conditions: ${conditions[*]}"
echo "Time window: ${time_start} to ${time_end} (width ${window_width}, step ${time_step})"
echo "Sig pairs dir: $sig_pairs_dir"
echo "Out fig: $out_fig  Out csv: $out_csv"

# activate conda (assumes conda is available)
source "$(conda info --base)/etc/profile.d/conda.sh" >/dev/null 2>&1 || true
conda activate ieeg || true

PY_SCRIPT="/hpc/home/$USER/coganlab/$USER/GlobalLocal/src/analysis/pac/plot_timeline.py"

mkdir -p sig_pairs

# build and run python command
cmd=(python "$PY_SCRIPT" --subj "${subj[@]}" --region "$region" --conditions "${conditions[@]}" \
     --time_start "$time_start" --time_end "$time_end" --window_width "$window_width" --time_step "$time_step" \
     --sig_pairs_dir "$sig_pairs_dir" --out_fig "$out_fig" --out_csv "$out_csv")

echo "Running: ${cmd[*]}"
"${cmd[@]}"
exit $?
