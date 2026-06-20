#!/bin/bash
#SBATCH --output=out/aligned_svm_ncv/slurm_%j.out
#SBATCH -e out/aligned_svm_ncv/slurm_%j.err
#SBATCH -p common,scavenger,coganlab-gpu
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH --time=72:00:00

default_subj=("D0057" "D0059" "D0063" "D0065" "D0069" "D0071" "D0077" "D0090" "D0094" "D0100" "D0102" "D0103" "D0107A" "D0110" "D0116" "D0117" "D0121" )
default_region="lpfc"
default_conditions=("stimulus_c" "stimulus_i")
default_input_dir="/hpc/home/rl330/coganlab/rl330/GlobalLocal/src/analysis/pac/envcorr_timewindow"
default_outdir="/hpc/home/rl330/coganlab/rl330/GlobalLocal/src/analysis/pac/cluster_stats"

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

input_dir=${4:-$default_input_dir}
outdir=${5:-$default_outdir}

echo "Subjects: ${subj[*]}"
echo "Region: $region"
echo "Conditions: ${conditions[*]}"
echo "Input dir: $input_dir"
echo "Output dir: $outdir"

source $(conda info --base)/etc/profile.d/conda.sh

conda activate ieeg # make sure this works

mkdir -p "$outdir"

python /hpc/home/$USER/coganlab/$USER/GlobalLocal/src/analysis/pac/env_plot.py \
    --subj "${subj[@]}" \
    --conditions "${conditions[@]}" \
    --input_dir "$input_dir" \
    --time_start -1.0 \
    --time_end 1.5 \
    --time_step 0.5 \
    --window_width 0.5 \
    --region "$region" \
    --outdir "$outdir" \
    --verbose

