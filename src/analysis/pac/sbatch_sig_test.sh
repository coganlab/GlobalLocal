#!/bin/bash
#SBATCH --output=out/aligned_svm_ncv/slurm_%j.out
#SBATCH -e out/aligned_svm_ncv/slurm_%j.err
#SBATCH -p common,scavenger,coganlab-gpu
#SBATCH -c 50
#SBATCH --mem=200G
#SBATCH --time=24:00:00



default_subjA=("D0063")
default_subjB=("D0063")


if [ $# -ge 1 ]; then
    
    IFS=' ' read -r -a subjA <<< "${1}"
else
    subjA=("${default_subjA[@]}")
fi

regionA=${2:-lpfc}
condA=${3:-stimulus_ci75}
tstartA=${4:-0.0}
tendA=${5:-0.5}

if [ $# -ge 6 ]; then

    IFS=' ' read -r -a subjB <<< "${6}"
else
    subjB=("${default_subjB[@]}")
fi

regionB=${7:-lpfc}
condB=${8:-stimulus_ci25}
tstartB=${9:-0.0}
tendB=${10:-0.5}

# Optional: enable permutation test using per-trial pkls and set number of permutations
# Usage: pass 11th arg as 'true' to enable, and 12th arg to set n_perm (default 200)
perm_trials_flag=${11:-true}
n_perm=${12:-200}

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ieeg # make sure this works

subjA_args=""
for subj in "${subjA[@]}"; do
    subjA_args="$subjA_args --subjA $subj"
done

subjB_args=""
for subj in "${subjB[@]}"; do
    subjB_args="$subjB_args --subjB $subj"
done


cmd=(python /hpc/home/$USER/coganlab/$USER/GlobalLocal/src/analysis/pac/sig_test.py)
cmd+=( $subjA_args --regionA "$regionA" --condA "$condA" )
cmd+=( $subjB_args --regionB "$regionB" --condB "$condB" )

# Generate windows from -1.0 to 1.5 with step 0.5 and append as repeated --windows args.
# This produces: -1.0:-0.5 -0.5:0.0 0.0:0.5 0.5:1.0 1.0:1.5
windows_str=$(python - <<'PY'
start=-1.0
end=1.5
step=0.5
cur=start
out=[]
while cur < end:
    out.append(f"{cur}:{cur+step}")
    cur = round(cur+step, 10)
print(" ".join(out))
PY
)

if [ -n "$windows_str" ]; then
    IFS=' ' read -r -a windows_arr <<< "$windows_str"
    for w in "${windows_arr[@]}"; do
        # Pass as single token --windows=VALUE so values beginning with '-' are not treated as options
        cmd+=( "--windows=${w}" )
    done
else
    # fallback to single-window behavior (previous behavior)
    cmd+=( --tstartA "$tstartA" --tendA "$tendA" )
    cmd+=( --tstartB "$tstartB" --tendB "$tendB" )
fi

if [ "$perm_trials_flag" = "true" ] || [ "$perm_trials_flag" = "1" ]; then
    echo "Including --perm_trials --n_perm ${n_perm} in command"
    cmd+=( --perm_trials --n_perm ${n_perm} )
fi

"${cmd[@]}"
