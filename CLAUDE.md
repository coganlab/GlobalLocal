# CLAUDE.md

Guidance for Claude Code (and other coding agents) working in this repository.

## Orientation

- `src/analysis/` is the library, `dcc_scripts/` launches it on the Duke Compute
  Cluster (DCC), `tests/` checks it, and `docs/` explains the design. README
  Part 1 maps every file; `docs/analysis_guide.md` covers every analysis path.
- Cluster analyses follow a four-layer pattern (README §1.2): library →
  `<thing>_dcc.py` core → `run_<thing>_dcc.py` entry point → `sbatch_*.sh` /
  `submit_*.sh`.
- Tests use synthetic MNE fixtures (`tests/conftest.py`) and need no real data:
  `make test-fast` (or `pytest -m "not slow"`).

## Running anything on the DCC

DCC policy: login nodes are only for file management, job submission and light
tasks. AI agents, VS Code servers and interactive work must run on compute
nodes. Research Computing monitors login nodes for these processes and can block
job submission until they are cleaned up. Setup for humans is in
[`docs/dcc_ai_agents.md`](docs/dcc_ai_agents.md).

### Rules

1. Send every DCC command through the proxy alias: `ssh dcc-agent '<command>'`.
   The alias lands in a Slurm job on a compute node.
2. Never connect directly to `dcc-login.oit.duke.edu` or `dcc-login-0N`, and
   never start `claude`, `tmux`, VS Code, Jupyter or Python there.
3. If you are already running on the DCC, check `hostname` first. A
   `dcc-login-*` hostname means you are on a login node: stop and tell the user.
   On a compute node (any other DCC hostname), run commands directly instead of
   through `ssh dcc-agent`.
4. The `dcc-agent` job is small. Real analyses go through `sbatch` (the
   `submit_*.sh` / `sbatch_*.sh` scripts), not inside the agent job. Quick
   checks (imports, `--help`, small tests) are fine in the agent job.
5. If `ssh dcc-agent` isn't available (a claude.ai/code cloud session, or a
   machine without the alias), don't try another host. Write out the commands
   for the user to run.

### Paths and environment

| What | Where |
|---|---|
| Repo clone | `/hpc/home/$USER/coganlab/$USER/GlobalLocal` |
| BIDS data | `/cwork/$USER/BIDS-1.1_GlobalLocal/` |
| Conda env | `ieeg` |
| Results | `dcc_scripts/<area>/results/` and the BIDS `derivatives/` tree |

Single-quote remote commands so `$USER` and `$(...)` expand on the DCC, not on
your machine. Non-interactive shells don't load conda on their own, and
`sbatch` passes the submitting shell's environment to the job, so start every
command with this preamble:

```bash
ssh dcc-agent 'cd /hpc/home/$USER/coganlab/$USER/GlobalLocal \
  && source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate ieeg \
  && <command>'
```

If `conda` isn't found, point the user to "Troubleshooting" in
`docs/dcc_ai_agents.md` rather than hard-coding paths.

### Common commands (after the preamble)

- **Sync code:** `git pull --ff-only`. Make code changes in a local clone or
  branch and push them; don't edit the DCC clone in place unless the user asks.
  If the pull isn't a fast-forward, stop and ask.
- **Submit:** `cd dcc_scripts/<area> && bash submit_<thing>.sh`. Submit scripts
  call their `sbatch_*.sh` by relative path and SLURM writes logs to the
  `--output` path relative to that directory, so always `cd` first. Read the
  submit script's arrays and environment-variable overrides before running it:
  one call can submit dozens of jobs.
- **Monitor:** `squeue -u $USER`;
  `sacct -j <jobid> --format=JobID,JobName%40,State,Elapsed,MaxRSS,ExitCode`;
  `tail -n 100 <log>`.
- **Cancel:** `scancel <jobid>`, only for jobs you submitted in this session
  unless the user says otherwise.

### Be careful with

- **`/cwork` and `results/`:** these hold hours of cluster output. Don't delete,
  move or overwrite anything there without asking.
- **Patient data:** don't print raw iEEG arrays, EDFs or clinical files into the
  conversation. Read logs, summaries, array shapes and result tables instead.
- **Large fan-outs:** say how many jobs and how much CPU/memory a submit script
  will request before running it.
- **Git:** SLURM `.out`/`.err` logs and `.npz`/`.png`/`.csv` outputs are
  gitignored. Don't force-add them.
