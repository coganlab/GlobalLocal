# Running AI agents and VS Code against the DCC

How to use Claude Code (or Codex, OpenCode, VS Code Remote-SSH) with the Duke
Compute Cluster without running anything on the shared login nodes. The rules
agents follow in this repo are in [`CLAUDE.md`](../CLAUDE.md); this page is the
human setup.

> **The rule:** DCC login nodes are only for file management, job submission
> and light tasks. AI agents, VS Code servers and interactive development must
> run on compute nodes. Research Computing monitors for processes like
> `.vscode-server/... agent host`, `claude`, `.claude/remote/...` and `codex` on
> login nodes, and can block your job submission until they are gone.

The supported way to do this is an SSH host alias backed by Duke's
`dcc-ssh-proxy`. Duke's own pages are the source of truth for the exact config:

- [Access via AI Agents](https://oit-rc.pages.oit.duke.edu/rcsupportdocs/dcc/ai-access/)
- [Access via VS Code](https://oit-rc.pages.oit.duke.edu/rcsupportdocs/dcc/vscode/)

## How the proxy works

```
your machine ── ssh dcc-agent ──► dcc-login        (only runs dcc-ssh-proxy.sh)
                                      │ finds or starts your Slurm job "agent"
                                      ▼
                                 compute node  ◄── your commands run here
```

- The first `ssh dcc-agent` starts a Slurm job for that profile. Later
  connections reuse the same job, so you land on the same node.
- The job ends when it reaches its wall time (`--time`, 8 hours by default), or
  `--idle-min` minutes after your last connection closes.
- Each alias has its own `--name`, so an agent job (`dcc-agent`) and a VS Code
  job (e.g. `dcc-cpu`) can run side by side without sharing an allocation.

## One-time setup

1. **SSH key.** Set one up so you aren't typing your NetID password for every
   connection: [DCC SSH keys](https://oit-rc.pages.oit.duke.edu/rcsupportdocs/dcc/login/#ssh-keys).

2. **Socket directory.** The aliases reuse one SSH connection through a control
   socket. If the directory is missing, every connection fails immediately:

   ```bash
   mkdir -p ~/.ssh/sockets && chmod 700 ~/.ssh/sockets
   ```

3. **Host block.** Copy the `dcc-agent` block from Duke's
   [Access via AI Agents](https://oit-rc.pages.oit.duke.edu/rcsupportdocs/dcc/ai-access/)
   page into `~/.ssh/config` on the machine that will run the agent. Its key
   pieces are:

   - a `ProxyCommand` that runs `dcc-ssh-proxy.sh --name agent --partition scavenger,common`
     on `dcc-login.oit.duke.edu` (with a longer idle timeout than the VS Code profiles)
   - `ServerAliveInterval 30`
   - `ControlMaster auto`, `ControlPath ~/.ssh/sockets/%r@%h-%p`, `ControlPersist 30m`

   Use the alias name `dcc-agent`: `CLAUDE.md` refers to it by that name.
   On Windows, the config must live in whichever environment runs the agent.
   WSL has its own `~/.ssh/config`, separate from `C:\Users\<you>\.ssh\config`.

4. **Test it.** You should see a compute-node hostname (not `dcc-login-0N`), and
   the agent job in the queue:

   ```bash
   ssh dcc-agent 'hostname; squeue -u $USER'
   ```

5. **Check conda works in non-interactive shells.** Agents run commands as
   `ssh dcc-agent '<command>'`, which doesn't load your interactive shell setup:

   ```bash
   ssh dcc-agent 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate ieeg && which python sbatch'
   ```

   If this says `conda: command not found`, see [Troubleshooting](#troubleshooting).

6. **Clean up old login-node processes.** If you previously connected VS Code or
   an agent straight to `dcc-login`, check every login node (01–05) for leftover
   processes. From a login node:

   ```bash
   ssh dcc-login.oit.duke.edu
   for n in 01 02 03 04 05; do
     echo "== dcc-login-$n"
     ssh dcc-login-$n 'ps -u $USER -o pid,lstart,cmd | grep -E "vscode-server|claude|codex" | grep -v grep'
   done
   # then, for each PID listed under dcc-login-0N:
   ssh dcc-login-0N 'kill <pid>'
   ```

   Also delete or rename any VS Code Remote-SSH host that points straight at
   `dcc-login.oit.duke.edu`, so VS Code doesn't reconnect there.

## Ways to work

### A. Claude Code on your machine, commands over SSH (recommended)

This is the setup Duke documents. Claude runs on your laptop or desktop and
only its commands go to the DCC. Run it in your **local** clone:

```bash
cd ~/path/to/GlobalLocal
claude
```

`CLAUDE.md` tells it to send DCC work through the alias. A typical loop:

```bash
# edit locally, commit, push; then on the DCC:
ssh dcc-agent 'cd /hpc/home/$USER/coganlab/$USER/GlobalLocal && git pull --ff-only'
ssh dcc-agent 'cd /hpc/home/$USER/coganlab/$USER/GlobalLocal \
  && source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate ieeg \
  && cd dcc_scripts/decoding && bash submit_specific_conditions_decoding_dcc.sh'
ssh dcc-agent 'squeue -u $USER'
```

- Because the SSH connection is reused, repeated calls don't ask for 2FA again.
- Nothing is left running on the DCC: the proxy job ends on its own after the
  idle timeout.
- Your machine has to stay on while Claude works.

### B. Claude Code inside the compute job

Use this when you want to start a long task, walk away, and steer it from your
phone or [claude.ai/code](https://claude.ai/code) with Remote Control.

```bash
ssh dcc-agent
hostname                      # must NOT be dcc-login-0N
tmux new -s claude
conda activate ieeg
cd /hpc/home/$USER/coganlab/$USER/GlobalLocal
claude --remote-control "GlobalLocal DCC"   # or plain `claude`
```

Detach with `Ctrl-b d`. To come back, run `ssh dcc-agent` (it reuses the job,
so you land on the same node) and then `tmux attach -t claude`.

Caveats:

- **tmux doesn't keep the job alive.** After you disconnect, the idle watchdog
  ends the job after `--idle-min`, and it always ends at `--time`. Claude dies
  with it. For this option, use a separate alias with a longer `--time` and a
  larger `--idle-min` (see [Tuning the job](#tuning-the-job)).
- **Avoid `scavenger` for this alias.** Scavenger jobs can be preempted, which
  kills the session mid-task. Use `--partition common`.
- **Outbound HTTPS.** Claude needs to reach `api.anthropic.com` from the compute
  node. Check with `ssh dcc-agent 'curl -sI https://api.anthropic.com | head -1'`.
- **Install once:** `curl -fsSL https://claude.ai/install.sh | bash` inside the
  job (installs to `~/.local/bin`, which is on the shared home filesystem). On
  first run, `/login` prints a URL to open in your local browser.

### C. Claude Code on the web (claude.ai/code)

Cloud sessions run in a container that can't reach the DCC and has none of the
data. Use them for code changes, synthetic-data tests (`make test-fast`) and
PRs. Then pull the branch on the DCC with option A or B to run it for real.
Don't put your DCC SSH key in a cloud environment.

### D. VS Code

In Remote-SSH, connect to a proxy alias from Duke's
[Access via VS Code](https://oit-rc.pages.oit.duke.edu/rcsupportdocs/dcc/vscode/)
page (e.g. `dcc-cpu`), **not** to `dcc-login.oit.duke.edu`. The VS Code server,
and the Claude Code extension if you use it, then run on the compute node. The
older OnDemand + `sshhost` route in README Part 4 also lands on a compute node.

## Tuning the job

The flags go on the `dcc-ssh-proxy.sh` line in the alias's `ProxyCommand`. Check
Duke's page for the full list.

| Flag | What it does |
|---|---|
| `--name agent` | Which job this alias reuses. Give each alias its own name |
| `--partition scavenger,common` | Where the job runs. Drop `scavenger` if preemption would hurt (option B) |
| `--mem <size>` | Memory for the job, e.g. `--mem 16G`. Keep the agent job small; real work goes through `sbatch` |
| `--time` | Wall time. 8 hours by default |
| `--idle-min` | Minutes after the last connection closes before the job is ended. `0` disables idle cleanup; then you must `scancel` the job yourself |

The agent job counts against your account's limits like any other job. To see
yours:

```bash
ssh dcc-agent 'sacctmgr show assoc u=$USER format=user%8,account%32,maxjobs%7,grptres%35,maxwall'
```

## Ending the job

```bash
ssh dcc-agent 'squeue -u $USER'      # find the job named for the alias
ssh dcc-agent 'scancel <jobid>'      # the connection drops as the job ends
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| Every `ssh dcc-agent` fails immediately | `~/.ssh/sockets` is missing. Run `mkdir -p ~/.ssh/sockets && chmod 700 ~/.ssh/sockets` |
| Repeated 2FA prompts | Connection reuse isn't working. Check the `Control*` lines and the socket directory. The built-in Windows OpenSSH client doesn't support `ControlMaster`; use WSL or follow the Windows notes on Duke's page |
| `conda: command not found` over `ssh dcc-agent '...'` | Your `~/.bashrc` starts with `[[ $- != *i* ]] && return` (README Part 4), so non-interactive shells exit before the `# >>> conda initialize >>>` block runs. Move that block above the guard line. It prints nothing, so it won't break VS Code or `scp` |
| Jobs submitted from the agent fail at `conda activate` | Same cause: `sbatch` copies the submitting shell's environment. Activate conda in the same command before calling `submit_*.sh` (see the preamble in `CLAUDE.md`) |
| `sbatch` accepted but nothing runs, or jobs won't submit | Check `squeue -u $USER` for the pending reason and the `sacctmgr` command above. `MaxJobs` of 0 means your account was restricted. Email [rescomputing@duke.edu](mailto:rescomputing@duke.edu) |
| SLURM log missing | Submit scripts must be run from their own directory (`cd dcc_scripts/<area>`), and the `--output` directory in the `sbatch_*.sh` header must exist |
| Option B session vanished | The job hit `--time` or `--idle-min`, or was preempted on `scavenger`. Check `sacct -u $USER -S today` |

## Data and privacy

Anything an agent reads (file contents, command output) is sent to the model
provider as part of the conversation. With Remote Control, the transcript is
also stored on Anthropic's servers. Check what Duke and the IRB allow for this
data before pointing an agent at it. `CLAUDE.md` tells Claude not to print raw
iEEG data, EDFs or clinical files, and to work from logs, summaries and result
tables instead.
