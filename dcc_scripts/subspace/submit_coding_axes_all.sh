#!/bin/bash
# Submit the coding-axes main run and every §3.3 control, one job each
# (docs/coding_axes.md). Any setting exported or prefixed here applies to all
# six, e.g.  DATA_SOURCE=synthetic N_PERM=50 bash submit_coding_axes_all.sh
# When they have finished, compare them with plot_coding_axes_controls.py.
cd "$(dirname "$0")" || exit 1

bash submit_coding_axes_dcc.sh                              # main result
REMOVE_MEAN=1 bash submit_coding_axes_dcc.sh                # overall activity
RT_CONTROL=adjust bash submit_coding_axes_dcc.sh            # RT regressed out
RT_CONTROL=match bash submit_coding_axes_dcc.sh             # RT-matched trials...
RT_CONTROL=random bash submit_coding_axes_dcc.sh            # ...and their count-matched control
ROIS=occ bash submit_coding_axes_dcc.sh                     # region
