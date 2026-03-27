__doc__="
Dump the tumx pane results into a directory an agent can read it.
"

OUT_DPATH=/data/joncrall/dvc-repos/shitspotter_expt_dvc/tmux-logs
mkdir -p "$OUT_DPATH"
TIMESTAMP=$(date +"%Y-%m-%dT%H-%M-%S")
FPATH=$OUT_DPATH/tmux-logs-$TIMESTAMP.log
tmux capture-pane -pS - > "$FPATH"
echo "FPATH = $FPATH"
