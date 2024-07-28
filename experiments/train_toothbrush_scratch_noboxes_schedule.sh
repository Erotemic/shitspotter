cmd_queue new "shitspotter_train_queue"

#cmd_queue submit "shitspotter_train_queue" -- bash ~/code/shitspotter/experiments/train_toothbrush_scratch_noboxes_v3.sh
#cmd_queue submit "shitspotter_train_queue" -- bash ~/code/shitspotter/experiments/train_toothbrush_scratch_noboxes_v4.sh
cmd_queue submit "shitspotter_train_queue" -- bash ~/code/shitspotter/experiments/train_toothbrush_scratch_noboxes_v5.sh
cmd_queue submit "shitspotter_train_queue" -- bash ~/code/shitspotter/experiments/train_toothbrush_scratch_noboxes_v6.sh
cmd_queue submit "shitspotter_train_queue" -- bash ~/code/shitspotter/experiments/train_toothbrush_scratch_noboxes_v7.sh
cmd_queue submit "shitspotter_train_queue" -- bash ~/code/shitspotter/experiments/train_toothbrush_scratch_noboxes_v8.sh


cmd_queue show "shitspotter_train_queue"


# Execute your queue.
cmd_queue run "shitspotter_train_queue" --backend=tmux --workers=2 --gpus="0,1"
