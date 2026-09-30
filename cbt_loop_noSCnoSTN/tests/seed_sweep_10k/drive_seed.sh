#!/bin/bash
# train one seed to 10k with the (now dynamic-adenosine) defaults, then run the full
# testing_script suite on the resulting checkpoint into its own folder.
set -e
SD=$1
cd "$(dirname "$0")"
echo "=== seed $SD: training 10000 iters ==="
python -u ../../train_supervised_thal.py --iters 10000 --seed "$SD" --tag "seed${SD}_10k"
echo "=== seed $SD: testing_script ==="
python -u run_testing_script.py "params_supervised_thal_seed${SD}_10k.pkl"
echo "=== seed $SD: DONE ==="
