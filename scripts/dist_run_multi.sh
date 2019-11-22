#!/usr/bin/env bash

python ./scripts/launch.py \
    --nnodes "$1" --node_rank "$2" --master_addr "$3" --nproc_per_node "$4" \
    "$5" --cfg "$6" --model-dir "$7"