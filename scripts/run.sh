#!/bin/sh

EPOCH=1
BATCH_SIZE=8
LR='5e-5'
GPU='dp'
AMP='True'

python src/main.py\
    epoch=$EPOCH\
    batch_size_per_gpu=$BATCH_SIZE\
    lr=$LR\
    amp=$AMP\
    +gpu=$GPU\

