#!/bin/sh

WANDB_PROJECT='template'
WANDB_ENTITY='youngerous'

CKPT_ROOT='/repo/pytorch-nlp-wandb-hydra-template/src/checkpoints/'
EPOCH=1
BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEP=1
EARLY_STOP_TOLERANCE=10
LR='5e-5'
GPU='ddp'
AMP='True'

python src/main.py\
    ckpt_root=$CKPT_ROOT\
    epoch=$EPOCH\
    batch_size=$BATCH_SIZE\
    gradient_accumulation_step=$GRADIENT_ACCUMULATION_STEP\
    early_stop_tolerance=$EARLY_STOP_TOLERANCE\
    lr=$LR\
    amp=$AMP\
    +wandb.project=$WANDB_PROJECT\
    +wandb.entity=$WANDB_ENTITY\
    +gpu=$GPU\

