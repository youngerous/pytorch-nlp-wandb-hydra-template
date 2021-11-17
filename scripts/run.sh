#!/bin/sh

EPOCH=10
BATCH_SIZE=8
LR=5e-5
WORKERS=0

python src/main.py\
    --epoch $EPOCH\
    --batch-size $BATCH_SIZE\
    --lr $LR\
    --workers $WORKERS\
    --amp\
    --distributed\
