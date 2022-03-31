#!bin/bash
export EXPR_ID=AI_test
export DATA_DIR=/home/volume/NAVE/cifar10
export CHECKPOINT_DIR=/home/volume/torch/NVAE/check_point
export CODE_DIR=/home/volume/NVAE
python evaluate.py --checkpoint $CHECKPOINT_DIR/eval-$EXPR_ID/checkpoint.pt --eval_mode=sample --temp=1 --readjust_bn --save=/home/volume/torch/NVAE/sampling --world_size=1 --readjust_bn
