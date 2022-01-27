
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}


#!/usr/bin/env bash


PYTHON=${PYTHON:-"python"}

CONFIG='/home/joon/fun/mmdetection/configs/yolox/custom_yolox_s_8x8_coco.py'
WORKING_DIR='/media/volume4/joon_fun/tensorflow-great-barrier-reef/working/220128'
MODEL_DIR='/media/volume4/joon_fun/tensorflow-great-barrier-reef/working/220128/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'

GPUS=4

CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
    $(dirname "$0")/train.py $CONFIG --work-dir $WORKING_DIR --load-from $MODEL_DIR --seed 1234 --deterministic --launcher pytorch ${@:3}
