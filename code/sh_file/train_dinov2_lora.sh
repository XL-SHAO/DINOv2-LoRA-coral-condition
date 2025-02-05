#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -jc gs-container_g16.72h
#$ -ac d=aip-cuda-12.0.1-blender-2,d_shm=256G
#$ -N train_dinov2lora_vitg_LR_3e-4_lineardecay_rank_24_spatial_transfer

. ~/net.sh

/home/songjian/anaconda3/envs/mmseg/bin/python script/train_dinov2_lora.py  \
    --batch_size 16 \
    --vit_type vitg \
    --learning_rate 3e-4 \
    --freeze_vit true \
    --low_rank 24 \
    --max_iters 480000 \
    --weight_decay 5e-4