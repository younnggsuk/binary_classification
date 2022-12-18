#!/bin/bash
torchrun --nproc_per_node=2 train.py \
    --exp_name "baseline_resnet50" \
    --seed 1111 \
    --root_dir "datas" \
    --log_dir "logs" \
    --data_list_json "datas/data_split.json" \
    --crop_size 448 \
    --batch_size_per_gpu 32 \
    --num_workers 8 \
    --arch "resnet50" \
    --in_chans 3 \
    --num_classes 2 \
    --optimizer "adamw" \
    --lr_scheduler "linear_warmup_cosine_anneling" \
    --lr 0.001 \
    --max_epochs 100 \
    --early_stop 20 \
    --save_best true \
    --saveckp_freq 10