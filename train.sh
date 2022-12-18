#!/bin/bash
torchrun --nproc_per_node=2 main.py \
    --exp_name "baseline_resnet50" \
    --seed 1111 \
    --root_dir "datas" \
    --log_dir "logs" \
    --data_list_json "datas/data_split.json" \
    --crop_size 448 \
    --batch_size_per_gpu 64 \
    --num_workers 8 \
    --arch "resnet50" \
    --in_chans 3 \
    --num_classes 2 \/home/ubuntu/ys/others/classification_backbone/logs/baseline_resnet50/checkpoints/checkpoint018_best.pth
    --optimizer "adamw" \
    --lr_scheduler "linear_warmup_cosine_anneling" \
    --lr 0.001 \
    --max_epochs 100 \
    --early_stop 40 \
    --save_best true \
    --saveckp_freq 10