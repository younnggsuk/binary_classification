#!/bin/bash
torchrun --nproc_per_node=2 test.py \
    --exp_name "baseline_resnet50_rgbd" \
    --root_dir "datas" \
    --result_dir "results" \
    --data_list_json "datas/data_split.json" \
    --crop_size 448 \
    --batch_size_per_gpu 16 \
    --num_workers 8 \
    --arch "resnet50" \
    --in_chans 4 \
    --num_classes 2 \
    --threshold 0.5 \
    --ckpt_path "logs/baseline_resnet50_rgbd/checkpoints/checkpoint_best.pth"