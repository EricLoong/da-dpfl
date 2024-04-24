#!/bin/bash
#python3.7 /Users/ericlong/PycharmProjects/ppfl/fedml_adpfl/main_adpfl.py --model 'resnet18' \
python3.8 /nfs/ppfl/fedml_dispfl/main_dispfl.py --model 'alex' \
--dataset 'HAM10000' \
--partition_method 'dir' \
--partition_alpha '0.5' \
--batch_size 128 \
--lr 0.1 \
--lr_decay 0.997 \
--epochs 5 \
--client_num_in_total 100 --frac 0.1 \
--comm_round 500 \
--dense_ratio 0.5 \
--anneal_factor 0.5 \
--seed 2022 \
--cs 'random' \
--dis_gradient_check \
--different_initial \
--tqdm