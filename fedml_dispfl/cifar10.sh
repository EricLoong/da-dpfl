#!/bin/bash
python /nfs/ppfl/fedml_dispfl/main_dispfl.py --model 'resnet18' \
--dataset 'cifar10' \
--partition_method 'dir' \
--partition_alpha '0.3' \
--batch_size 128 \
--lr 0.1 \
--lr_decay 0.998 \
--epochs 5 \
--client_num_in_total 100 --frac 0.1 \
--comm_round 500 \
--dense_ratio 0.5 \
--anneal_factor 0.5 \
--seed 2023 \
--cs 'random' \
--dis_gradient_check \
--different_initial
