#!/bin/bash
python3.8 /nfs/ppfl/fedml_gossipfl/main_gossipfl.py --model 'resnet18' \
--dataset 'cifar10' \
--partition_method 'dir' \
--partition_alpha '0.3' \
--batch_size 128 \
--lr 0.1 \
--lr_decay 0.998 \
--epochs 5 \
--target_sparsity 0.5 \
--client_num_in_total 100 --frac 0.1 \
--comm_round 500 \
--seed 2022
