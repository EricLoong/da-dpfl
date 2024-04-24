#!/bin/bash
python3.8 /nfs/ppfl/fedml_dfedsam/main_dfedsam.py --model 'vgg11' \
--dataset 'cifar100' \
--partition_method 'dir' \
--partition_alpha '0.3' \
--batch_size 128 \
--lr 0.1 \
--lr_decay 0.998 \
--epochs 5 \
--rho 0.01 \
--cs 'random' \
--client_num_in_total 100 --frac 0.1 \
--comm_round 500 \
--seed 2022 \
--tqdm
