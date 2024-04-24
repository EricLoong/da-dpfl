#!/bin/bash
python3.8 /nfs/ppfl/fedml_dfedavgm/main_dfedavgm.py --model 'alex' \
--dataset 'HAM10000' \
--partition_method 'dir' \
--partition_alpha '0.5' \
--batch_size 128 \
--lr 0.1 \
--lr_decay 0.997 \
--epochs 5 \
--cs 'random' \
--client_num_in_total 100 --frac 0.1 \
--comm_round 500 \
--seed 2022 \
--tqdm
