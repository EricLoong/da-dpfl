#!/bin/bash
python3.8 /nfs/ppfl/fedml_ditto/main_ditto.py --model 'alex' \
--dataset 'HAM10000' \
--partition_method 'dir' \
--partition_alpha '0.5' \
--batch_size 128 \
--lr 0.1 \
--lr_decay 0.998 \
--client_num_in_total 100 --frac 0.1 \
--comm_round 500 \
--seed 2022 \
--lamda 0.5 \
--epochs 2 \
--local_epochs 3 \
--seed 2022 \

