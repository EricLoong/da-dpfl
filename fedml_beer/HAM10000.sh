#!/bin/bash
python3.8 /nfs/da-dpfl/fedml_beer/main_beer.py --model 'alex' \
--dataset 'HAM10000' \
--partition_method 'dir' \
--partition_alpha '0.5' \
--batch_size 128 \
--lr 0.05 \
--lr_decay 0.997 \
--epochs 5 \
--client_num_in_total 100 --frac 0.1 \
--comm_round 300 \
--seed 2023 \
--compression_type 'gsgd' \
--compression_params 9 \
--graph_type 'expander' \
--graph_params 5 \
--gamma 0.01 \
--tqdm
