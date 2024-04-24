#!/bin/bash
python /nfs/da-dpfl/fedml_beer/main_beer.py --model 'vgg11' \
--dataset 'cifar100' \
--partition_method 'dir' \
--partition_alpha '0.3' \
--batch_size 128 \
--lr 0.05 \
--lr_decay 0.998 \
--epochs 5 \
--client_num_in_total 100 --frac 0.1 \
--comm_round 500 \
--seed 2022 \
--compression_type 'gsgd' \
--compression_params 9 \
--graph_type 'expander' \
--graph_params 5 \
--gamma 0.005 \
--tqdm