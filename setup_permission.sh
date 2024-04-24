#!/bin/bash

# List of all algorithms
declare -a algorithms=("dfedavgm" "dispfl" "fedavg" "gossipfl" "beer" "dadpfl" "dfedsam" "ditto" "feddst")

# Data names
declare -a data_names=("cifar10" "cifar100" "HAM10000")

# Loop through each algorithm
for algo in "${algorithms[@]}"; do
    # Create the fedml_X directory
    mkdir -p "fedml_$algo"

    # Loop through each data name and create the required directory structure
    for data_name in "${data_names[@]}"; do
        mkdir -p "fedml_$algo/$algo/LOG/$data_name"
    done

    # Change permissions for each X under fedml_X
    chmod -R 777 "fedml_$algo/$algo"
done

# Create a directory called data in the current working directory and open its permission
mkdir -p data
chmod -R 777 data

echo "Directories created and permissions set."
