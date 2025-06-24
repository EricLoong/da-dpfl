# ‘DA-DPFL’: Dynamic Aggregation & Decentralized Personalized Federated Learning

## Running Environment:
![Python](https://img.shields.io/badge/Python-3.7-brightgreen.svg) 
![Python](https://img.shields.io/badge/Python-3.8-brightgreen.svg) 


## Structure (description of folders)

### data
- the folder to store the data


### fedml_api (interface for multiple modules)

  - data_preprocessing: processing tools

  - model: the folder to store the model (parent models) and training tools

  - standalone: the folder to store the different algorithms 

    - client.py: the functions executed in client such as training and pruning

    - api.py: the functions executed in the server, i.e. aggregation, and the whole logic in algorithm.
    
    - model_trainer.py: store the class of model for corresponding algorithm

  - utils: the folder to store the tools for logging, FLOPS computation, etc.
  
### fedml_core 
- the folder to store the core functions of FedML

### fedml_xxx 
- main running interface for different baselines, where .sh files stored.



## Example to use the code:
### Change the directory to the root of the your project
replace work_dir
```
/nfs/da-dpfl/
```
in config.yaml with the root of your project
```
/your_path_to_project/
```
### Install dependencies and setup the permissions
```
pip3 install -r requirements.txt
```
```
sh setup_permission.sh
```

### Run CIFAR10 experiments

# Format - sh /your_directory/algorithm_name/data_name.sh
```
- sh /your_path_to_fedml/fedml_dadpfl/cifar10.sh
- sh /your_path_to_fedml/fedml_dispfl/cifar10.sh
```





## Acknowledgements (Codes)
- https://github.com/diaoenmao/Pruning-Deep-Neural-Networks-from-a-Sparsity-Perspective
- https://github.com/rong-dai/DisPFL
- https://github.com/liboyue/beer
