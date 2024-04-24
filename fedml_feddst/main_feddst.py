import argparse
import logging
import os
import random
import sys
import pdb
import numpy as np
import torch
from datetime import datetime

sys.path.insert(0, os.path.abspath("/nfs/ppfl/"))
from fedml_api.model.cv.vgg import vgg11
from fedml_api.model.cv.AlexNet import AlexNetCIFAR, AlexNet
from fedml_api.model.cv.resnet_ham import wide_resnet50_2
from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from fedml_api.data_preprocessing.cifar100.data_loader import (
    load_partition_data_cifar100,
)
from fedml_api.data_preprocessing.tiny_imagenet.data_loader import (
    load_partition_data_tiny,
)
from fedml_api.model.cv.cnn_cifar10 import cnn_cifar10, cnn_cifar100
from fedml_api.model.cv.wideresnet import wideresnet28_4
from fedml_api.model.cv.lenet5 import LeNet5
from fedml_api.standalone.feddst.feddst_api import FedDSTAPI
from fedml_api.standalone.feddst.feddst_model_trainer import FedDSTModelTrainer
from fedml_api.data_preprocessing.HAM10000.data_loader import (
    load_partition_data_HAM10000,
)
from fedml_api.model.cv.resnet import (
    customized_resnet18,
    tiny_resnet18,
    customized_resnet50,
)
from fedml_api.utils.common_args import add_common_args
import fedml_api.utils.config as config_loader


def logger_config(log_path, logging_name):
    logger = logging.getLogger(logging_name)
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(log_path, mode="w", encoding="UTF-8")
    handler.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    parser = add_common_args(parser)
    # Training settings
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        metavar="N",
        help="network architecture, supporting 'alexnet', 'resnet18', 'vgg11'",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        metavar="N",
        help="dataset used for training",
    )

    parser.add_argument(
        "--momentum", type=float, default=0, metavar="N", help="momentum"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="/nfs/ppfl/data/",
        help="data directory, please feel free to change the directory to the right place",
    )

    parser.add_argument(
        "--partition_method",
        type=str,
        default="dir",
        metavar="N",
        help="current supporting three types of data partition, one called 'dir' short for Dirichlet"
        "one called 'n_cls' short for how many classes allocated for each client"
        "and one called 'my_part' for partitioning all clients into PA shards with default latent Dir=0.3 distribution",
    )

    parser.add_argument(
        "--partition_alpha",
        type=float,
        default=0.3,
        metavar="PA",
        help="available parameters for data partition method",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        metavar="N",
        help="local batch size for training",
    )

    parser.add_argument(
        "--client_optimizer", type=str, default="sgd", help="SGD with momentum; adam"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate (default: 0.1)",
    )

    parser.add_argument(
        "--lr_decay",
        type=float,
        default=0.998,
        metavar="LR_decay",
        help="learning rate decay (default: 0.998)",
    )

    parser.add_argument(
        "--wd", help="weight decay parameter;", type=float, default=5e-4
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="EP",
        help="local training epochs for each client",
    )

    parser.add_argument(
        "--client_num_in_total",
        type=int,
        default=100,
        metavar="NN",
        help="number of workers in a distributed cluster",
    )

    parser.add_argument(
        "--frac",
        type=float,
        default=0.1,
        metavar="NN",
        help="selection fraction each round",
    )

    parser.add_argument(
        "--comm_round",
        type=int,
        default=500,
        help="how many round of communications we shoud use",
    )

    parser.add_argument(
        "--frequency_of_the_test",
        type=int,
        default=1,
        help="the frequency of the algorithms",
    )

    parser.add_argument("--gpu", type=int, default=0, help="gpu")

    parser.add_argument("--ci", type=int, default=0, help="CI")
    parser.add_argument("--tag", type=str, default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--rr_interval", type=int, default=1, help="Frequency of remove and regrow mask"
    )
    parser.add_argument(
        "--dense_ratio", type=float, default=0.5, help="local density ratio"
    )
    parser.add_argument(
        "--rigl",
        action="store_true",
        help="whether adopt RigL-like to regrow pruned mask",
    )

    parser.add_argument("--cs", type=str, default="v0")
    parser.add_argument("--active", type=float, default=1.0)

    parser.add_argument("--public_portion", type=float, default=0)
    parser.add_argument("--erk_power_scale", type=float, default=1)
    parser.add_argument("--dis_gradient_check", action="store_true")
    parser.add_argument("--strict_avg", action="store_true")
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--uniform", action="store_true")
    parser.add_argument("--save_masks", action="store_true")
    parser.add_argument("--different_initial", action="store_true")
    parser.add_argument("--record_mask_diff", action="store_true")
    parser.add_argument("--diff_spa", action="store_true")
    parser.add_argument("--global_test", action="store_true")
    parser.add_argument(
        "--anneal_factor", type=float, default=0.3, help="anneal factor for pruning"
    )

    return parser


def load_data(args, dataset_name):
    global train_data_num, test_data_num, train_data_global, test_data_global, train_data_local_num_dict, train_data_local_dict, raw_train_data_local_dict, test_data_local_dict, class_num
    if dataset_name == "cifar10":
        args.data_dir += "cifar10"
        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_cifar10(
            args.data_dir,
            args.partition_method,
            args.partition_alpha,
            args.client_num_in_total,
            args.batch_size,
            logger,
        )
    elif dataset_name == "cifar100":
        args.data_dir += "cifar100"
        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_cifar100(
            args.data_dir,
            args.partition_method,
            args.partition_alpha,
            args.client_num_in_total,
            args.batch_size,
            logger,
        )
    elif dataset_name == "HAM10000":
        args.data_dir += "HAM10000"
        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_HAM10000(
            args.data_dir,
            args.partition_method,
            args.partition_alpha,
            args.client_num_in_total,
            args.batch_size,
            logger,
        )
    else:
        raise Exception("Error: no dataset named " + dataset_name)
    # raw_training_dict yet to be implemented
    # elif dataset_name == "tiny":
    #     args.data_dir += "tiny_imagenet"
    #     train_data_num, test_data_num, train_data_global, test_data_global, \
    #     train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
    #     class_num = load_partition_data_tiny(args.data_dir, args.partition_method,
    #                                          args.partition_alpha, args.client_num_in_total,
    #                                              args.batch_size, logger)

    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]
    return dataset


def create_model(args, model_name, class_num):
    model = None
    if model_name == "cnn_cifar10":
        model = cnn_cifar10()
    elif model_name == "cnn_cifar100":
        model = cnn_cifar100()
    elif model_name == "resnet18" and args.dataset != "tiny":
        model = customized_resnet18(class_num=class_num)
    elif model_name == "resnet50" and args.dataset != "tiny":
        model = customized_resnet50(class_num=class_num)
    elif model_name == "resnet18" and args.dataset == "tiny":
        model = tiny_resnet18(class_num=class_num)
    elif model_name == "vgg11":
        model = vgg11(class_num)
    elif model_name == "alex":
        model = AlexNet(class_num)
    elif model_name == "alex_cifar10":
        model = AlexNetCIFAR()
    elif model_name == "wide_resnet28_4":
        model = wideresnet28_4(class_num)

    else:
        raise Exception("Error: no model " + model_name)
    return model


def custom_model_trainer(args, model, logger):
    return FedDSTModelTrainer(model, args, logger)


if __name__ == "__main__":

    parser = add_args(argparse.ArgumentParser(description="FedDST-standalone"))
    args = parser.parse_args()
    # print("torch version{}".format(torch.__version__))

    config_loader.Config.initialize(args)

    current_datetime = datetime.now()
    current_datetime_str = str(current_datetime.date())
    data_partition = args.partition_method
    if data_partition != "homo":
        data_partition += str(args.partition_alpha)
    args.identity = "FedDST" + "-" + data_partition
    args.client_num_per_round = int(args.client_num_in_total * args.frac)
    args.identity += "-mdl" + args.model
    args.identity += (
        "-cm" + str(args.comm_round) + "-total_clnt" + str(args.client_num_in_total)
    )
    args.identity += "-neighbor" + str(args.client_num_per_round)
    args.identity += "-prune_freq" + str(args.rr_interval)
    args.identity += "-seed" + str(args.seed)

    cur_dir = os.path.abspath(__file__).rsplit("/", 1)[0]
    log_path = os.path.join(
        cur_dir,
        "feddst/"
        + "LOG/"
        + args.dataset
        + "/"
        + current_datetime_str
        + args.identity
        + ".log",
    )
    logger = logger_config(log_path=log_path, logging_name=args.identity)
    logger.info(args)
    device = torch.device("cuda:" + str(args.gpu))
    args.device = torch.device(
        "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    logger.info(device)
    logger.info("running at device{}".format(device))

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # load data
    dataset = load_data(args, args.dataset)

    # create model.
    model = create_model(args, model_name=args.model, class_num=len(dataset[-1][0]))
    # print(model)
    model_trainer = custom_model_trainer(args, model, logger)
    logger.info(model)

    feddstAPI = FedDSTAPI(dataset, device, args, model_trainer, logger)
    feddstAPI.train()
