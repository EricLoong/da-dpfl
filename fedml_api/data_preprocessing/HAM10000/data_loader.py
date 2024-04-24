import logging
import math
import pdb
import numpy as np
import torch
import random
import os
import pandas as pd
from glob import glob
import torch.utils.data as data
import torchvision.transforms as transforms
from .datasets import HAM10000, HAM10000_truncated
from sklearn.model_selection import train_test_split
import fedml_api.utils.config as config_loader


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = []

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = []
        for i in range(7):
            if i in unq:
                tmp.append(unq_cnt[np.argwhere(unq == i)][0, 0])
            else:
                tmp.append(0)
        net_cls_counts.append(tmp)
    return net_cls_counts


def record_part(y_test, train_cls_counts, test_dataidxs, logger):
    test_cls_counts = []

    for net_i, dataidx in enumerate(test_dataidxs):
        unq, unq_cnt = np.unique(y_test[dataidx], return_counts=True)
        tmp = []
        for i in range(7):
            if i in unq:
                tmp.append(unq_cnt[np.argwhere(unq == i)][0, 0])
            else:
                tmp.append(0)
        test_cls_counts.append(tmp)
        logger.debug(
            "DATA Partition: Train %s; Test %s"
            % (str(train_cls_counts[net_i]), str(tmp))
        )
    return


def _data_transforms_HAM10000():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),  # new
            transforms.RandomAdjustSharpness(random.uniform(0, 4.0)),
            transforms.RandomAutocontrast(),
            transforms.Pad(3),
            transforms.RandomRotation(10),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Pad(3),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return train_transforms, test_transforms


def load_HAM10000_data(data_dir):
    all_image_path = glob(os.path.join(data_dir, "*", "*.jpg"))
    # extracts the image id to match it with the .csv label file
    imageid_path_dict = {
        os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path
    }
    lesion_type_dict = {
        "nv": "Melanocytic nevi",
        "mel": "Melanoma",
        "bkl": "Benign keratosis-like lesions ",
        "bcc": "Basal cell carcinoma",
        "akiec": "Actinic keratoses",
        "vasc": "Vascular lesions",
        "df": "Dermatofibroma",
    }

    df = pd.read_csv(os.path.join(data_dir, "HAM10000_metadata.csv"))
    # .map maps value from keys
    # .get returns all values
    df["path"] = df["image_id"].map(imageid_path_dict.get)
    df["cell_type"] = df["dx"].map(lesion_type_dict.get)
    df["cell_type_idx"] = pd.Categorical(df["cell_type"]).codes

    # creating train, test sets
    df_train, df_test = train_test_split(df, test_size=0.2)

    # Copy fewer class to balance the number of 7 classes
    data_aug_rate = [18, 12, 6, 53, 0, 44, 6]
    for i in range(7):
        if data_aug_rate[i]:
            augmented_data = [df_train.loc[df_train["cell_type_idx"] == i, :]] * (
                data_aug_rate[i] - 1
            )
            df_train = pd.concat([df_train] + augmented_data, ignore_index=True)

    df_train = df_train.reset_index(drop=True)[["path", "cell_type_idx"]]
    df_test = df_test.reset_index(drop=True)[["path", "cell_type_idx"]]
    y_train = df_train["cell_type_idx"]
    y_test = df_test["cell_type_idx"]

    return df_train, y_train, df_test, y_test


# Note that X_train and X_test are not actually used
def partition_data(datadir, partition, n_nets, alpha, logger):
    logger.info("*********partition data***************")
    df_train, y_train, df_test, y_test = load_HAM10000_data(datadir)

    if partition == "n_cls":
        n_client = n_nets
        n_cls = 7

        n_data_per_clnt = len(y_train) / n_client
        clnt_data_list = np.random.lognormal(
            mean=np.log(n_data_per_clnt), sigma=0, size=n_client
        )
        clnt_data_list = (
            clnt_data_list / np.sum(clnt_data_list) * len(y_train)
        ).astype(int)
        cls_priors = np.zeros(shape=(n_client, n_cls))
        for i in range(n_client):
            cls_priors[i][random.sample(range(n_cls), int(alpha))] = 1.0 / alpha

        prior_cumsum = np.cumsum(cls_priors, axis=1)

        idx_list = [np.where(y_train == i)[0] for i in range(n_cls)]
        cls_amount = [len(idx_list[i]) for i in range(n_cls)]
        net_dataidx_map = {}
        for j in range(n_client):
            net_dataidx_map[j] = []

        while np.sum(clnt_data_list) != 0:
            curr_clnt = np.random.randint(n_client)
            # If current node is full resample a client
            # print('Remaining Data: %d' %np.sum(clnt_data_list))
            if clnt_data_list[curr_clnt] <= 0:
                continue
            clnt_data_list[curr_clnt] -= 1
            curr_prior = prior_cumsum[curr_clnt]
            while True:
                cls_label = np.argmax(np.random.uniform() <= curr_prior)
                # Redraw class label if trn_y is out of that class
                if cls_amount[cls_label] <= 0:
                    cls_amount[cls_label] = np.random.randint(
                        0, len(idx_list[cls_label])
                    )
                    continue
                cls_amount[cls_label] -= 1
                net_dataidx_map[curr_clnt].append(
                    idx_list[cls_label][cls_amount[cls_label]]
                )

                break

    elif partition == "dir":
        n_client = n_nets
        n_cls = 7

        n_data_per_clnt = len(y_train) / n_client
        clnt_data_list = np.random.lognormal(
            mean=np.log(n_data_per_clnt), sigma=0, size=n_client
        )
        clnt_data_list = (
            clnt_data_list / np.sum(clnt_data_list) * len(y_train)
        ).astype(int)
        cls_priors = np.random.dirichlet(alpha=[alpha] * n_cls, size=n_client)
        prior_cumsum = np.cumsum(cls_priors, axis=1)

        idx_list = [np.where(y_train == i)[0] for i in range(n_cls)]
        cls_amount = [len(idx_list[i]) for i in range(n_cls)]
        net_dataidx_map = {}
        for j in range(n_client):
            net_dataidx_map[j] = []

        while np.sum(clnt_data_list) != 0:
            curr_clnt = np.random.randint(n_client)
            # If current node is full resample a client
            # print('Remaining Data: %d' %np.sum(clnt_data_list))
            if clnt_data_list[curr_clnt] <= 0:
                continue
            clnt_data_list[curr_clnt] -= 1
            curr_prior = prior_cumsum[curr_clnt]
            while True:
                cls_label = np.argmax(np.random.uniform() <= curr_prior)
                # Redraw class label if trn_y is out of that class
                if cls_amount[cls_label] <= 0:
                    continue
                cls_amount[cls_label] -= 1
                net_dataidx_map[curr_clnt].append(
                    idx_list[cls_label][cls_amount[cls_label]]
                )
                break

    elif partition == "my_part":
        n_shards = int(alpha)
        n_client = n_nets
        n_cls = 7

        n_data_per_clnt = len(y_train) / n_client
        clnt_data_list = np.random.lognormal(
            mean=np.log(n_data_per_clnt), sigma=0, size=n_client
        )
        clnt_data_list = (
            clnt_data_list / np.sum(clnt_data_list) * len(y_train)
        ).astype(int)
        cls_priors = np.zeros(shape=(n_client, n_cls))

        # default partition method with Dirichlet=0.3
        cls_priors_tmp = np.random.dirichlet(alpha=[0.3] * n_cls, size=int(n_shards))

        for i in range(n_client):
            cls_priors[i] = cls_priors_tmp[int(i / int(n_client / n_shards))]

        prior_cumsum = np.cumsum(cls_priors, axis=1)

        idx_list = [np.where(y_train == i)[0] for i in range(n_cls)]
        cls_amount = [len(idx_list[i]) for i in range(n_cls)]
        net_dataidx_map = {}
        for j in range(n_client):
            net_dataidx_map[j] = []

        while np.sum(clnt_data_list) != 0:
            curr_clnt = np.random.randint(n_client)
            # If current node is full resample a client
            # print('Remaining Data: %d' %np.sum(clnt_data_list))
            if clnt_data_list[curr_clnt] <= 0:
                continue
            clnt_data_list[curr_clnt] -= 1
            curr_prior = prior_cumsum[curr_clnt]
            while True:
                cls_label = np.argmax(np.random.uniform() <= curr_prior)
                # Redraw class label if trn_y is out of that class
                if cls_amount[cls_label] <= 0:
                    cls_amount[cls_label] = len(idx_list[cls_label])
                    continue
                cls_amount[cls_label] -= 1
                net_dataidx_map[curr_clnt].append(
                    idx_list[cls_label][cls_amount[cls_label]]
                )
                break

    elif partition == 'iid':
        n_client = n_nets  
        n_train = len(y_train)
        idx_shuffled = np.random.permutation(n_train)
        net_dataidx_map = {}
        data_per_client = [n_train // n_client + (1 if x < n_train % n_client else 0) for x in range(n_client)]
        start = 0
        for i, num_data in enumerate(data_per_client):
            end = start + num_data
            net_dataidx_map[i] = idx_shuffled[start:end]
            start = end

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    return df_train, y_train, df_test, y_test, net_dataidx_map, traindata_cls_counts


def get_dataloader_HAM10000(
    df_train,
    df_test,
    train_bs,
    test_bs,
    dataidxs=None,
    test_idxs=None,
    cache_train_data_set=None,
    cache_test_data_set=None,
    logger=None,
):
    transform_train, transform_test = _data_transforms_HAM10000()
    dataidxs = np.array(dataidxs)
    logger.info("train_num{}  test_num{}".format(len(dataidxs), len(test_idxs)))
    train_ds = HAM10000_truncated(
        df_train,
        dataidxs=dataidxs,
        transform=transform_train,
        cache_data_set=cache_train_data_set,
    )
    test_ds = HAM10000_truncated(
        df_test,
        dataidxs=test_idxs,
        transform=transform_test,
        cache_data_set=cache_test_data_set,
    )

    config = config_loader.Config()
    args = config.get_args()

    if args.pin_memory:
        print("pin_memory Enabled!!! num_workers: {}".format(args.num_workers))
        train_dl = data.DataLoader(
            dataset=train_ds,
            batch_size=train_bs,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        test_dl = data.DataLoader(
            dataset=test_ds,
            batch_size=test_bs,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        train_dl = data.DataLoader(
            dataset=train_ds, batch_size=train_bs, shuffle=False, drop_last=False
        )
        test_dl = data.DataLoader(
            dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False
        )
    return train_dl, test_dl


def load_partition_data_HAM10000(
    data_dir, partition_method, partition_alpha, client_number, batch_size, logger
):
    (
        df_train,
        y_train,
        df_test,
        y_test,
        net_dataidx_map,
        traindata_cls_counts,
    ) = partition_data(
        data_dir, partition_method, client_number, partition_alpha, logger
    )
    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    transform_train, transform_test = _data_transforms_HAM10000()
    cache_train_data_set = HAM10000(df_train, transform_train)
    cache_test_data_set = HAM10000(df_test, transform_test)
    idx_test = [[] for i in range(7)]
    # checking
    for label in range(7):
        idx_test[label] = np.where(y_test == label)[0]
    test_dataidxs = [[] for i in range(client_number)]
    tmp_tst_num = math.ceil(len(cache_test_data_set) / client_number)
    for client_idx in range(client_number):
        for label in range(7):
            # each has 100 pieces of testing data
            label_num = math.ceil(
                traindata_cls_counts[client_idx][label]
                / sum(traindata_cls_counts[client_idx])
                * tmp_tst_num
            )
            rand_perm = np.random.permutation(len(idx_test[label]))
            if len(test_dataidxs[client_idx]) == 0:
                test_dataidxs[client_idx] = idx_test[label][rand_perm[:label_num]]
            else:
                test_dataidxs[client_idx] = np.concatenate(
                    (test_dataidxs[client_idx], idx_test[label][rand_perm[:label_num]])
                )
        dataidxs = net_dataidx_map[client_idx]
        train_data_local, test_data_local = get_dataloader_HAM10000(
            df_train,
            df_test,
            batch_size,
            batch_size,
            dataidxs,
            test_dataidxs[client_idx],
            cache_train_data_set=cache_train_data_set,
            cache_test_data_set=cache_test_data_set,
            logger=logger,
        )
        local_data_num = len(train_data_local.dataset)
        data_local_num_dict[client_idx] = local_data_num
        logger.info(
            "client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num)
        )
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

    record_part(y_test, traindata_cls_counts, test_dataidxs, logger)

    return (
        None,
        None,
        None,
        None,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        traindata_cls_counts,
    )
