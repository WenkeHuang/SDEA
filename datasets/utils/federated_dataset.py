from abc import abstractmethod
from argparse import Namespace
from torch import nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from typing import Tuple
from torchvision import datasets
import numpy as np
import torch.optim


class FederatedDataset:
    """
    Federated learning evaluation setting.
    """
    NAME = None
    SETTING = None
    N_SAMPLES_PER_Class = None
    N_CLASS = None
    Nor_TRANSFORM = None

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.train_loaders = []
        self.test_loader = []
        self.args = args

    @abstractmethod
    def get_data_loaders(self, selected_domain_list=[]) -> Tuple[DataLoader, DataLoader]:
        """
        Creates and returns the training and test loaders for the current task.
        The current training loader and all test loaders are stored in self.
        :return: the current training and test loaders
        """
        pass

    @staticmethod
    @abstractmethod
    def get_backbone(parti_num, names_list, model_name='') -> nn.Module:
        """
        Returns the backbone to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_transform() -> transforms:
        """
        Returns the transform to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_normalization_transform() -> transforms:
        """
        Returns the transform used for normalizing the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_denormalization_transform() -> transforms:
        """
        Returns the transform used for denormalizing the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_scheduler(model, args: Namespace) -> torch.optim.lr_scheduler:
        """
        Returns the scheduler to be used for to the current dataset.
        """
        pass

    @staticmethod
    def get_epochs():
        pass

    @staticmethod
    def get_batch_size():
        pass


def partition_label_skew_loaders(clean_train_dataset: datasets, noise_train_dataset:datasets,
                                 evil_type,client_type,
                                 test_dataset: datasets,
                                 setting: FederatedDataset) -> Tuple[list, DataLoader, dict]:
    n_class = setting.N_CLASS
    n_participants = setting.args.parti_num
    n_class_sample = setting.N_SAMPLES_PER_Class
    min_size = 0
    min_require_size = 10

    if hasattr(clean_train_dataset, 'targets'):
        y_train = clean_train_dataset.targets
    else:
        y_train = clean_train_dataset.dataset.targets
    N = len(y_train)
    net_dataidx_map = {}

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_participants)]
        for k in range(n_class):
            idx_k = [i for i, j in enumerate(y_train) if j == k]
            np.random.shuffle(idx_k)
            if n_class_sample != None:
                idx_k = idx_k[0:n_class_sample * n_participants]
            beta = setting.args.beta
            if beta == 0:
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.array_split(idx_k, n_participants))]
            else:
                proportions = np.random.dirichlet(np.repeat(a=beta, repeats=n_participants))
                proportions = np.array([p * (len(idx_j) < N / n_participants) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_participants):
        np.random.shuffle(idx_batch[j])
        if n_class_sample != None:
            idx_batch[j] = idx_batch[j][0:n_class_sample * n_class]
        net_dataidx_map[j] = idx_batch[j]

    net_cls_counts = record_net_data_stats(y_train, net_dataidx_map,n_class)

    if evil_type in ['PairFlip','SymFlip']:
        for j in range(n_participants):
            train_sampler = SubsetRandomSampler(net_dataidx_map[j])
            if client_type[j]==True: # Loading Clean Data
                train_loader = DataLoader(clean_train_dataset,
                                      batch_size=setting.args.local_batch_size, sampler=train_sampler, num_workers=1, drop_last=False)
            elif client_type[j]==False: # Loading Noisy data
                train_loader = DataLoader(noise_train_dataset,
                                      batch_size=setting.args.local_batch_size, sampler=train_sampler, num_workers=1, drop_last=False)
            setting.train_loaders.append(train_loader)
    else:
        for j in range(n_participants):
            train_sampler = SubsetRandomSampler(net_dataidx_map[j])
            train_loader = DataLoader(clean_train_dataset,
            batch_size=setting.args.local_batch_size, sampler=train_sampler, num_workers=4, drop_last=False)
            setting.train_loaders.append(train_loader)

    test_loader = DataLoader(test_dataset,
                             batch_size=setting.args.local_batch_size, shuffle=False, num_workers=4)
    setting.test_loader = test_loader

    return setting.train_loaders, setting.test_loader, net_cls_counts

def record_net_data_stats(y_train, net_dataidx_map, n_class):
    net_cls_counts = {}
    y_train = np.array(y_train)
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp={}
        for i in range(n_class):
            if i in unq:
                tmp[i] = unq_cnt[unq==i][0]
            else:
                tmp[i] = 0
        net_cls_counts[net_i] = tmp

    data_list = []
    for net_id, data in net_cls_counts.items():
        n_total = 0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    print('Data statistics: %s' % str(net_cls_counts))
    # save_data_stat(net_cls_counts)
    return net_cls_counts


def save_data_stat(net_cls_counts):
    path = 'datastat.csv'
    with open(path, 'w') as f:
        for k1 in net_cls_counts:
            data = net_cls_counts[k1]
            out_str = ''
            for k2 in data:
                out_str += str(data[k2]) + ','
            out_str += '\n'
            f.write(out_str)