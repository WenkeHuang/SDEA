import numpy as np
import torch

import torchvision.transforms as transforms


from backbone.SimpleCNN import SimpleCNN

from datasets.utils.utils import noisify
from utils.conf import data_path
from PIL import Image
from datasets.utils.federated_dataset import FederatedDataset, partition_label_skew_loaders
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize
from torchvision.datasets import MNIST


class MyCleanMNIST(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False, data_name=None) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.data_name = data_name
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.dataset = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        dataobj = MNIST(self.root, self.train, self.transform, self.target_transform, self.download)

        return dataobj

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        img, target = self.dataset[index]

        return img, target


class MyNoiseMNIST(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, noise_data_type=None, noise_data_rate=0,
                 target_transform=None, download=False, data_name=None) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.data_name = data_name
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.dataset = self.__build_truncated_dataset__()

        self.cls_num = len(self.dataset.classes)
        self.noise_data_type = noise_data_type
        t = self.train_labels = np.asarray([[self.dataset.targets[i]] for i in range(len(self.dataset.targets))])
        self.train_noisy_labels, self.actual_noise_rate = noisify(
            train_labels=t,
            noise_type=noise_data_type,
            noise_rate=noise_data_rate,
            nb_classes=self.cls_num)

    def __build_truncated_dataset__(self):
        dataobj = MNIST(self.root, self.train, self.transform, self.target_transform, self.download)

        return dataobj

    def __getitem__(self, index: int):
        img, target = self.dataset[index]
        # img = transforms.ToPILImage()(img)
        # 错误的数据换label
        if self.train and self.noise_data_type is not None:
            target = self.train_noisy_labels[index][0]
        return img, target


class FedMNIST(FederatedDataset):
    NAME = 'fl_mnist'
    SETTING = 'label_skew'
    N_SAMPLES_PER_Class = None
    N_CLASS = 10
    Singel_Channel_Nor_TRANSFORM = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize((0.1307, 0.1307, 0.1307),
                                 (0.3081, 0.3081, 0.3081))])

    def get_data_loaders(self, train_transform=None):

        pri_aug = self.args.pri_aug
        if pri_aug == 'weak':
            train_transform = self.Singel_Channel_Nor_TRANSFORM

        clean_train_dataset = MyCleanMNIST(root=data_path(), train=True,
                                           download=False, transform=train_transform)
        if self.args.evils in ['PairFlip', 'SymFlip']:
            noise_train_dataset = MyNoiseMNIST(root=data_path(), train=True,
                                               noise_data_type=self.args.evils,
                                               noise_data_rate=self.args.noise_data_rate,
                                               download=False, transform=train_transform)
        else:
            noise_train_dataset = MyCleanMNIST(root=data_path(), train=True,
                                               download=False, transform=train_transform)

        test_transform = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor(),
             transforms.Lambda(lambda x: x.repeat(3, 1, 1)), self.get_normalization_transform()])

        test_dataset = MNIST(data_path(), train=False,
                             download=False, transform=test_transform)

        # traindls, testdl, net_cls_counts = partition_label_skew_loaders(train_dataset, test_dataset, self)

        '''
        Generate benign and malicious clients
        '''
        if self.args.evils == 'None':  # Data 和 Model Poisoning Attacks under this supervision
            good_scale = self.args.parti_num
            bad_scale = 0
        else:
            # good_scale = self.args.parti_num * (1 - self.args.bad_client_rate)
            bad_scale = int(self.args.parti_num * self.args.bad_client_rate)
            good_scale = self.args.parti_num - bad_scale
        client_type = np.repeat(True, good_scale).tolist() + (np.repeat(False, bad_scale)).tolist()

        print('The Evils Type {}, Ture: Benign and False: Malicious '.format(self.args.evils))
        print(client_type)
        traindls, testdl, net_cls_counts = partition_label_skew_loaders(
            clean_train_dataset=clean_train_dataset,
            noise_train_dataset=noise_train_dataset,
            evil_type=self.args.evils,
            client_type=client_type,
            test_dataset=test_dataset,
            setting=self)

        return traindls, testdl, net_cls_counts, client_type

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), FedMNIST.Singel_Channel_Nor_TRANSFORM])
        return transform

    @staticmethod
    def get_backbone(parti_num, names_list, model_name=''):
        nets_list = []

        for j in range(parti_num):
            nets_list.append(SimpleCNN(FedMNIST.N_CLASS))

        return nets_list

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(mean=(0.1307, 0.1307, 0.1307),
                                         std=(0.3081, 0.3081, 0.3081))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(mean=(0.1307, 0.1307, 0.1307),
                                std=(0.3081, 0.3081, 0.3081))
        return transform
