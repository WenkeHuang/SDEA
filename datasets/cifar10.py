import numpy as np
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from datasets.utils.utils import noisify
from utils.conf import data_path
from PIL import Image
from datasets.utils.federated_dataset import FederatedDataset, partition_label_skew_loaders
from typing import Tuple
from backbone.SimpleCNN import SimpleCNN
import torchvision.transforms as T
from datasets.transforms.denormalization import DeNormalize

class MyNoiseCIFAR10(CIFAR10):
    """
    Overrides the CIFAR10 dataset to change the getitem function.
    """

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, noise_data_type=None, noise_data_rate=0, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        super(MyNoiseCIFAR10, self).__init__(root, train, transform, target_transform, download)

        self.cls_num = len(self.classes)
        self.noise_data_type = noise_data_type
        t = self.train_labels = np.asarray([[self.targets[i]] for i in range(len(self.targets))])
        self.train_noisy_labels, self.actual_noise_rate = noisify(
            train_labels=t,
            noise_type=noise_data_type,
            noise_rate=noise_data_rate,
            nb_classes=self.cls_num)

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.train and self.noise_data_type is not None:
            target = self.train_noisy_labels[index][0]
        return img, target


class MyCleanCIFAR10(CIFAR10):
    """
    Overrides the CIFAR10 dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        super(MyCleanCIFAR10, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class FedLeaCIFAR10(FederatedDataset):
    NAME = 'fl_cifar10'
    SETTING = 'label_skew'
    N_SAMPLES_PER_Class = None
    N_CLASS = 10
    torchvision_normalization = T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615))
    torchvision_denormalization = DeNormalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615))

    Nor_TRANSFORM = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         torchvision_normalization])

    def get_data_loaders(self, train_transform=None):
        pri_aug = self.args.pri_aug
        if pri_aug == 'weak':
            train_transform = self.Nor_TRANSFORM

        clean_train_dataset = MyCleanCIFAR10(root=data_path(), train=True,
                                             download=False, transform=train_transform)
        if self.args.evils in ['PairFlip','SymFlip']:
            noise_train_dataset = MyNoiseCIFAR10(root=data_path(), train=True,
                                                 noise_data_type=self.args.evils,
                                                 noise_data_rate=self.args.noise_data_rate,
                                                 download=False, transform=train_transform)
        else:
            noise_train_dataset = MyCleanCIFAR10(root=data_path(), train=True,
                                             download=False, transform=train_transform)

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])
        test_dataset = CIFAR10(data_path(), train=False,
                               download=False, transform=test_transform)

        if self.args.evils =='None':
            good_scale = self.args.parti_num
            bad_scale = 0
        else:
            bad_scale = int(self.args.parti_num * self.args.bad_client_rate)
            good_scale = self.args.parti_num - bad_scale
        client_type = np.repeat(True, good_scale).tolist() + (np.repeat(False, bad_scale)).tolist()

        print('The Evils Type {}, Ture: Benign and False: Malicious '.format(self.args.evils))
        print(client_type)
        traindls, testdl, net_cls_counts = partition_label_skew_loaders(
            clean_train_dataset= clean_train_dataset,
            noise_train_dataset= noise_train_dataset,
            evil_type = self.args.evils,
            client_type = client_type,
            test_dataset = test_dataset,
            setting=self)

        return traindls, testdl, net_cls_counts,client_type

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), FedLeaCIFAR10.Nor_TRANSFORM])
        return transform

    @staticmethod
    def get_backbone(parti_num, names_list, model_name=''):
        nets_list = []

        for j in range(parti_num):
            nets_list.append(SimpleCNN(FedLeaCIFAR10.N_CLASS))
        return nets_list

    @staticmethod
    def get_normalization_transform():
        transform = FedLeaCIFAR10.torchvision_normalization
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = FedLeaCIFAR10.torchvision_denormalization
        return transform
