from abc import abstractmethod
from argparse import Namespace
from torchvision.transforms import transforms
from torch.utils.data import DataLoader,SubsetRandomSampler
from torchvision import datasets
from PIL import ImageFilter
import numpy as np
import random


class PublicDataset:
    NAME = None

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.train_loader = None
        self.args = args

    @abstractmethod
    def get_data_loaders(self) -> DataLoader:
        """
        Creates and returns the training and test loaders for the current task.
        The current training loader and all test loaders are stored in self.
        :return: the current training and test loaders
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
    def get_epochs():
        pass

    @staticmethod
    def get_batch_size():
        pass

def random_loaders(train_dataset: datasets,
                      setting: PublicDataset) -> DataLoader:
    public_scale = setting.args.public_len
    # y_train = train_dataset.targets
    n_train = len(train_dataset)
    idxs = np.random.permutation(n_train)
    if public_scale!=None:
        idxs = idxs[0:public_scale]
    train_sampler = SubsetRandomSampler(idxs)
    train_loader = DataLoader(train_dataset,batch_size=setting.args.public_batch_size, sampler=train_sampler, num_workers=1)
    setting.train_loader=train_loader

    return setting.train_loader

class ThreeCropsTransform:
    """Take two random crops of one image as the query and key."""
    def __init__(self, transform):
        self.transform1 = transform[0]
        self.transform2 = transform[1]
        self.transform3 = transform[2]

    def __call__(self, x):
        q = self.transform1(x)
        k = self.transform2(x)
        v = self.transform2(x)
        return [q, k,v]

class FourCropsTransform:
    """Take four random crops of one image."""
    def __init__(self, transform):
        self.transform1 = transform[0]
        self.transform2 = transform[1]
        self.transform3 = transform[2]
        self.transform4 = transform[3]

    def __call__(self, x):
        q = self.transform1(x)
        k = self.transform2(x)
        u = self.transform3(x)
        v = self.transform4(x)
        return [q,k,u,v]

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x