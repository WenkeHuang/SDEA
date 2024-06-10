import torchvision.transforms as transforms
from utils.conf import data_path
from PIL import Image
from datasets.utils.federated_dataset import FederatedDataset, partition_label_skew_loaders
from datasets.utils.public_dataset import PublicDataset, random_loaders, FourCropsTransform, GaussianBlur
from backbone.ResNet import resnet50
from datasets.transforms.denormalization import DeNormalize
from torch.utils.data import Dataset
import numpy as np
import os
import torchvision.transforms as T

class TinyImagenet(Dataset):
    def __init__(self, root: str, train: bool = True, transform: transforms = None,
                 target_transform: transforms = None, download: bool = False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data = []
        for num in range(20):
            self.data.append(np.load(os.path.join(
                root, 'processed/x_%s_%02d.npy' %
                      ('train' if self.train else 'val', num + 1))))
        self.data = np.concatenate(np.array(self.data))

        self.targets = []
        for num in range(20):
            self.targets.append(np.load(os.path.join(
                root, 'processed/y_%s_%02d.npy' %
                      ('train' if self.train else 'val', num + 1))))
        self.targets = np.concatenate(np.array(self.targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(np.uint8(255 * img))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class MyTinyImagenet(TinyImagenet):

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        super(MyTinyImagenet, self).__init__(
            root, train, transform, target_transform, download)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(np.uint8(255 * img))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class PublicTinyImagenet(PublicDataset):
    NAME = 'pub_tyimagenet'

    torchvision_normalization = T.Normalize(mean=(0.485, 0.456, 0.406),
                                            std=(0.229, 0.224, 0.225))
    torchvision_denormalization = DeNormalize(mean=(0.485, 0.456, 0.406),
                                              std=(0.229, 0.224, 0.225))

    strong_aug = transforms.Compose(
        [
            transforms.RandomResizedCrop(32),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            torchvision_normalization
        ]
    )

    weak_aug = transforms.Compose(
        [
            # transforms.ToPILImage(),
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            torchvision_normalization
        ]
    )

    def get_data_loaders(self):
        pub_aug = self.args.pub_aug
        if pub_aug == 'weak':
            selected_transform = [self.weak_aug, self.weak_aug, self.weak_aug, self.weak_aug]
        elif pub_aug == 'strong':
            selected_transform = [self.strong_aug, self.strong_aug, self.strong_aug, self.strong_aug]
        elif pub_aug == 'asy':
            selected_transform = [self.weak_aug, self.strong_aug, self.strong_aug, self.weak_aug]

        train_dataset = MyTinyImagenet(data_path() + 'TINYIMG', train=True,
                                       download=False, transform=FourCropsTransform(selected_transform))
        traindl = random_loaders(train_dataset, self)
        return traindl

    @staticmethod
    def get_normalization_transform():
        transform = PublicTinyImagenet.torchvision_normalization
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = PublicTinyImagenet.torchvision_denormalization
        return transform
