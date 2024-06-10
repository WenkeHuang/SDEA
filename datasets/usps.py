import torch
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder, ImageFolder, SVHN, USPS
from utils.conf import data_path
from datasets.utils.public_dataset import PublicDataset, random_loaders, FourCropsTransform, GaussianBlur
from datasets.transforms.denormalization import DeNormalize
import torchvision.transforms as T


class MyUSPS(torch.utils.data.Dataset):
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
        dataobj = USPS(self.root, self.train, self.transform, self.target_transform, self.download)

        return dataobj

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        img, target = self.dataset[index]

        return img, target


class PublicUSPS(PublicDataset):
    NAME = 'pub_usps'

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
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
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
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            torchvision_normalization
        ]
    )

    def get_data_loaders(self):
        pub_aug = self.args.pub_aug
        if pub_aug == 'weak':
            selected_transform = [self.weak_aug, self.weak_aug, self.weak_aug, self.weak_aug]
        elif pub_aug == 'strong':
            selected_transform = [self.weak_aug, self.weak_aug, self.strong_aug, self.strong_aug]
        elif pub_aug == 'asy':
            selected_transform = [self.weak_aug, self.weak_aug, self.strong_aug, self.weak_aug]

        train_dataset = MyUSPS(data_name='syn', root=data_path(),
                               transform=FourCropsTransform(selected_transform))
        traindl = random_loaders(train_dataset, self)
        return traindl

    @staticmethod
    def get_normalization_transform():
        transform = PublicUSPS.torchvision_normalization
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = PublicUSPS.torchvision_denormalization
        return transform
