import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder, ImageFolder
from utils.conf import data_path
from datasets.utils.public_dataset import PublicDataset, random_loaders, FourCropsTransform, GaussianBlur
from datasets.transforms.denormalization import DeNormalize
import torchvision.transforms as T


class ImageFolder_Custom(ImageFolder):
    def __init__(self, data_name, root, train=True, transform=None, target_transform=None):
        self.data_name = data_name
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if train:
            self.imagefolder_obj = ImageFolder(self.root + self.data_name + '/train/', self.transform, self.target_transform)
        else:
            self.imagefolder_obj = ImageFolder(self.root + self.data_name + '/val/', self.transform, self.target_transform)

        self.samples = self.imagefolder_obj.samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        img = self.imagefolder_obj.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class PublicSYN(PublicDataset):
    NAME = 'pub_syn'

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
            selected_transform = [self.weak_aug, self.weak_aug, self.strong_aug, self.strong_aug]
        elif pub_aug == 'asy':
            selected_transform = [self.weak_aug, self.weak_aug, self.strong_aug, self.weak_aug]

        train_dataset = ImageFolder_Custom(data_name='syn', root=data_path(),
                                           transform=FourCropsTransform(selected_transform))
        traindl = random_loaders(train_dataset, self)
        return traindl

    @staticmethod
    def get_normalization_transform():
        transform = PublicSYN.torchvision_normalization
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = PublicSYN.torchvision_denormalization
        return transform
