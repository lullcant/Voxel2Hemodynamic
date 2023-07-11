from torchvision.datasets import CIFAR10 as tmp
import torchvision.transforms as transforms


class CIFAR10(tmp):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, ):
        super(CIFAR10, self).__init__(root, train, transform, target_transform, download)
        self.labels = self.targets


def get_dataset(dataset='cifar10'):
    if dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_valid = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_dataset = CIFAR10(root='./data', download=True, transform=transform_train, train=True)
        valid_dataset = CIFAR10(root='./data', download=True, transform=transform_valid, train=False)
    else:
        assert 'No adaptable dataset'
        train_dataset = None
        valid_dataset = None

    return train_dataset, valid_dataset
