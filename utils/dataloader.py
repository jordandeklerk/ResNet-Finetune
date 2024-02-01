import torch
from torchvision import datasets, transforms

class CarDatasetLoader:
    def __init__(self, path_base, batch_size, num_workers):
        self.path_base = path_base
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = self.get_train_transform()
        self.test_transform = self.get_test_transform()

        self.train_dset, self.val_dset, self.test_dset = self.load_datasets()

        self.train_loader = self.get_data_loader(self.train_dset, shuffle=True)
        self.val_loader = self.get_data_loader(self.val_dset, shuffle=True)
        self.test_loader = self.get_data_loader(self.test_dset, shuffle=False)

    def get_train_transform(self):
        return transforms.Compose([
            transforms.TrivialAugmentWide(interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.RandomErasing(p=0.1),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_test_transform(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_datasets(self):
        train_data_dir = self.path_base + '/car_data/car_data/train'
        test_data_dir = self.path_base + '/car_data/car_data/test'

        train_dset = datasets.ImageFolder(train_data_dir, transform=self.train_transform)
        test_dset = datasets.ImageFolder(test_data_dir, transform=self.test_transform)

        val_count = round(len(train_dset) * 0.2)
        train_count = len(train_dset) - val_count

        train_dset, _ = torch.utils.data.random_split(train_dset, [train_count, val_count])
        _, val_dset = torch.utils.data.random_split(train_dset, [train_count, val_count])

        return train_dset, val_dset, test_dset

    def get_data_loader(self, dataset, shuffle):
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, pin_memory=True, num_workers=self.num_workers)

    def print_dataset_sizes(self):
        print(f"Data loaded: there are {len(self.train_dset)} images in the training data.")
        print(f"Data loaded: there are {len(self.val_dset)} images in the validation data.")
        print(f"Data loaded: there are {len(self.test_dset)} images in the testing data.")
