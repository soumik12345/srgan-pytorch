from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


class TrainDataset(Dataset):

    def __init__(self, image_files, crop_size, scale):
        super(TrainDataset, self).__init__()
        self.image_files = image_files
        crop_size -= (crop_size % scale)
        self.hr_transform = transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.ToTensor()
        ])
        self.lr_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(
                crop_size // scale,
                interpolation=Image.BICUBIC
            ),
            transforms.ToTensor()
        ])

    def __getitem__(self, item):
        image_file = self.image_files[item]
        hr_image = self.hr_transform(Image.open(image_file))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_files)

    def get_loader(self, num_workers, batch_size):
        return DataLoader(
            dataset=self, num_workers=num_workers,
            batch_size=batch_size, shuffle=True
        )


class ValidationDataset(Dataset):

    def __init__(self, image_files, crop_size, scale):
        super(ValidationDataset, self).__init__()
        self.image_files = image_files
        self.crop_size = crop_size
        self.scale = scale

    def __getitem__(self, item):
        image_file = self.image_files[item]
        
        try:
            hr_image = Image.open(image_file)
            hr_image = transforms.CenterCrop(
                self.crop_size
            )(hr_image)
            lr_image = transforms.Resize(
                self.crop_size // self.scale,
                interpolation=Image.BICUBIC
            )(hr_image)
            hr_restore = transforms.Resize(
                self.crop_size,
                interpolation=Image.BICUBIC
            )(lr_image)

            hr_image = transforms.ToTensor()(hr_image)
            lr_image = transforms.ToTensor()(lr_image)
            hr_restore = transforms.ToTensor()(hr_restore)
        except:
            print(image_file)

        return lr_image, hr_image, hr_restore

    def __len__(self):
        return len(self.image_files)

    def get_loader(self, num_workers, batch_size):
        return DataLoader(
            dataset=self, num_workers=num_workers,
            batch_size=batch_size, shuffle=True
        )


class TestDataset(Dataset):

    def __init__(self, hr_files, lr_images, scale):
        super(TestDataset, self).__init__()
        self.hr_files = hr_files
        self.lr_files = lr_images
        self.scale = scale

    def __getitem__(self, item):
        lr_image = Image.open(self.lr_files[item])
        hr_image = Image.open(self.hr_files[item])
        hr_restore = transforms.Resize(
            (
                self.scale * lr_image.size[1],
                self.scale * lr_image.size[0]
            ), interpolation=Image.BICUBIC
        )(lr_image)

        lr_image = transforms.ToTensor(lr_image)
        hr_image = transforms.ToTensor(hr_image)
        hr_restore = transforms.ToTensor(hr_restore)

        return lr_image, hr_image, hr_restore

    def __len__(self):
        return len(self.hr_files)
