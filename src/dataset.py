import torch
from PIL import Image
from torchvision import transforms
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
