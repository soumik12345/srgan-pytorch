import torch
from glob import glob
from matplotlib import pyplot as plt
from torchvision.transforms import ToPILImage
from src.models import Generator, Discriminator
from src.dataset import TrainDataset, ValidationDataset


train_dataset = TrainDataset(glob('./VOC2012/JPEGImages/*')[:16000], 88, 4)
print(len(train_dataset))
x, y = train_dataset[0]
print(x.shape, y.shape)

plt.imshow(ToPILImage()(x))
plt.show()
plt.imshow(ToPILImage()(y))
plt.show()


val_dataset = ValidationDataset(glob('./VOC2012/JPEGImages/*')[16000:17000], 4)
print(len(val_dataset))
x, y, y_res = val_dataset[0]
print(x.shape, y.shape, y_res.shape)

plt.imshow(ToPILImage()(x))
plt.show()
plt.imshow(ToPILImage()(y))
plt.show()
plt.imshow(ToPILImage()(y_res))
plt.show()


generator = Generator(scale=2)
x = torch.ones((1, 3, 44, 44))
y = generator(x)
print(x.shape, y.shape)


discriminator = Discriminator()
x = torch.ones((1, 3, 88, 88))
y = discriminator(x)
print(x.shape, y.shape)
