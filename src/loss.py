import torch
import torchvision


class TotalVarianceLoss(torch.nn.Module):

    def __init__(self, tv_loss_weight=1):
        super(TotalVarianceLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class GeneratorLoss(torch.nn.Module):

    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg16 = torchvision.models.vgg.vgg16(pretrained=True)
        network = torch.nn.Sequential(*list(vgg16.features)[:31]).eval()
        for param in network.parameters():
            param.requires_grad = False
        self.network = network
        self.mse = torch.nn.MSELoss()
        self.tv = TotalVarianceLoss()

    def forward(self, labels, images, targets):
        adversarial_loss = torch.mean(1 - labels)
        perception_loss = self.mse(
            self.network(images),
            self.network(targets)
        )
        image_loss = self.mse(images, targets)
        tv_loss = self.tv(images)
        loss = image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss
        return loss
