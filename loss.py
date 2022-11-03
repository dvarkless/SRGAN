import torch
from torch import nn
from torchvision.models.resnet import resnet50, ResNet50_Weights


class FullLoss(nn.Module):
    def __init__(self):
        super().__init__()
        loss_network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.clear_losses()


    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = -torch.log(out_labels)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        self.image_loss = self.image_loss + image_loss
        self.adv_loss = self.adv_loss + 0.001 * adversarial_loss
        self.perc_loss = self.perc_loss + 0.007 * perception_loss

        return image_loss + 0.001 * adversarial_loss + 0.004 * perception_loss

    def get_losses(self):
        return self.image_loss, self.adv_loss, self.perc_loss

    def clear_losses(self):
        self.image_loss = 0.0
        self.adv_loss = 0.0
        self.perc_loss = 0.0


class AdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        loss_network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.clear_losses()


    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = -torch.log(out_labels)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        self.adv_loss = self.adv_loss + 0.001 * adversarial_loss
        self.perc_loss = self.perc_loss + 0.007 * perception_loss

        return adversarial_loss + 0.1 * perception_loss

    def get_losses(self):
        return 0, self.adv_loss, self.perc_loss

    def clear_losses(self):
        self.adv_loss = 0.0
        self.perc_loss = 0.0


if __name__ == "__main__":
    g_loss = FullLoss()
    print(g_loss)
