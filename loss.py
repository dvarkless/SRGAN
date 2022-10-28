import torch
from torch import nn
from torchvision.models.resnet import resnet50, ResNet50_Weights


class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        loss_network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
        print(loss_network.__class__.__name__)
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)

        return image_loss + 0.001 * adversarial_loss + 0.007 * perception_loss


if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
