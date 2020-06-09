import torchvision.models as torch_models
from torch import nn


def build_model():
    model = torch_models.resnet50(pretrained=True)
    model = nn.Sequential(
        model.conv1,
        model.bn1,
        nn.ReLU(True),
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(start_dim=1)
    )
    return model
