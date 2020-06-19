import torchvision.models as torch_models
from torch import nn
import copy


class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        model = torch_models.resnet50(pretrained=True)
        self.model = nn.Sequential(
            model.conv1,
            model.bn1,
            nn.ReLU(True),
            model.layer1,
            model.layer2,
            model.layer3,   
        )
        self.class_head = nn.Sequential(
            model.layer4,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(2048, num_classes)
        )
        self.signature_head = nn.Sequential(
            copy.deepcopy(model.layer4),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1)
        )
    def forward(self, x):
        x = self.model(x)
        features = self.signature_head(x)
        clas = self.class_head(x)
        return features, clas

def build_model(num_classes):
    # model = torch_models.resnet50(pretrained=True)
    # model = nn.Sequential(
    #     model.conv1,
    #     model.bn1,
    #     nn.ReLU(True),
    #     model.layer1,
    #     model.layer2,
    #     model.layer3,
    #     model.layer4,
    #     nn.AdaptiveAvgPool2d(1),
    #     nn.Flatten(start_dim=1)
    # )
    # return model
    return Model(num_classes)
