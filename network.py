# Referring to https://github.com/Tobias-Fischer/rt_gene/blob/master/rt_gene/src/rt_gene/gaze_estimation_models_pytorch.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class GazeEstimationAbstractModel(nn.Module):

    def __init__(self):
        super(GazeEstimationAbstractModel, self).__init__()

    @staticmethod
    def _create_fc_layers(in_features, out_features):
        x_l = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True)
        )
        x_r = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True)
        )

        concat = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True)
        )

        fc = nn.Sequential(
            nn.Linear(514, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_features)
        )

        return x_l, x_r, concat, fc

    def forward(self, left_eye, right_eye, headpose):
        left_x = self.left_features(left_eye)
        left_x = torch.flatten(left_x, 1)
        left_x = self.xl(left_x)

        right_x = self.right_features(right_eye)
        right_x = torch.flatten(right_x, 1)
        right_x = self.xr(right_x)

        eyes_x = torch.cat((left_x, right_x), dim=1)
        eyes_x = self.concat(eyes_x)

        eyes_headpose = torch.cat((eyes_x, headpose), dim=1)

        fc_output = self.fc(eyes_headpose)

        return fc_output

    @staticmethod
    def _init_weights(modules):
        for m in modules:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)


class GazeEstimationModelVGG(GazeEstimationAbstractModel):

    def __init__(self, num_out=2):
        super(GazeEstimationModelVGG, self).__init__()
        _left_model = models.vgg16(pretrained=True)
        _right_model = models.vgg16(pretrained=True)

        # remove the last ConvBRelu layer
        _left_modules = [module for module in _left_model.features]
        _left_modules.append(_left_model.avgpool)
        self.left_features = nn.Sequential(*_left_modules)

        _right_modules = [module for module in _right_model.features]
        _right_modules.append(_right_model.avgpool)
        self.right_features = nn.Sequential(*_right_modules)

        for param in self.left_features.parameters():
            param.requires_grad = True
        for param in self.right_features.parameters():
            param.requires_grad = True

        self.xl, self.xr, self.concat, self.fc = GazeEstimationAbstractModel._create_fc_layers(
            in_features=_left_model.classifier[0].in_features,
            out_features=num_out)

        GazeEstimationAbstractModel._init_weights(self.modules())
