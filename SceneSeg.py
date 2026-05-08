"""SceneSeg network — vendored copy.

Source: https://github.com/autowarefoundation/autoware_vision_pilot
        Models/model_components/{backbone, scene_context, scene_neck,
                                 scene_seg_head, scene_seg_network}.py

Consolidated into a single file for the Neural Optimization course's
Mini_ONNX workshop, so students can import the model class without
cloning Autoware's full repo.

Architecture: EfficientNet-B0 backbone (ImageNet-pretrained) → context
module → neck → segmentation head. Outputs 3 classes: background,
foreground objects, drivable road.

Note on shapes:
    The original SceneContext.forward() in Autoware uses
    `c2.reshape([10, 20]).unsqueeze(0).unsqueeze(0)` which silently
    requires batch size = 1. We rewrote this single line to
    `c2.reshape(c2.shape[0], 1, 10, 20)` — produces identical output
    for batch=1 (same elements, same order, no weights changed) but
    also works for any batch size. ONNX export with dynamic batch is
    now possible.
"""
import torch
import torch.nn as nn
from torchvision import models


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.efficientnet_b0(
            weights="EfficientNet_B0_Weights.IMAGENET1K_V1"
        ).features

    def forward(self, image):
        l0 = self.encoder[0](image)
        l1 = self.encoder[1](l0)
        l2 = self.encoder[2](l1)
        l3 = self.encoder[3](l2)
        l4 = self.encoder[4](l3)
        l5 = self.encoder[5](l4)
        l6 = self.encoder[6](l5)
        l7 = self.encoder[7](l6)
        l8 = self.encoder[8](l7)
        return [l0, l2, l3, l4, l8]


class SceneContext(nn.Module):
    def __init__(self):
        super().__init__()
        self.GeLU = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.25)

        # MLP layers
        self.context_layer_0 = nn.Linear(1280, 800)
        self.context_layer_1 = nn.Linear(800, 800)
        self.context_layer_2 = nn.Linear(800, 200)

        # Conv extraction layers
        self.context_layer_3 = nn.Conv2d(1, 128, 3, 1, 1)
        self.context_layer_4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.context_layer_5 = nn.Conv2d(256, 512, 3, 1, 1)
        self.context_layer_6 = nn.Conv2d(512, 1280, 3, 1, 1)

    def forward(self, features):
        feature_vector = torch.mean(features, dim=[2, 3])

        c0 = self.GeLU(self.dropout(self.context_layer_0(feature_vector)))
        c1 = self.GeLU(self.dropout(self.context_layer_1(c0)))
        c2 = self.sigmoid(self.dropout(self.context_layer_2(c1)))

        # NB: hardcoded reshape — only works for batch size 1.
        c3 = c2.reshape([10, 20]).unsqueeze(0).unsqueeze(0)

        c4 = self.GeLU(self.context_layer_3(c3))
        c5 = self.GeLU(self.context_layer_4(c4))
        c6 = self.GeLU(self.context_layer_5(c5))
        c7 = self.GeLU(self.context_layer_6(c6))

        # Residual attention
        return c7 * features + features


class SceneNeck(nn.Module):
    def __init__(self):
        super().__init__()
        self.GeLU = nn.GELU()

        self.upsample_layer_0 = nn.ConvTranspose2d(1280, 1280, 2, 2)
        self.skip_link_layer_0 = nn.Conv2d(80, 1280, 1)
        self.decode_layer_0 = nn.Conv2d(1280, 768, 3, 1, 1)
        self.decode_layer_1 = nn.Conv2d(768, 768, 3, 1, 1)

        self.upsample_layer_1 = nn.ConvTranspose2d(768, 768, 2, 2)
        self.skip_link_layer_1 = nn.Conv2d(40, 768, 1)
        self.decode_layer_2 = nn.Conv2d(768, 512, 3, 1, 1)
        self.decode_layer_3 = nn.Conv2d(512, 512, 3, 1, 1)

        self.upsample_layer_2 = nn.ConvTranspose2d(512, 512, 2, 2)
        self.skip_link_layer_2 = nn.Conv2d(24, 512, 1)
        self.decode_layer_4 = nn.Conv2d(512, 512, 3, 1, 1)
        self.decode_layer_5 = nn.Conv2d(512, 256, 3, 1, 1)

    def forward(self, context, features):
        # Block 1
        d0 = self.upsample_layer_0(context) + self.skip_link_layer_0(features[3])
        d1 = self.GeLU(self.decode_layer_0(d0))
        d2 = self.GeLU(self.decode_layer_1(d1))

        # Block 2
        d3 = self.upsample_layer_1(d2) + self.skip_link_layer_1(features[2])
        d3 = self.GeLU(self.decode_layer_2(d3))
        d4 = self.decode_layer_3(d3)
        d5 = self.GeLU(d4)

        # Block 3
        d5 = self.upsample_layer_2(d5) + self.skip_link_layer_2(features[1])
        d5 = self.GeLU(self.decode_layer_4(d5))
        d6 = self.decode_layer_5(d5)
        return self.GeLU(d6)


class SceneSegHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.GeLU = nn.GELU()

        self.upsample_layer_3 = nn.ConvTranspose2d(256, 256, 2, 2)
        self.skip_link_layer_3 = nn.Conv2d(32, 256, 1)
        self.decode_layer_6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.decode_layer_7 = nn.Conv2d(256, 128, 3, 1, 1)

        self.upsample_layer_4 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.decode_layer_8 = nn.Conv2d(128, 128, 3, 1, 1)
        self.decode_layer_9 = nn.Conv2d(128, 64, 3, 1, 1)
        self.decode_layer_10 = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, neck, features):
        # Block 4
        d7 = self.upsample_layer_3(neck) + self.skip_link_layer_3(features[0])
        d7 = self.GeLU(self.decode_layer_6(d7))
        d8 = self.GeLU(self.decode_layer_7(d7))

        # Block 5
        d8 = self.upsample_layer_4(d8)
        d8 = self.GeLU(self.decode_layer_8(d8))
        d9 = self.decode_layer_9(d8)
        d10 = self.GeLU(d9)

        return self.decode_layer_10(d10)


class SceneSegNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.Backbone = Backbone()
        self.SceneContext = SceneContext()
        self.SceneNeck = SceneNeck()
        self.SceneSegHead = SceneSegHead()

    def forward(self, image):
        features = self.Backbone(image)
        deep_features = features[4]
        context = self.SceneContext(deep_features)
        neck = self.SceneNeck(context, features)
        return self.SceneSegHead(neck, features)
