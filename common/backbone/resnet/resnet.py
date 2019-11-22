"""
Modified from torchvision, but exposes features from different stages
"""

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import warnings

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

model_layers = {
    'resnet18': [2, 2, 2, 2],
    'resnet34': [3, 4, 6, 3],
    'resnet50': [3, 4, 6, 3],
    'resnet101': [3, 4, 23, 3],
    'resnet152': [3, 8, 36, 3],
}


def conv3x3(in_planes, out_planes, stride=1, dilation=1, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, dilation=dilation,
                     padding=padding, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, **kwargs):
        super(BasicBlock, self).__init__()
        # if dilation == 1:
        #     self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        # elif dilation == 2:
        #     self.conv1 = conv3x3(inplanes, planes, stride, dilation, padding=2)
        # else:
        #     raise ValueError('dilation must be 1 or 2!')
        self.conv1 = conv3x3(inplanes, planes, stride, dilation, padding=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, stride_in_1x1=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1 if not stride_in_1x1 else stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # if dilation == 1:
        #     self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride if not stride_in_1x1 else 1,
        #                            dilation=dilation, padding=1, bias=False)
        # elif dilation == 2:
        #     self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride if not stride_in_1x1 else 1,
        #                            dilation=dilation, padding=2, bias=False)
        # else:
        #     raise ValueError('dilation must be 1 or 2!')
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride if not stride_in_1x1 else 1,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=None, expose_stages=None, dilations=None, stride_in_1x1=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        layers_planes = [64, 128, 256, 512]
        layers_strides = [1, 2, 2, 2]
        layers_dilations = dilations if dilations is not None else [1, 1, 1, 1]
        for i, dilation in enumerate(layers_dilations):
            if dilation == 2:
                layers_strides[i] = 1
        layers_planes = layers_planes[:len(layers)]
        layers_strides = layers_strides[:len(layers)]
        layers_dilations = layers_dilations[:len(layers)]
        for i, (planes, blocks, stride, dilation) in enumerate(zip(layers_planes, layers, layers_strides, layers_dilations)):
            layer = self._make_layer(block, planes, blocks, stride=stride, dilation=dilation, stride_in_1x1=stride_in_1x1)
            self.__setattr__('layer{}'.format(i + 1), layer)
        self.num_layers = i + 1
        self.has_fc_head = 6 in expose_stages
        self.expose_stages = expose_stages
        if self.has_fc_head:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)
            self.expose_stages.remove(6)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, stride_in_1x1=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation, stride_in_1x1=stride_in_1x1))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        expose_feats = {}
        feats = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        feats['body1'] = x

        for i in range(self.num_layers):
            x = self.__getattr__("layer{}".format(i + 1))(x)
            feats['body{}'.format(i + 2)] = x

        if self.has_fc_head:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            expose_feats['cls_score'] = x

        if self.expose_stages is not None:
            for expose_stage in self.expose_stages:
                feat_name = 'body{}'.format(expose_stage)
                expose_feats[feat_name] = feats[feat_name]

        return expose_feats

    def load_pretrained_state_dict(self, state_dict):
        """Load state dict of pretrained model
        Args:
            state_dict (dict): state dict to load
        """
        new_state_dict = self.state_dict()
        miss_keys = []
        for k in new_state_dict.keys():
            if k in state_dict.keys():
                new_state_dict[k] = state_dict[k]
            else:
                miss_keys.append(k)
        if len(miss_keys) > 0:
            warnings.warn('miss keys: {}'.format(miss_keys))
        self.load_state_dict(new_state_dict)

    def frozen_parameters(self, frozen_stages=None, frozen_bn=False):
        if frozen_bn:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    for param in module.parameters():
                        param.requires_grad = False
        if frozen_stages is not None:
            for stage in frozen_stages:
                assert (stage >= 1) and (stage <= 6)
                if stage == 1:
                    for param in self.conv1.parameters():
                        param.requires_grad = False
                    for param in self.bn1.parameters():
                        param.requires_grad = False
                elif stage < 6:
                    for param in self.__getattr__("layer{}".format(stage - 1)).parameters():
                        param.requires_grad = False
                else:
                    for param in self.fc.parameters():
                        param.requires_grad = False

    def bn_eval(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


def resnet18(pretrained=False, pretrained_model_path=None, num_classes=None, expose_stages=None, dilations=None, **kwargs):
    """Constructs a ResNet-18 model
    Args:
        pretrained (bool): if True, load pretrained model. Default: False
        pretrained_model_path (str, optional): only effective when pretrained=True,
                                            if not specified, use pretrained model from model_zoo.
        num_classes (int): number of classes for the fc output score.
        expose_stages (list, optional): list of expose stages, e.g. [4, 5] means expose conv4 and conv5 stage output.
                                        if not specified, only expose output of end_stage.
    """

    if num_classes is None:
        assert expose_stages is not None, "num_class and expose_stages is both None"
        assert 6 not in expose_stages, "can't expose the 6th stage for num_classes is None"

    if expose_stages is None:
        expose_stages = [6]

    end_stage = max(expose_stages)
    assert end_stage <= 6, "the max expose_stage is out of range"

    layers = model_layers['resnet18'][:end_stage - 1]

    model = ResNet(block=BasicBlock, layers=layers, num_classes=num_classes, expose_stages=expose_stages, dilations=dilations)

    if pretrained:
        if pretrained_model_path is not None:
            state_dict = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)
        else:
            state_dict = model_zoo.load_url(model_urls['resnet18'])
        model.load_pretrained_state_dict(state_dict)
    return model


def resnet34(pretrained=False, pretrained_model_path=None, num_classes=None, expose_stages=None, dilations=None, **kwargs):
    """Constructs a ResNet-34 model
    Args:
        pretrained (bool): if True, load pretrained model. Default: False
        pretrained_model_path (str, optional): only effective when pretrained=True,
                                            if not specified, use pretrained model from model_zoo.
        num_classes (int): number of classes for the fc output score.
        expose_stages (list, optional): list of expose stages, e.g. [4, 5] means expose conv4 and conv5 stage output.
                                        if not specified, only expose output of end_stage.
    """

    if num_classes is None:
        assert expose_stages is not None, "num_class and expose_stages is both None"
        assert 6 not in expose_stages, "can't expose the 6th stage for num_classes is None"

    if expose_stages is None:
        expose_stages = [6]

    end_stage = max(expose_stages)
    assert end_stage <= 6, "the max expose_stage is out of range"

    layers = model_layers['resnet34'][:end_stage - 1]

    model = ResNet(block=BasicBlock, layers=layers, num_classes=num_classes, expose_stages=expose_stages,
                   dilations=dilations)

    if pretrained:
        if pretrained_model_path is not None:
            state_dict = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)
        else:
            state_dict = model_zoo.load_url(model_urls['resnet34'])
        model.load_pretrained_state_dict(state_dict)
    return model


def resnet50(pretrained=False, pretrained_model_path=None, num_classes=None, expose_stages=None, dilations=None, stride_in_1x1=False):
    """Constructs a ResNet-50 model
    Args:
        pretrained (bool): if True, load pretrained model. Default: False
        pretrained_model_path (str, optional): only effective when pretrained=True,
                                            if not specified, use pretrained model from model_zoo.
        num_classes (int): number of classes for the fc output score.
        expose_stages (list, optional): list of expose stages, e.g. [4, 5] means expose conv4 and conv5 stage output.
                                        if not specified, only expose output of end_stage.
    """

    if num_classes is None:
        assert expose_stages is not None, "num_class and expose_stages is both None"
        assert 6 not in expose_stages, "can't expose the 6th stage for num_classes is None"

    if expose_stages is None:
        expose_stages = [6]

    end_stage = max(expose_stages)
    assert end_stage <= 6, "the max expose_stage is out of range"

    layers = model_layers['resnet50'][:end_stage - 1]

    model = ResNet(block=Bottleneck, layers=layers, num_classes=num_classes, expose_stages=expose_stages,
                   dilations=dilations, stride_in_1x1=stride_in_1x1)

    if pretrained:
        if pretrained_model_path is not None:
            state_dict = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)
        else:
            state_dict = model_zoo.load_url(model_urls['resnet50'])
        model.load_pretrained_state_dict(state_dict)
    return model


def resnet101(pretrained=False, pretrained_model_path=None, num_classes=None, expose_stages=None, dilations=None, stride_in_1x1=False):
    """Constructs a ResNet-101 model
    Args:
        pretrained (bool): if True, load pretrained model. Default: False
        pretrained_model_path (str, optional): only effective when pretrained=True,
                                            if not specified, use pretrained model from model_zoo.
        num_classes (int): number of classes for the fc output score.
        expose_stages (list, optional): list of expose stages, e.g. [4, 5] means expose conv4 and conv5 stage output.
                                        if not specified, only expose output of end_stage.
    """

    if num_classes is None:
        assert expose_stages is not None, "num_class and expose_stages is both None"
        assert 6 not in expose_stages, "can't expose the 6th stage for num_classes is None"

    if expose_stages is None:
        expose_stages = [6]

    end_stage = max(expose_stages)
    assert end_stage <= 6, "the max expose_stage is out of range"

    layers = model_layers['resnet101'][:end_stage - 1]

    model = ResNet(block=Bottleneck, layers=layers, num_classes=num_classes, expose_stages=expose_stages,
                   dilations=dilations, stride_in_1x1=stride_in_1x1)

    if pretrained:
        if pretrained_model_path is not None:
            state_dict = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)
        else:
            state_dict = model_zoo.load_url(model_urls['resnet101'])
        model.load_pretrained_state_dict(state_dict)
    return model


def resnet152(pretrained=False, pretrained_model_path=None, num_classes=None, expose_stages=None, dilations=None, stride_in_1x1=False):
    """Constructs a ResNet-152 model
    Args:
        pretrained (bool): if True, load pretrained model. Default: False
        pretrained_model_path (str, optional): only effective when pretrained=True,
                                            if not specified, use pretrained model from model_zoo.
        num_classes (int): number of classes for the fc output score.
        expose_stages (list, optional): list of expose stages, e.g. [4, 5] means expose conv4 and conv5 stage output.
                                        if not specified, only expose output of end_stage.
    """

    if num_classes is None:
        assert expose_stages is not None, "num_class and expose_stages is both None"
        assert 6 not in expose_stages, "can't expose the 6th stage for num_classes is None"

    if expose_stages is None:
        expose_stages = [6]

    end_stage = max(expose_stages)
    assert end_stage <= 6, "the max expose_stage is out of range"

    layers = model_layers['resnet152'][:end_stage - 1]

    model = ResNet(block=Bottleneck, layers=layers, num_classes=num_classes, expose_stages=expose_stages,
                   dilations=dilations, stride_in_1x1=stride_in_1x1)

    if pretrained:
        if pretrained_model_path is not None:
            state_dict = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)
        else:
            state_dict = model_zoo.load_url(model_urls['resnet152'])
        model.load_pretrained_state_dict(state_dict)
    return model
