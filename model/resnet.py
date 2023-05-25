import torch.nn as nn
import torchvision


class ResNet(nn.Module):
    def __init__(self, arch='resnet18', num_classes=200, pretrained=True, activation='tanh', classifier='linear'):
        super().__init__()
        assert arch in torchvision.models.__dict__.keys(), f'{arch} is not supported!'
        resnet = torchvision.models.__dict__[arch](pretrained=pretrained)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.feat_dim = resnet.fc.in_features
        self.neck = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        if classifier == 'linear':
            self.classfier_head = nn.Linear(in_features=self.feat_dim, out_features=num_classes)
            init_weights(self.classfier_head, init_method='He')
        # elif classifier.startswith('mlp'):
        #     sf = float(classifier.split('-')[1])
        #     self.classfier_head = MLPHead(self.feat_dim, mlp_scale_factor=sf, projection_size=num_classes, init_method='He', activation='relu')
        else:
            raise AssertionError(f'{classifier} classifier is not supported.')
        # self.project = nn.Linear(in_features=self.feat_dim, out_features=256)
        # init_weights(self.project, init_method='He')

    def forward(self, x):
        N = x.size(0)
        x = self.backbone(x)
        x = self.neck(x).view(N, -1)
        logits = self.classfier_head(x)
        # feat = self.project(x)
        return logits
    
class ResNetfeat(nn.Module):
    def __init__(self, arch='resnet18', num_classes=200, pretrained=True, activation='tanh', classifier='linear', project_dim=128):
        super().__init__()
        assert arch in torchvision.models.__dict__.keys(), f'{arch} is not supported!'
        resnet = torchvision.models.__dict__[arch](pretrained=pretrained)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.feat_dim = resnet.fc.in_features
        # self.project_dim = project_dim
        self.neck = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        if classifier == 'linear':
            self.classfier_head = nn.Linear(in_features=self.feat_dim, out_features=num_classes)
            init_weights(self.classfier_head, init_method='He')
        # elif classifier.startswith('mlp'):
        #     sf = float(classifier.split('-')[1])
        #     self.classfier_head = MLPHead(self.feat_dim, mlp_scale_factor=sf, projection_size=num_classes, init_method='He', activation='relu')
        else:
            raise AssertionError(f'{classifier} classifier is not supported.')
        # self.project = nn.Linear(in_features=self.feat_dim, out_features= self.project_dim)
        # init_weights(self.project, init_method='He')

    def forward(self, x):
        N = x.size(0)
        x = self.backbone(x)
        feat = self.neck(x).view(N, -1)
        logits = self.classfier_head(feat)
        # feat = self.project(x)
        return logits, feat

class ResNetfeatproj(nn.Module):
    def __init__(self, arch='resnet18', num_classes=200, pretrained=True, activation='tanh', classifier='linear', project_dim=128):
        super().__init__()
        assert arch in torchvision.models.__dict__.keys(), f'{arch} is not supported!'
        resnet = torchvision.models.__dict__[arch](pretrained=pretrained)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.feat_dim = resnet.fc.in_features
        self.project_dim = project_dim
        self.neck = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        if classifier == 'linear':
            self.classfier_head = nn.Linear(in_features=self.feat_dim, out_features=num_classes)
            init_weights(self.classfier_head, init_method='He')
        # elif classifier.startswith('mlp'):
        #     sf = float(classifier.split('-')[1])
        #     self.classfier_head = MLPHead(self.feat_dim, mlp_scale_factor=sf, projection_size=num_classes, init_method='He', activation='relu')
        else:
            raise AssertionError(f'{classifier} classifier is not supported.')
        self.project = nn.Linear(in_features=self.feat_dim, out_features= self.project_dim)
        init_weights(self.project, init_method='He')

    def forward(self, x):
        N = x.size(0)
        x = self.backbone(x)
        x = self.neck(x).view(N, -1)
        logits = self.classfier_head(x)
        feat = self.project(x)
        return logits, feat
    
    
def init_weights(module, init_method='He'):
    for _, m in module.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if init_method == 'He':
                nn.init.kaiming_normal_(m.weight.data)
            elif init_method == 'Xavier':
                nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, val=0)