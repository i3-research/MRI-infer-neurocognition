import sys
import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import monai
from monai.data import ImageDataset
from monai.networks.layers.factories import Conv, Dropout, Pool, Norm
from monai.transforms import Compose, AddChannel, ScaleIntensity, EnsureType
from collections import OrderedDict
from typing import Callable, Sequence
import argparse

pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, choices=['resnet', 'densenet'])
    parser.add_argument('--im_type', type=str, choices=['int', 'rav', 'int_rav'])
    parser.add_argument('--iq', type=str, choices=['all', 'fiq', 'viq', 'piq'])
    parser.add_argument('--iq_type', type=str, choices=['absolute', 'residual'])
    parser.add_argument('--folds', type=int, default=0, help='fold number')
    args = parser.parse_args()
    return args

# ----------------------- Densenet -------------------------# 
class _DenseLayer(nn.Sequential):
    def __init__(
        self, spatial_dims: int, in_channels: int, growth_rate: int, bn_size: int, dropout_prob: float
    ) -> None:
        super(_DenseLayer, self).__init__()

        out_channels = bn_size * growth_rate
        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]
        dropout_type: Callable = Dropout[Dropout.DROPOUT, spatial_dims]

        self.add_module("norm1", norm_type(in_channels))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", conv_type(in_channels, out_channels, kernel_size=1, bias=False))

        self.add_module("norm2", norm_type(out_channels))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", conv_type(out_channels, growth_rate, kernel_size=3, padding=1, bias=False))

        if dropout_prob > 0:
            self.add_module("dropout", dropout_type(dropout_prob))

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(
        self, spatial_dims: int, layers: int, in_channels: int, bn_size: int, growth_rate: int, dropout_prob: float
    ) -> None:
        super(_DenseBlock, self).__init__()
        for i in range(layers):
            layer = _DenseLayer(spatial_dims, in_channels, growth_rate, bn_size, dropout_prob)
            in_channels += growth_rate
            self.add_module("denselayer%d" % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int) -> None:
        super(_Transition, self).__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]
        pool_type: Callable = Pool[Pool.AVG, spatial_dims]

        self.add_module("norm", norm_type(in_channels))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", conv_type(in_channels, out_channels, kernel_size=1, bias=False))
        self.add_module("pool", pool_type(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """
    Densenet based on: "Densely Connected Convolutional Networks" https://arxiv.org/pdf/1608.06993.pdf
    Adapted from PyTorch Hub 2D version:
    https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

    Args:
        spatial_dims: number of spatial dimensions of the input image.
        in_channels: number of the input channel.
        out_channels: number of the output classes.
        init_features: number of filters in the first convolution layer.
        growth_rate: how many filters to add each layer (k in paper).
        block_config: how many layers in each pooling block.
        bn_size: multiplicative factor for number of bottle neck layers.
                      (i.e. bn_size * k features in the bottleneck layer)
        dropout_prob: dropout rate after each dense layer.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 24, 16),
        bn_size: int = 4,
        dropout_prob: float = 0.0,
    ) -> None:

        super(DenseNet, self).__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]
        pool_type: Callable = Pool[Pool.MAX, spatial_dims]
        avg_pool_type: Callable = Pool[Pool.ADAPTIVEAVG, spatial_dims]

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", conv_type(in_channels, init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("norm0", norm_type(init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", pool_type(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        in_channels = init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                spatial_dims=spatial_dims,
                layers=num_layers,
                in_channels=in_channels,
                bn_size=bn_size,
                growth_rate=growth_rate,
                dropout_prob=dropout_prob,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            in_channels += num_layers * growth_rate
            if i == len(block_config) - 1:
                self.features.add_module("norm5", norm_type(in_channels))
            else:
                _out_channels = in_channels // 2
                trans = _Transition(spatial_dims, in_channels=in_channels, out_channels=_out_channels)
                self.features.add_module("transition%d" % (i + 1), trans)
                in_channels = _out_channels

        # pooling and classification
        self.class_layers = nn.Sequential(
            OrderedDict(
                [
                    ("relu", nn.ReLU(inplace=True)),
                    ("norm", avg_pool_type(1)),
                    ("flatten", nn.Flatten(1)), 
                    ("class", nn.Linear(in_channels, out_channels)),
                ]
            )
        )
        '''
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(1024*4*4*4, 1024)
        self.fc2 = nn.Linear(1024, out_channels)

        '''
        # Avoid Built-in function isinstance was called with the wrong arguments warning
        # pytype: disable=wrong-arg-types
        for m in self.modules():
            if isinstance(m, conv_type):  # type: ignore
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, norm_type):  # type: ignore
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
        # pytype: enable=wrong-arg-types

    def forward(self, x):
        x = self.features(x)
        x = self.class_layers(x) # remove if fc1/fc2 
        #x = torch.flatten(x, 1)
        #x = self.fc1(x)
        #x = self.fc2(x)
        return x


def densenet121(**kwargs) -> DenseNet:
    model = DenseNet(init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), **kwargs)
    return model


def densenet169(**kwargs) -> DenseNet:
    model = DenseNet(init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), **kwargs)
    return model


def densenet201(**kwargs) -> DenseNet:
    model = DenseNet(init_features=64, growth_rate=32, block_config=(6, 12, 48, 32), **kwargs)
    return model


def densenet264(**kwargs) -> DenseNet:
    model = DenseNet(init_features=64, growth_rate=32, block_config=(6, 12, 64, 48), **kwargs)
    return model

# ------------------------------ Resnet ----------------------------------#

def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
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

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
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

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model


def main():
    opts = parse_options()
    data_dir = '/home/marafath/scratch/abide_iq_combined'

    np.random.seed(42)   
    idx = np.random.permutation(850)
    if opts.im_type == 'int':
        datalist = os.listdir(os.path.join(data_dir, "int"))
        data_dir2 = os.path.join(data_dir, "int")
    elif opts.im_type == 'rav':
        datalist = os.listdir(os.path.join(data_dir, "rav"))
        data_dir2 = os.path.join(data_dir, "rav")
    elif opts.im_type == 'int_rav':
        datalist = os.listdir(os.path.join(data_dir, "int_rav"))
        data_dir2 = os.path.join(data_dir, "int_rav")

    fold = opts.folds
    train_image = []
    train_label = []
    validation_image = []
    validation_label = []

    start = fold*170
    end = (fold+1)*170

    for i in range(len(datalist)):
        nifti_name = datalist[idx[i]]
        if opts.iq == 'all':
            if opts.iq_type == 'absolute':
                iq_label = [float(nifti_name.split('_')[2]), float(nifti_name.split('_')[6]), float(nifti_name.split('_')[10])]
            else:
                iq_label = [float(nifti_name.split('_')[4]), float(nifti_name.split('_')[8]), float(nifti_name.split('_')[12])]
            no_out_classes = 3
        elif opts.iq == 'fiq':
            if opts.iq_type == 'absolute':
                iq_label = float(nifti_name.split('_')[2])
            else:
                iq_label = float(nifti_name.split('_')[4])
            no_out_classes = 1
        elif opts.iq == 'viq':
            if opts.iq_type == 'absolute':
                iq_label = float(nifti_name.split('_')[6])
            else:
                iq_label = float(nifti_name.split('_')[8])
            no_out_classes = 1
        elif opts.iq == 'piq':
            if opts.iq_type == 'absolute':
                iq_label = float(nifti_name.split('_')[10])
            else:
                iq_label = float(nifti_name.split('_')[12])
            no_out_classes = 1
            
        if i >= start and i < end:
            validation_image.append(os.path.join(data_dir2, nifti_name))
            validation_label.append(iq_label)
        else:
            train_image.append(os.path.join(data_dir2, nifti_name))
            train_label.append(iq_label)
            
    train_label = torch.tensor(train_label)
    validation_label = torch.tensor(validation_label)
            
    # Define transforms
    train_transforms = Compose([ScaleIntensity(), AddChannel(), EnsureType()])
    val_transforms = Compose([ScaleIntensity(), AddChannel(), EnsureType()])
    
    # create a training data loader
    train_ds = ImageDataset(image_files=train_image, labels=train_label, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=1, pin_memory=pin_memory)

    # create a validation data loader
    val_ds = ImageDataset(image_files=validation_image, labels=validation_label, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=1, pin_memory=pin_memory)
    
    # Create DenseNet121/ResNet18, MSELoss and Adam optimizer
    if opts.arch == 'densenet':
        model = densenet121(spatial_dims=3, in_channels=1, out_channels=no_out_classes).to(device)
    elif opts.arch == 'resnet':
        model = generate_model(model_depth=18, n_input_channels=1, n_classes=no_out_classes, shortcut_type='B').to(device)
        
    loss_function = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    
    def mae(t1, t2):
        diff = t1 - t2
        return torch.absolute(diff) #/ diff.numel()
    
    val_interval = 1
    best_metric = 1e5
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    writer = SummaryWriter()
    total_loss = 0
    max_epochs = 50
    
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.squeeze(-1)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
        if (epoch + 1) % val_interval == 0:
            model.eval()
            metric_count = 0
            total_loss = 0
            best_predicted = np.array([])
            actual_label = np.array([])
            for val_data in val_loader:
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                with torch.no_grad():
                    val_outputs = model(val_images)
                    val_outputs = val_outputs.squeeze(-1)
                    val_loss = mae(val_outputs, val_labels)
                    metric_count += len(val_loss)
                    total_loss += torch.sum(val_loss)
                    best_predicted = np.append(best_predicted, val_outputs.detach().cpu().numpy()) 
                    actual_label = np.append(actual_label, val_labels.cpu().detach().numpy())

            metric = total_loss / metric_count
            metric_values.append(metric)

            if metric < best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), f'/home/marafath/scratch/abide_saved_models/{opts.im_type}_{opts.arch}_{opts.iq}_{opts.iq_type}_f{opts.folds}.pth')
                np.save(f'predicted_{opts.im_type}_{opts.arch}_{opts.iq}_{opts.iq_type}_f{opts.folds}', best_predicted)
                np.save(f'actual_{opts.im_type}_{opts.arch}_{opts.iq}_{opts.iq_type}_f{opts.folds}', actual_label)
                print("saved new best metric model")

            print(f"Current epoch: {epoch+1} current accuracy: {metric:.4f} ")
            print(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")
            writer.add_scalar("val_accuracy", metric, epoch + 1)

    print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()
    
if __name__ == '__main__':
    main()
    


