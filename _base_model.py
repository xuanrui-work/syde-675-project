import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class CNNEncoder(nn.Module):
    def __init__(
        self,
        input_shape=(3, 32, 32),
        conv_out_channels=(64, 128, 128),
        conv_kernel_sizes=(3, 3, 3),
        conv_strides=(1, 1, 1),
        pool_kernel_sizes=(2, 2, 2),
        pool_strides=(2, 2, 2),
        fc_out_features=None,
        activation_cls=nn.ReLU,
    ):
        super().__init__()

        self.input_shape = input_shape

        out_channels = input_shape[0]
        output_shape = input_shape
        conv_layers = nn.ModuleList()

        for i in range(len(conv_out_channels)):
            layer = nn.Sequential(
                nn.Conv2d(out_channels, conv_out_channels[i], conv_kernel_sizes[i], conv_strides[i]),
                activation_cls(),
                nn.MaxPool2d(pool_kernel_sizes[i], pool_strides[i]) if pool_kernel_sizes[i] is not None else nn.Identity(),
            )
            conv_layers.append(layer)

            h_out, w_out = output_shape[1:]
            h_out, w_out = self._conv_output_hw(h_out, w_out, conv_kernel_sizes[i], conv_strides[i], 0)
            if pool_kernel_sizes[i] is not None:
                h_out, w_out = self._pool_output_hw(h_out, w_out, pool_kernel_sizes[i], pool_strides[i], 0)
            
            out_channels = conv_out_channels[i]
            output_shape = (out_channels, h_out, w_out)
        
        self.output_shape = output_shape
        self.conv_layers = conv_layers
        
        fc_layers = None

        if fc_out_features:
            out_features = np.prod(output_shape).astype(int)
            fc_layers = nn.ModuleList()

            for i in range(len(fc_out_features)):
                layer = nn.Sequential(
                    nn.Linear(out_features, fc_out_features[i]),
                    activation_cls(),
                )
                fc_layers.append(layer)
                out_features = fc_out_features[i]
            
            self.output_shape = (out_features,)
        
        self.fc_layers = fc_layers
    
    def _conv_output_hw(self, h_in, w_in, kernel_size, stride, padding):
        h_out = np.floor((h_in + 2*padding - (kernel_size - 1) - 1) / stride + 1).astype(int)
        w_out = np.floor((w_in + 2*padding - (kernel_size - 1) - 1) / stride + 1).astype(int)
        return (h_out, w_out)
    
    def _pool_output_hw(self, h_in, w_in, kernel_size, stride, padding):
        h_out = np.floor((h_in + 2*padding - (kernel_size - 1) - 1) / stride + 1).astype(int)
        w_out = np.floor((w_in + 2*padding - (kernel_size - 1) - 1) / stride + 1).astype(int)
        return (h_out, w_out)
    
    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        if self.fc_layers is not None:
            x = torch.flatten(x, start_dim=1)
            for layer in self.fc_layers:
                x = layer(x)
        return x

class FCHead(nn.Module):
    def __init__(
        self,
        input_shape=(3, 32, 32),
        fc_out_features=(3072, 2048),
        op_out_features=10,
        activation_cls=nn.ReLU,
    ):
        super().__init__()

        if fc_out_features is None:
            fc_out_features = []

        out_features = np.prod(input_shape).astype(int)
        self.in_features = out_features
        fc_layers = nn.ModuleList()

        for i in range(len(fc_out_features)):
            layer = nn.Sequential(
                nn.Linear(out_features, fc_out_features[i]),
                activation_cls(),
            )
            fc_layers.append(layer)
            out_features = fc_out_features[i]
        
        fc_layers.append(nn.Linear(out_features, op_out_features))
        
        self.out_features = op_out_features
        self.fc_layers = fc_layers
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        for layer in self.fc_layers:
            x = layer(x)
        return x

class CNNClassifier(nn.Module):
    def __init__(
        self,
        input_shape=(3, 32, 32),
        num_classes=10,
        cnn_encoder_config=None,
        fc_head_config=None,
    ):
        super().__init__()

        if cnn_encoder_config is None:
            cnn_encoder_config = {}
        if fc_head_config is None:
            fc_head_config = {}
        
        encoder = CNNEncoder(
            input_shape=input_shape,
            **cnn_encoder_config
        )
        cls_head = FCHead(
            input_shape=encoder.output_shape,
            op_out_features=num_classes,
            **fc_head_config
        )

        self.encoder = encoder
        self.cls_head = cls_head
    
    def forward(self, x):
        features = self.encoder(x)
        features = torch.flatten(features, start_dim=1)
        y_pred = self.cls_head(features)
        return (y_pred, features)
