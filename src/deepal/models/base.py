from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Credits to https://github.com/sharpenb/Posterior-Network/tree/main/src/architectures


def linear_sequential(input_dims, hidden_dims, output_dim, p_drop=None):
    dims = [np.prod(input_dims)] + hidden_dims + [output_dim]
    num_layers = len(dims) - 1
    layers = []
    for i in range(num_layers):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < num_layers - 1:
            layers.append(nn.ReLU())
            if p_drop is not None:
                layers.append(nn.Dropout(p=p_drop))
    return nn.Sequential(*layers)


class AbstractModel(nn.Module):
    def __init__(self, input_dims, output_dim, linear_hidden_dims, p_drop):
        super().__init__()
        self.input_dims, self.output_dim, self.linear_hidden_dims = input_dims, output_dim, linear_hidden_dims
        self.p_drop = p_drop
        self.emb_dim = linear_hidden_dims[-1]

    @abstractmethod
    def init_model(self):
        pass

    def get_embedding_dim(self):
        return self.emb_dim

    @staticmethod
    def loss_fc(soft_output_pred, soft_output):
        return F.cross_entropy(soft_output_pred, soft_output)


class LinearSeq(AbstractModel):
    def __init__(self, input_dims, output_dim, linear_hidden_dims, p_drop):
        super().__init__(input_dims, output_dim, linear_hidden_dims, p_drop)
        self.linear = self.init_model()

    def init_model(self):
        return linear_sequential(
            input_dims=self.input_dims,
            hidden_dims=self.linear_hidden_dims,
            output_dim=self.output_dim,
            p_drop=self.p_drop
        )

    def forward(self, x: torch.FloatTensor):
        batch_size = x.size(0)
        output = self.linear(x.view(batch_size, -1))
        feature_extractor = torch.nn.Sequential(*list(self.linear.children())[:-1])
        features = feature_extractor(x.view(batch_size, -1))
        return output, features
