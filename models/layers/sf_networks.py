import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import MaxPool2d, MaxUnpool2d
from utils.utils import calculate_flatten_size, check_output_padding_needed
from models.layers.building_blocks import MLP, Conv, DeConv


class Permute(nn.Module):
    """
    Given dimensions (0, 3, 1, 2), it permutes the tensors to given dim.
    It was created with nn.Module to create a sequential module using nn.Sequaltial()
    """

    def __init__(self, dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)


class Reshape(nn.Module):
    """
    Given dimension k in [N, k], it divides into [N, ?, 4, 4] where ? * 4 * 4 = k
    It was created with nn.Module to create a sequential module using nn.Sequaltial()
    """

    def __init__(self, fc_dim, reduced_feature_dim):
        super(Reshape, self).__init__()
        self.fc_dim = fc_dim
        self.reduced_feature_dim = reduced_feature_dim

    def forward(self, x):
        N = x.shape[0]
        if torch.numel(x) < N * self.reduced_feature_dim * self.reduced_feature_dim:
            return x.view(N, -1, self.reduced_feature_dim, 1)
        else:
            return x.view(N, -1, self.reduced_feature_dim, self.reduced_feature_dim)


class EncoderLastAct(nn.Module):
    """
    Given dimension k in [N, k], it divides into [N, ?, 4, 4] where ? * 4 * 4 = k
    It was created with nn.Module to create a sequential module using nn.Sequaltial()
    """

    def __init__(self, alpha):
        super(EncoderLastAct, self).__init__()
        self._alpha = alpha

    def forward(self, x):
        return torch.minimum(
            torch.tensor(self._alpha), torch.maximum(torch.tensor(0.0), x)
        )


class ConvNetwork(nn.Module):
    """
    State encoding module
    -----------------------------------------
    1. Define each specific layer for encoder and decoder
    2. Use nn.Sequential in the end to sequentialize each networks
    """

    def __init__(
        self,
        state_dim: tuple,
        action_dim: int,
        encoder_conv_layers: list,
        decoder_conv_layers: list,
        fc_dim: int = 256,
        sf_dim: int = 256,
        activation: nn.Module = nn.ReLU(),
    ):
        super(ConvNetwork, self).__init__()

        s_dim, _, in_channels = state_dim

        # Activation functions
        self.act = activation

        ### Encoding module
        self.en_pmt = Permute((0, 3, 1, 2))

        # Define the fully connected layers
        flat_dim, output_shape = calculate_flatten_size(state_dim, encoder_conv_layers)
        reduced_feature_dim = output_shape[0]

        self.conv = nn.ModuleList()
        for layer in encoder_conv_layers:
            if layer["type"] == "conv":
                element = Conv(
                    in_channels=in_channels,
                    out_channels=layer["out_filters"],
                    kernel_size=layer["kernel_size"],
                    stride=layer["stride"],
                    padding=layer["padding"],
                    activation=layer["activation"],
                )
                in_channels = layer["out_filters"]

            elif layer["type"] == "pool":
                element = MaxPool2d(
                    kernel_size=layer["kernel_size"],
                    stride=layer["stride"],
                    padding=layer["padding"],
                    return_indices=True,
                )
            self.conv.append(element)

        #
        self.en_flatter = torch.nn.Flatten()

        #
        dummy_input = torch.zeros((state_dim)).unsqueeze(0)
        out = self.en_pmt(dummy_input)

        for fn in self.conv:
            out, _ = fn(out)
        out = self.en_flatter(out)

        feature_input_dim = out.shape[-1]
        self.en_feature = MLP(
            input_dim=feature_input_dim,  # agent pos concat
            hidden_dims=(feature_input_dim,),
            activation=self.act,
        )

        self.encoder = MLP(
            input_dim=feature_input_dim,  # agent pos concat
            hidden_dims=(fc_dim, fc_dim),
            output_dim=sf_dim,
            activation=self.act,
        )

        self.en_last_act = nn.Sigmoid()

        ### Decoding module
        # preprocess
        self.de_action = MLP(
            input_dim=action_dim, hidden_dims=(fc_dim,), activation=self.act
        )

        self.de_state_feature = MLP(
            input_dim=sf_dim,
            hidden_dims=(fc_dim,),
            activation=self.act,
        )

        # main decoding module
        self.de_concat = MLP(
            input_dim=2 * fc_dim,
            hidden_dims=(
                2 * fc_dim,
                flat_dim,
            ),
            activation=self.act,
        )

        self.reshape = Reshape(fc_dim, reduced_feature_dim)

        self.de_conv = nn.ModuleList()
        for layer in decoder_conv_layers:
            if layer["type"] == "conv_transpose":
                element = DeConv(
                    in_channels=in_channels,
                    out_channels=layer["in_filters"],
                    kernel_size=layer["kernel_size"],
                    stride=layer["stride"],
                    padding=layer["padding"],
                    output_padding=layer["output_padding"],
                    activation=layer["activation"],
                )
                in_channels = layer["in_filters"]
            elif layer["type"] == "conv":
                element = Conv(
                    in_channels=in_channels,
                    out_channels=layer["out_filters"],
                    kernel_size=layer["kernel_size"],
                    stride=layer["stride"],
                    padding=layer["padding"],
                    activation=layer["activation"],
                )
                in_channels = layer["out_filters"]
            elif layer["type"] == "pool":
                element = MaxUnpool2d(
                    kernel_size=layer["kernel_size"],
                    stride=layer["stride"],
                    padding=layer["padding"],
                )
            self.de_conv.append(element)

        self.de_last_act = nn.Identity()

        self.de_pmt = Permute((0, 2, 3, 1))

    def pre_grad_cam(self, x: torch.Tensor):
        """
        For grad-cam to visualize the feature activation
        """
        out = self.en_pmt(x)
        for fn in self.conv:
            out, _ = fn(out)
        return out

    def post_grad_cam(self, state: torch.Tensor):
        """
        For grad-cam to visualize the feature activation
        """
        out = self.en_flatter(state)
        out = self.en_feature(out)
        out = self.encoder(out)
        out = self.en_last_act(out)
        return out

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool | None = None,
    ):
        """forward eats in observation to output the feature vector

        Args:
            state (torch.Tensor): 1D or 2d state of the environment
            deterministic (bool, optional): Not used but here exists for code consistency

        Returns:
            feature: latent representations of the given state
        """
        # forward method
        indices = []
        sizes = []

        out = self.en_pmt(state)

        for fn in self.conv:
            output_dim = out.shape
            out, info = fn(out)
            if isinstance(fn, nn.MaxPool2d):
                indices.append(info)
                sizes.append(output_dim)

        out = self.en_flatter(out)
        out = self.en_feature(out)
        out = self.encoder(out)
        features = self.en_last_act(out)
        return features, {
            "indices": indices,
            "output_dim": sizes,
            "loss": torch.tensor(0.0),
        }

    def decode(self, features, actions, conv_dict):
        """This reconstruct full state given phi_state and actions"""

        indices = conv_dict["indices"][::-1]  # indices should be backward
        output_dim = conv_dict["output_dim"][::-1]  # to keep dim correct

        features = self.de_state_feature(features)
        actions = self.de_action(actions)

        out = torch.cat((features, actions), axis=-1)
        out = self.de_concat(out)
        out = self.reshape(out)

        i = 0
        for fn in self.de_conv:
            if isinstance(fn, nn.MaxUnpool2d):
                out = fn(out, indices[i], output_size=output_dim[i])
                i += 1
            else:
                out, _ = fn(out)
        out = self.de_last_act(out)
        reconstructed_state = self.de_pmt(out)
        return reconstructed_state


class AutoEncoder(nn.Module):
    """
    State encoding module
    -----------------------------------------
    1. Define each specific layer for encoder and decoder
    2. Use nn.Sequential in the end to sequentialize each networks
    """

    def __init__(
        self,
        state_dim: tuple,
        action_dim: int,
        fc_dim: int = 256,
        sf_dim: int = 256,
        activation: nn.Module = nn.ELU(),
    ):
        super(AutoEncoder, self).__init__()

        first_dim: int
        second_dim: int
        in_channels: int
        if len(state_dim) == 3:
            first_dim, second_dim, in_channels = state_dim
        elif len(state_dim) == 1:
            first_dim = state_dim[0]
            second_dim = 1
            in_channels = 1
        else:
            raise ValueError("State dimension is not correct.")

        input_dim = int(first_dim * second_dim * in_channels)

        # Parameters
        self.logstd_range = (-10, 2)

        # Activation functions
        self.act = activation

        ### Encoding module
        self.flatter = nn.Flatten()

        self.encoder = MLP(
            input_dim=input_dim,
            hidden_dims=(fc_dim, fc_dim, fc_dim),
            output_dim=sf_dim,
            activation=self.act,
        )

        self.encoder_filter = nn.ReLU()

        # self.encoder = nn.Sequential(self.flatter, self.tensorEmbed, self.en_vae)
        self.encoder = nn.Sequential(self.flatter, self.encoder, self.encoder_filter)

        ### Decoding module
        self.de_latent = MLP(
            input_dim=sf_dim,
            hidden_dims=(int(fc_dim / 2),),
            activation=self.act,
        )

        self.de_action = MLP(
            input_dim=action_dim,
            hidden_dims=(int(fc_dim / 2),),
            activation=self.act,
        )

        self.concat = MLP(
            input_dim=fc_dim,
            hidden_dims=(fc_dim,),
            activation=self.act,
        )

        self.decoder = MLP(
            input_dim=fc_dim,
            hidden_dims=(fc_dim, fc_dim, fc_dim),
            output_dim=input_dim,
            activation=self.act,
        )

        self.decoder = nn.Sequential(self.concat, self.decoder)

    def forward(self, state: torch.Tensor, deterministic: bool = True):
        """
        Input = x: 1D or 2D tensor arrays
        """
        out = self.flatter(state)
        phi = self.encoder(out)

        return phi, {}

    def decode(self, phi: torch.Tensor, actions: torch.Tensor):
        """This reconstruct full state given phi_state and actions"""
        out1 = self.de_latent(phi)
        out2 = self.de_action(actions)

        out = torch.cat((out1, out2), axis=-1)
        reconstructed_state = self.decoder(out)
        return reconstructed_state
