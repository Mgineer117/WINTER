import numpy as np
from math import floor
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import MaxPool2d, MaxUnpool2d
from torch.distributions import MultivariateNormal
from utils.utils import calculate_flatten_size
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
        sf_dim: int,
        snac_split_ratio: int,
        encoder_conv_layers: list,
        decoder_conv_layers: list,
        fc_dim: int = 256,
        activation: nn.Module = nn.ReLU(),
    ):
        super(ConvNetwork, self).__init__()

        s_dim, _, in_channels = state_dim

        self.sf_r_dim = floor(sf_dim * snac_split_ratio)
        self.sf_s_dim = sf_dim - self.sf_r_dim

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
            output_dim=self.sf_r_dim + self.sf_s_dim,
            activation=self.act,
        )

        self.en_last_act = nn.Sigmoid()

        ### Decoding module
        # preprocess
        self.de_action = MLP(
            input_dim=action_dim, hidden_dims=(fc_dim,), activation=self.act
        )

        self.de_feature = MLP(
            input_dim=self.sf_s_dim,
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
        out = self.en_pmt(state)

        for fn in self.conv:
            out, info = fn(out)

        out = self.en_flatter(out)
        out = self.en_feature(out)
        out = self.encoder(out)
        features = self.en_last_act(out)
        return features, {}

    def decode(self, features, actions):
        """This reconstruct full state given phi_state and actions"""
        features = self.de_feature(features)
        actions = self.de_action(actions)

        out = torch.cat((features, actions), axis=-1)
        out = self.de_concat(out)
        out = self.reshape(out)

        for fn in self.de_conv:
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
        sf_dim: int,
        snac_split_ratio: int,
        fc_dim: int = 256,
        activation: nn.Module = nn.ELU(),
    ):
        super(AutoEncoder, self).__init__()

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
        self.sf_r_dim = floor(sf_dim * snac_split_ratio)
        self.sf_s_dim = sf_dim - self.sf_r_dim
        self.fc_dim = fc_dim

        self.logstd_range = (-5, 2)

        # Activation functions
        self.act = activation

        ### Encoding module
        self.flatter = nn.Flatten()

        self.encoder = MLP(
            input_dim=input_dim,
            hidden_dims=(fc_dim, fc_dim, fc_dim),
            output_dim=self.sf_r_dim + self.sf_s_dim,
            activation=self.act,
        )

        self.encoder_filter = nn.Sigmoid()

        # self.encoder = nn.Sequential(self.flatter, self.tensorEmbed, self.en_vae)
        self.encoder = nn.Sequential(self.flatter, self.encoder, self.encoder_filter)

        ### Decoding module
        self.de_latent = MLP(
            input_dim=self.sf_s_dim,
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


class VAE(nn.Module):
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
        sf_r_dim: int,
        sf_s_dim: int,
        fc_dim: int = 256,
        activation: nn.Module = nn.ReLU(),
    ):
        super(VAE, self).__init__()

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
        self.sf_r_dim = sf_r_dim
        self.sf_s_dim = sf_s_dim
        self.fc_dim = fc_dim

        self.logstd_range = (-5, 2)

        # Activation functions
        self.act = activation

        ### Encoding module
        self.flatter = nn.Flatten()

        self.en_vae = MLP(
            input_dim=input_dim,
            hidden_dims=(fc_dim, fc_dim, fc_dim),
            activation=self.act,
        )

        # self.encoder = nn.Sequential(self.flatter, self.tensorEmbed, self.en_vae)
        self.encoder = nn.Sequential(self.flatter, self.en_vae)

        self.mu = MLP(
            input_dim=fc_dim,
            hidden_dims=(sf_r_dim + sf_s_dim,),
            activation=self.act,
        )
        self.logstd = MLP(
            input_dim=fc_dim,
            hidden_dims=(sf_r_dim + sf_s_dim,),
            activation=nn.Softplus(),
        )

        ### Decoding module
        self.de_latent = MLP(
            input_dim=self.sf_s_dim,
            hidden_dims=(fc_dim,),
            activation=self.act,
        )

        self.de_action = MLP(
            input_dim=action_dim,
            hidden_dims=(fc_dim,),
            activation=self.act,
        )

        self.concat = MLP(
            input_dim=int(2 * fc_dim),
            hidden_dims=(fc_dim,),
            activation=self.act,
        )

        self.de_vae = MLP(
            input_dim=fc_dim,
            hidden_dims=(fc_dim, fc_dim, fc_dim),
            output_dim=input_dim,
            activation=self.act,
        )

        self.decoder = nn.Sequential(self.concat, self.de_vae)

    def forward(self, state: torch.Tensor, deterministic=True):
        """
        Input = x: 1D or 2D tensor arrays
        """
        # 1D -> 2D for consistency
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        out = self.flatter(state)
        out = self.encoder(out)

        if deterministic:
            feature = F.tanh(self.mu(out))
            kl_loss = torch.tensor(0.0).to(feature.device)
        else:
            mu = F.tanh(self.mu(out))
            logstd = torch.clamp(
                self.logstd(out),
                min=self.logstd_range[0],
                max=self.logstd_range[1],
            )

            std = torch.exp(logstd)
            cov = torch.diag_embed(std**2)

            dist = MultivariateNormal(loc=mu, covariance_matrix=cov)

            feature = dist.rsample()

            # feature = torch.concatenate((mu_reward, state_feature), axis=-1)

            # if self.sf_s_dim == 0:

            # mu_reward, mu_state = mu[:, : self.sf_r_dim], mu[:, self.sf_s_dim :]
            # logstd_reward, logstd_state = logstd[:, :dim_half], logstd[:, dim_half:]

            # state_std = torch.exp(logstd_state)
            # state_cov = torch.diag_embed(state_std**2)

            # dist = MultivariateNormal(loc=mu_state, covariance_matrix=state_cov)

            # state_feature = dist.rsample()
            # feature = torch.concatenate((mu_reward, state_feature), axis=-1)

            kl = -0.5 * torch.sum(1 + torch.log(std**2) - mu**2 - std**2, dim=-1)
            kl_loss = torch.mean(kl)

        return feature, {"loss": kl_loss}

    def decode(self, features, actions):
        """This reconstruct full state given phi_state and actions"""
        out2 = self.de_latent(features)
        out1 = self.de_action(actions)

        out = torch.cat((out1, out2), axis=-1)
        reconstructed_state = self.decoder(out)
        return reconstructed_state
