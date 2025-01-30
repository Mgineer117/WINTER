import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os


class BasePolicy(nn.Module):
    def __init__(self):
        super(BasePolicy, self).__init__()

        # networks
        self.feaNet = None
        self.psiNet = None
        self.dcnNet = None
        self._options = None

        self.device = torch.device("cpu")

        # constants
        self._dtype = torch.float32

        # utils
        self.l1_loss = F.l1_loss
        self.mse_loss = F.mse_loss
        self.huber_loss = F.smooth_l1_loss
        self.mqe2D_loss = lambda x, y: torch.mean(
            torch.sum(torch.pow(x - y, 4), -1), axis=-1
        )
        self.mqe4D_loss = lambda x, y: torch.mean(
            torch.mean(
                torch.mean(torch.mean(torch.pow(x - y, 4), -1), axis=-1), axis=-1
            ),
            axis=0,
        )

        # self.multiply_weights = lambda x, y: torch.einsum(
        #     "naf,nf->na", x, y
        # )  # ~ [N, |A|]
        self.multiply_weights = lambda x, y: torch.sum(
            torch.mul(x, y), axis=-1, keepdim=True
        )

    def weighted_mse_loss(self, x, y):
        weights = torch.where(y != 0.0, 5.0, 1.0)
        return torch.mean(weights * torch.pow(y - x, 4))

    def compute_gradient_norm(self, models, names, device, dir="None", norm_type=2):
        grad_dict = {}
        for i, model in enumerate(models):
            if model is not None:
                total_norm = torch.tensor(0.0, device=device)
                try:
                    for param in model.parameters():
                        if (
                            param.grad is not None
                        ):  # Only consider parameters that have gradients
                            param_grad_norm = torch.norm(param.grad, p=norm_type)
                            total_norm += param_grad_norm**norm_type
                except:
                    try:
                        param_grad_norm = torch.norm(model.grad, p=norm_type)
                    except:
                        param_grad_norm = torch.tensor(0.0)
                    total_norm += param_grad_norm**norm_type

                total_norm = total_norm ** (1.0 / norm_type)
                grad_dict[dir + "/grad/" + names[i]] = total_norm.item()

        return grad_dict

    def compute_weight_norm(self, models, names, device, dir="None", norm_type=2):
        norm_dict = {}
        for i, model in enumerate(models):
            if model is not None:
                total_norm = torch.tensor(0.0, device=device)
                try:
                    for param in model.parameters():
                        param_norm = torch.norm(param, p=norm_type)
                        total_norm += param_norm**norm_type
                except:
                    param_norm = torch.norm(model, p=norm_type)
                    total_norm += param_norm**norm_type
                total_norm = total_norm ** (1.0 / norm_type)
                norm_dict[dir + "/weight/" + names[i]] = total_norm.item()

        return norm_dict

    def learn(self):
        pass

    def learnPsi(self):
        pass

    def discover_options(self):
        pass

    def split(self, x:torch.Tensor, num_reward_features: int):
        if len(x.shape) == 1:
            x_r, x_s = x[:num_reward_features], x[num_reward_features:]
        else:
            x_r, x_s = x[:, :num_reward_features], x[:, num_reward_features:]
        
        return x_r, x_s

    def get_features(self):
        pass

    def get_cumulative_features(self):
        pass
