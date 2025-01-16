import torch
import numpy as np


def get_flat_params_from(model):
    # pdb.set_trace()
    params = []
    for param in model.parameters():
        params.append(param.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind : prev_ind + flat_size].view(param.size())
        )
        prev_ind += flat_size


def get_flat_grad_from(inputs, grad_grad=False):
    grads = []
    for param in inputs:
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            if param.grad is None:
                grads.append(torch.zeros(param.view(-1).shape))
            else:
                grads.append(param.grad.view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad
