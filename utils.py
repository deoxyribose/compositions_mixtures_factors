import torch

def rel_err(true_parameter,estimated_parameter):
    if type(true_parameter) != torch.Tensor:
        true_parameter = torch.Tensor(true_parameter)
    norm = torch.norm
    return norm(true_parameter-estimated_parameter)/norm(true_parameter)