import torch
import torch.nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import logging

def to_numpy(tensor_data):
    return tensor_data.detach().to('cpu').data.numpy()

def param_count(model):
    total_param_size = 0
    for name, param in model.named_parameters():
        each_param_size = np.prod(param.size())
        total_param_size += each_param_size
    logging.info(f'total_param_size: {total_param_size}')
    return total_param_size

def init_parameters(model):
    total_param_size = 0
    for name, param in model.named_parameters():
        each_param_size = np.prod(param.size())
        total_param_size += each_param_size
        logging.info('{} {} {}'.format(name, param.size(), each_param_size))

        if param.requires_grad == True:
            if len(param.size()) > 1: # is weight or bias
                if 'conv' in name and name.endswith('.weight'):
                    torch.nn.init.kaiming_uniform_(param, mode='fan_out', nonlinearity='relu')
                elif '.bn' in name or '_bn' in name:
                    if name.endswith('.weight'):
                        torch.nn.init.constant(param, 1)
                elif 'bias' in name:
                    torch.nn.init.constant_(param, 0)
                else:
                    torch.nn.init.xavier_uniform_(param)
            else:
                if 'bias' in name:
                    param.data.zero_()
                else:
                    torch.nn.init.uniform_(param)

    logging.info(f'total_param_size: {total_param_size}')

def normalize_output(output_emb, embedding_norm):
    if embedding_norm == 'l2':
        norm = torch.norm(output_emb.detach(), p=2, dim=1, keepdim=True)
        output_norm = output_emb / norm
    elif embedding_norm == 'unit_range':
        norm = torch.norm(output_emb.detach(), p=2, dim=1, keepdim=True)
        div_norm = 1.0 / norm
        ones_norm = torch.ones_like(div_norm)
        scaler = torch.where(norm > 1.0, div_norm, ones_norm)
        output_norm = output_emb * scaler
    else: # none
        output_norm = output_emb
    return output_norm