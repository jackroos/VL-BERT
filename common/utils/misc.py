import os
import numpy as np
import torch
import torch.nn.functional as F
import logging


def block_digonal_matrix(*blocks):
    """
    Construct block diagonal matrix
    :param blocks: blocks of block diagonal matrix
    :param device
    :param dtype
    :return: block diagonal matrix
    """
    assert len(blocks) > 0
    rows = [block.shape[0] for block in blocks]
    cols = [block.shape[1] for block in blocks]
    out = torch.zeros((sum(rows), sum(cols)),
                      device=blocks[0].device,
                      dtype=blocks[0].dtype)
    cur_row = 0
    cur_col = 0
    for block, row, col in zip(blocks, rows, cols):
        out[cur_row:(cur_row + row), cur_col:(cur_col + col)] = block
        cur_row += row
        cur_col += col

    return out


def print_and_log(string, logger=None):
    print(string)
    if logger is None:
        logging.info(string)
    else:
        logger.info(string)


def summary_parameters(model, logger=None):
    """
    Summary Parameters of Model
    :param model: torch.nn.module_name
    :param logger: logger
    :return: None
    """

    print_and_log('>> Trainable Parameters:', logger)
    trainable_paramters = [(str(n), str(v.dtype), str(tuple(v.shape)), str(v.numel()))
                           for n, v in model.named_parameters() if v.requires_grad]
    max_lens = [max([len(item) + 4 for item in col]) for col in zip(*trainable_paramters)]
    raw_format = '|' + '|'.join(['{{:{}s}}'.format(max_len) for max_len in max_lens]) + '|'
    raw_split = '-' * (sum(max_lens) + len(max_lens) + 1)
    print_and_log(raw_split, logger)
    print_and_log(raw_format.format('Name', 'Dtype', 'Shape', '#Params'), logger)
    print_and_log(raw_split, logger)

    for name, dtype, shape, number in trainable_paramters:
        print_and_log(raw_format.format(name, dtype, shape, number), logger)
        print_and_log(raw_split, logger)

    num_trainable_params = sum([v.numel() for v in model.parameters() if v.requires_grad])
    total_params = sum([v.numel() for v in model.parameters()])
    non_trainable_params = total_params - num_trainable_params
    print_and_log('>> {:25s}\t{:.2f}\tM'.format('# TrainableParams:', num_trainable_params / (1.0 * 10 ** 6)), logger)
    print_and_log('>> {:25s}\t{:.2f}\tM'.format('# NonTrainableParams:', non_trainable_params / (1.0 * 10 ** 6)), logger)
    print_and_log('>> {:25s}\t{:.2f}\tM'.format('# TotalParams:', total_params / (1.0 * 10 ** 6)), logger)




def clip_grad(named_parameters, max_norm, logger=logging, std_verbose=False, log_verbose=False):
    """Clips gradient norm of an iterable of parameters.
    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.
    :param named_parameters: dict, named parameters of pytorch module
    :param max_norm: float or int, max norm of the gradients
    :param logger: logger to write verbose info
    :param std_verbose: verbose info in stdout
    :param log_verbose: verbose info in log

    :return Total norm of the parameters (viewed as a dict: param name -> param grad norm).
    """
    max_norm = float(max_norm)
    parameters = [(n, p) for n, p in named_parameters if p.grad is not None]
    total_norm = 0
    param_to_norm = {}
    param_to_shape = {}
    for n, p in parameters:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm ** 2
        param_to_norm[n] = param_norm
        param_to_shape[n] = tuple(p.size())
        if np.isnan(param_norm.item()):
            raise ValueError("the param {} was null.".format(n))

    total_norm = total_norm ** (1. / 2)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef.item() < 1:
        logger.info('---Clip grad! Total norm: {:.3f}, clip coef: {:.3f}.'.format(total_norm, clip_coef))
        for n, p in parameters:
            p.grad.data.mul_(clip_coef)

    if std_verbose:
        print('---Total norm {:.3f} clip coef {:.3f}-----------------'.format(total_norm, clip_coef))
        for name, norm in sorted(param_to_norm.items(), key=lambda x: -x[1]):
            print("{:<60s}: {:.3f}, ({}: {})".format(name, norm, np.prod(param_to_shape[name]), param_to_shape[name]))
        print('-------------------------------', flush=True)
    if log_verbose:
        logger.info('---Total norm {:.3f} clip coef {:.3f}-----------------'.format(total_norm, clip_coef))
        for name, norm in sorted(param_to_norm.items(), key=lambda x: -x[1]):
            logger.info("{:<60s}: {:.3f}, ({}: {})".format(name, norm, np.prod(param_to_shape[name]), param_to_shape[name]))
        logger.info('-------------------------------')

    return {name: norm.item() for name, norm in param_to_norm.items()}


def bn_fp16_half_eval(m):
    classname = str(m.__class__)
    if 'BatchNorm' in classname and (not m.training):
        m.half()


def soft_cross_entropy(input, target, reduction='mean'):
    """
    Cross entropy loss with input logits and soft target
    :param input: Tensor, size: (N, C)
    :param target: Tensor, size: (N, C)
    :param reduction: 'none' or 'mean' or 'sum', default: 'mean'
    :return: loss
    """
    eps = 1.0e-1
    # debug = False
    valid = (target.sum(1) - 1).abs() < eps
    # if debug:
    #     print('valid', valid.sum().item())
    #     print('all', valid.numel())
    #     print('non valid')
    #     print(target[valid == 0])
    if valid.sum().item() == 0:
        return input.new_zeros(())
    if reduction == 'mean':
        return (- F.log_softmax(input[valid], 1) * target[valid]).sum(1).mean(0)
    elif reduction == 'sum':
        return (- F.log_softmax(input[valid], 1) * target[valid]).sum()
    elif reduction == 'none':
        l = input.new_zeros((input.shape[0], ))
        l[valid] = (- F.log_softmax(input[valid], 1) * target[valid]).sum(1)
        return l
    else:
        raise ValueError('Not support reduction type: {}.'.format(reduction))






