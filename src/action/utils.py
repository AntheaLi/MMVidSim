import os, sys
import torch
import collections


def dict_to_device(ob, device='cuda:0'):
    if isinstance(ob, collections.Mapping):
        # d = {}
        # for k, v in ob.items():
            # print(k, v)
            # d[k] = dict_to_gpu(v)
        return {k: dict_to_device(v, device) for k, v in ob.items()}
    elif isinstance(ob, str):
        return ob
    elif isinstance(ob, list):
        return ob
    else:
        return ob.to(device).float()
    

def update_dict(d_from, d_to):

    for k in d_from:
        d_to[k] += d_from[k]
    
    return d_to


def filter_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)



def save_checkpoint(models, model_names, dirname, epoch=None, prepend_epoch=False, optimizers=None, optimizer_names=None):
    
    if len(models) != len(model_names) or (optimizers is not None and len(optimizers) != len(optimizer_names)):
        raise ValueError('Number of models, model names, or optimizers does not match.')

    with torch.no_grad():
        for model, model_name in zip(models, model_names):
            filename = f'net_{model_name}.pth'
            if prepend_epoch:
                filename = f'{epoch}_' + filename
            torch.save(model.state_dict(), os.path.join(dirname, filename))

        if optimizers is not None:
            filename = 'checkpt.pth'
            if prepend_epoch:
                filename = f'{epoch}_' + filename
            checkpt = {'epoch': epoch}
            for opt, optimizer_name in zip(optimizers, optimizer_names):
                checkpt[f'opt_{optimizer_name}'] = opt.state_dict()
            torch.save(checkpt, os.path.join(dirname, filename))

def load_checkpoint(models, model_names, dirname, epoch=None, optimizers=None, optimizer_names=None, strict=True):
    if len(models) != len(model_names) or (optimizers is not None and len(optimizers) != len(optimizer_names)):
        raise ValueError('Number of models, model names, or optimizers does not match.')
    
    for model, model_name in zip(models, model_names):
        filename = f'net_{model_name}.pth'
        if epoch is not None:
            filename = f'{epoch}_' + filename
        model.load_state_dict(torch.load(os.path.join(dirname, filename)), strict=strict)

    start_epoch = 0
    if optimizers is not None:
        filename = os.path.join(dirname, 'checkpt.pth')
        if epoch is not None:
            filename = f'{epoch}_' + filename
        if os.path.exists(filename):
            checkpt = torch.load(filename)
            start_epoch = checkpt['epoch']
            for opt, optimizer_name in zip(optimizers, optimizer_names):
                opt.load_state_dict(checkpt[f'opt_{optimizer_name}'])
            print(f'resuming from checkpoint {filename}')
        else:
            response = input(f'Checkpoint {filename} not found for resuming, refine saved models instead? (y/n) ')
            if response != 'y':
                sys.exit()

    return start_epoch