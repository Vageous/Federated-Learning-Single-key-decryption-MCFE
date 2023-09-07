import torch
from gmpy2 import mpz



def init(global_model):
        weight_model={}
        for name,params in global_model.state_dict().items():
            weight_model[name]=torch.zeros_like(params)
        return weight_model


def init_cipher(global_model,args):
    weight_model={}
    for name,params in global_model.state_dict().items():
        if name==args.layer:
            len=params.numel()
            weight_model[name]=[[mpz(1) for col in range(args.num_user+2)] for row in range(len)]
        else:
            weight_model[name]=torch.zeros_like(params)
    return weight_model

