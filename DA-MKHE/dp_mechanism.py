import numpy as np
import torch

def cal_sensitivity(lr, clip, dataset_size):
#     return 2 * lr * clip / dataset_size
    return 2 * lr * clip

def Laplace(epsilon, sensitivity, size):
    noise_scale = sensitivity / epsilon
    return np.random.laplace(0, scale=noise_scale, size=size)

def Gaussian_Simple(epsilon, delta, sensitivity, size):
    noise_scale = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
    return np.random.normal(0, noise_scale, size=size)


 
def clip_gradients(net,args):
        if args.dp_mechanism == 'Laplace':
            # Laplace use 1 norm
            for k, v in net.named_parameters():
                v.grad /= max(1, v.grad.norm(1) / args.dp_clip)
        elif args.dp_mechanism == 'Gaussian':
            # Gaussian use 2 norm
            for k, v in net.named_parameters():
                v.grad /= max(1, v.grad.norm(2) / args.dp_clip)

def add_noise(net,args,idxs):
    sensitivity = cal_sensitivity(args.lr, args.dp_clip, len(idxs))
    if args.dp_mechanism == 'Laplace':
        with torch.no_grad():
            for k, v in net.named_parameters():
                noise = Laplace(epsilon=args.dp_epsilon, sensitivity=sensitivity, size=v.shape)
                noise = torch.from_numpy(noise).to(torch.device("cuda" if torch.cuda.is_available()==args.device else "cpu"))
                v += noise
    elif args.dp_mechanism == 'Gaussian':
        with torch.no_grad():
            for k, v in net.named_parameters():
                noise = Gaussian_Simple(epsilon=args.dp_epsilon, delta=args.dp_delta, sensitivity=sensitivity, size=v.shape)
                noise = torch.from_numpy(noise).to(torch.device("cuda" if torch.cuda.is_available()==args.device else "cpu"))
                v += noise
