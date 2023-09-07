import argparse
import torch

def parser_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--epoch",default=40)
    parser.add_argument("--batchsize",default=16)
    parser.add_argument("--lr",default=0.1)
    parser.add_argument("--momentum",default=0.01)
    parser.add_argument("--dataset",default="mnist")
    parser.add_argument("--num_user",default=20)
    parser.add_argument("--local_round",default=1)
    parser.add_argument("--frac",default=0.5)
    parser.add_argument("--bits",default=20)
    parser.add_argument("--ID",default=["user1","user2"])
    parser.add_argument("--scale",default=10000)
    parser.add_argument("--flag",default=0)
    parser.add_argument("--layer",default="conv2.weight")
    parser.add_argument("--device",default=1)
    parser.add_argument("--rate",default=0.3)
    parser.add_argument("--droptype",default=0)
    parser.add_argument("--iid",default=0)
    parser.add_argument("--dp_mechanism",default='Gaussian')
    parser.add_argument("--dp_epsilon",default=100)
    parser.add_argument("--dp_delta",default=1e-5)
    parser.add_argument("--dp_clip",default=20)

    args=parser.parse_args()
    return args
    
