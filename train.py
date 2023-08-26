#!/bin/env python

import model
import torch
import torch.nn as nn
import argparse as arg

prase = arg.ArgumentParser(
    prog="train",
    description="T@V",
    epilog="Example: train --model ./model.pk --dataset ./data/prompt ./data/svg" 
)
prase.add_argument(
    "--model",
    action="store",
    default="./model.pk",
    type=str,
    help="select a model"
)
prase.add_argument(
    "--dataset",
    metavar=("X","Y"),
    nargs=2,
    required=True,
    help="add dataset"
)

prase.add_argument(
    "--epoch",
    action="store",
    default=100,
    type=int,
    help="epoch",
)
gpu_groups = prase.add_mutually_exclusive_group(required=True)
gpu_groups.add_argument("-cpu",action="store_true")
gpu_groups.add_argument("-gpu",action="store_true")


args = prase.parse_args()
file,dataset,epoch = args.model,args.dataset,args.epoch
device = "cuda" if args.gpu else "cpu"

if args.gpu:
    print("Using GPU ")
else:
    print("Using CPU ")

print("Model : ",file)


NLP_FL = model.NET().to(device=device)


for i in range(epoch):
    print(i)