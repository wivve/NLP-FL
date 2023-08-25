#!/bin/env python

import argparse as arg


prase = arg.ArgumentParser(
    prog="train",
    description="T@V",
    epilog="Example: train --model ./model.pk --dataset ./data/x ./data/y" 
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
gpu_groups = prase.add_mutually_exclusive_group(required=True)
gpu_groups.add_argument("-cpu",action="store_true")
gpu_groups.add_argument("-gpu",action="store_true")


args = prase.parse_args()
model,dataset = args.model,args.dataset
GPU = args.gpu

if args.cpu:
    print("Using CPU ")
else:
    print("Using GPU ")

print("Model : ",model)


