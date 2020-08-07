# Copyright (c) Liu Ziyi.
# Licensed under the MIT license.

import os
import logging
from argparse import ArgumentParser
from datetime import datetime

import torch
import torch.nn as nn
from nni.nas.pytorch.callbacks import ArchitectureCheckpoint, LRSchedulerCallback
from nni.nas.pytorch.darts import DartsTrainer

import datasets
from model import CNN
from nni.common import init_logger, init_standalone_logger
from utils import accuracy


""" Usage
local(cpu): python3 search.py --data-path \
    /Users/liuziyi/Workspace/with-zhangchu/soybean-and-cotton/df_records_cotton_self_supervised.csv
cloud(gpu): python3 search.py --data-path \
    /data/data/with-zhangchu/results/df_records_cotton_self_supervised.csv
"""
if __name__ == "__main__":
    parser = ArgumentParser("darts")
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--layers", default=3, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--log-frequency", default=10, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--channels", default=16, type=int)
    parser.add_argument("--unrolled", default=False, action="store_true")
    parser.add_argument("--visualization", default=False, action="store_true")
    args = parser.parse_args()

    # Init logger
    os.makedirs("./logs", exist_ok=True)
    log_filename = datetime.now().strftime("%Y-%m-%d_%I:%M:%S-%p")
    init_logger("./logs/" + log_filename)
    init_standalone_logger()

    # Data Loader
    dataset_train, dataset_valid = datasets.get_dataset(args.data_path)
    # dataset_train.data = dataset_train.data[:128, :]
    # dataset_valid.data = dataset_valid.data[:128, :]

    model = CNN(200, 1, args.channels, 7, args.layers)
    # criterion = nn.CrossEntropyLoss()

    criterion =

    optim = torch.optim.SGD(model.parameters(), 0.001,
                            momentum=0.9, weight_decay=3.0E-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, args.epochs, eta_min=0.001)

    trainer = DartsTrainer(model,
                           loss=criterion,
                           metrics=lambda output, target: accuracy(
                               output, target, topk=(1,)),
                           optimizer=optim,
                           num_epochs=args.epochs,
                           dataset_train=dataset_train,
                           dataset_valid=dataset_valid,
                           batch_size=args.batch_size,
                           log_frequency=args.log_frequency,
                           unrolled=args.unrolled,
                           callbacks=[LRSchedulerCallback(lr_scheduler), ArchitectureCheckpoint("./checkpoints")])
    if args.visualization:
        trainer.enable_visualization()
    trainer.train()
