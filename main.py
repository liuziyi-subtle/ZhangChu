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
from darts.model import CNN
# from nni.common import init_logger, init_standalone_logger
from utils import accuracy, init_logger, init_standalone_logger


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
    # os.makedirs("./logs", exist_ok=True)
    # log_filename = datetime.now().strftime("%Y-%m-%d_%I:%M:%S-%p")
    # init_logger("./logs/" + log_filename)
    # init_standalone_logger()

    # load data.
    dataset_train = datasets.get_dataset(
        args.data_path, name="seed", pretrain=True)

    # model
    # encoder = CandidateArc(200, 1, args.channels, 7, args.layers)
    # model = SimCLR()

    # optimizer & scheduler

    # train
