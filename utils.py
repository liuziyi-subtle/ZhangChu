# Copyright (c) Liu Ziyi.
# Licensed under the MIT license.

from datetime import datetime
import time
import sys
import logging
from io import TextIOBase


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)  # 选择概率最大的k个
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)  # 转换为对应类别
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # print("len(output): ", len(output), "len(target): ", len(target))
    # print(correct)

    res = dict()
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res["acc{}".format(k)] = correct_k.mul_(1.0 / batch_size).item()
    return res


""" 此部分改写自nni的common中的log部分 """
log_level_map = {
    'fatal': logging.FATAL,
    'error': logging.ERROR,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG
}

_time_format = '%m/%d/%Y, %I:%M:%S %p'


class _LoggerFileWrapper(TextIOBase):
    def __init__(self, logger_file):
        self.file = logger_file

    def write(self, s):
        if s != '\n':
            cur_time = datetime.now().strftime(_time_format)
            self.file.write('[{}] PRINT '.format(cur_time) + s + '\n')
            self.file.flush()
        return len(s)


def init_logger(log_filepath, log_level_name='info'):
    """Initialize logger.
    This will redirect anything from logging.getLogger() as 
    well as stdout to specified file. logger_file_path: path 
    of logger file (path-like object).
    """
    log_level = log_level_map.get(log_level_name, logging.INFO)
    log_file = open(log_filepath, 'w')
    fmt = '[%(asctime)s] %(levelname)s (%(name)s/%(threadName)s) %(message)s'
    logging.Formatter.converter = time.localtime
    formatter = logging.Formatter(fmt, _time_format)
    handler = logging.StreamHandler(log_file)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()  # nni中是使用的root
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    # 继承TextIOBase并重写write()函数
    sys.stdout = _LoggerFileWrapper(log_file)


def init_standalone_logger(logger_name):
    """
    Initialize root logger for standalone mode.
    This will set NNI's log level to INFO and print its log to stdout.
    """
    fmt = '[%(asctime)s] %(levelname)s (%(name)s) %(message)s'
    formatter = logging.Formatter(fmt, _time_format)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    standalone_logger = logging.getLogger(logger_name)
    standalone_logger.addHandler(handler)
    standalone_logger.setLevel(logging.INFO)
    standalone_logger.propagate = False

    # Following line does not affect NNI loggers, but without this user's logger won't be able to
    # print log even it's level is set to INFO, so we do it for user's convenience.
    # If this causes any issue in future, remove it and use `logging.info` instead of
    # `logging.getLogger('xxx')` in all examples.
    logging.basicConfig()


def plot_matrix(mat, axis, title):
    rows, cols = mat.shape
    x = np.expand_dims(range(cols), axis=1).repeat(rows, axis=1)
    y = mat.T
    axis.plot(x, y)
    axis.set_title(title)


def trim_axs(axs, N):
    """
    Reduce *axs* to *N* Axes. All further Axes are removed from the figure.
    """
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]
