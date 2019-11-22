# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Yuwen Xiong
# Modified by Dazhi Cheng
# --------------------------------------------------------

import time
import logging
import os


class Speedometer(object):
    def __init__(self, batch_size, frequent=50,
                 batches_per_epoch=None, epochs=None):
        self.batch_size = batch_size
        self.frequent = frequent
        self.batches_per_epoch = batches_per_epoch
        self.epochs = epochs
        self.epoch = -1
        self.init = False
        self.tic = 0
        self.last_count = 0
        self.data_in_time = 0.0
        self.data_transfer_time = 0.0
        self.forward_time = 0.0
        self.backward_time = 0.0
        self.optimizer_time = 0.0
        self.metric_time = 0.0

    def __call__(self, param):
        """Callback to Show speed."""
        count = param.nbatch
        if self.last_count > count:
            self.init = False
        self.last_count = count
        self.data_in_time += param.data_in_time
        self.data_transfer_time += param.data_transfer_time
        self.forward_time += param.forward_time
        self.backward_time += param.backward_time
        self.optimizer_time += param.optimizer_time
        self.metric_time += param.metric_time

        if self.init:
            if count % self.frequent == 0:
                speed = self.frequent * self.batch_size / (time.time() - self.tic)
                data_in_time = self.data_in_time / self.frequent
                data_transfer_time = self.data_transfer_time / self.frequent
                forward_time = self.forward_time / self.frequent
                backward_time = self.backward_time / self.frequent
                optimizer_time = self.optimizer_time / self.frequent
                metric_time = self.metric_time / self.frequent
                eta = ((self.epochs - self.epoch - 1) * self.batches_per_epoch + self.batches_per_epoch - param.nbatch) \
                    * self.batch_size / speed
                eta = int(eta / 60.0)
                eta_m = eta % 60
                eta_h = int((eta - eta_m) / 60) % 24
                eta_d = int((eta - eta_m - eta_h * 60) / (24 * 60))
                s = ''
                if param.eval_metric is not None:
                    prefix = "Epoch[%d] Batch [%d]\t" % (param.epoch, count)
                    name, value = param.eval_metric.get()
                    s = prefix + "Speed: %.2f samples/s ETA: %d d %2d h %2d m\tData: %.3f Tran: %.3f F: %.3f B: %.3f O: %.3f M: %.3f\tTrain-" \
                        % (speed, eta_d, eta_h, eta_m, data_in_time, data_transfer_time, forward_time, backward_time, optimizer_time, metric_time)
                    for n, v in zip(name, value):
                        s += "%s=%f,\t" % (n, v)
                else:
                    prefix = "Epoch[%d] Batch [%d]\t" % (param.epoch, count)
                    s = prefix + "Speed: %.2f ETA: %d d %2d h %2d m samples/s\tData: %.3f Tran: %.3f F: %.3f B: %.3f O: %.3f M: %.3f" \
                        % (speed, eta_d, eta_h, eta_m, data_in_time, data_transfer_time, forward_time, backward_time, optimizer_time, metric_time)

                if param.rank is not None:
                    s = 'Rank[%3d]' % param.rank + s

                logging.info(s)
                print(s)
                self.tic = time.time()
                self.data_in_time = 0.0
                self.data_transfer_time = 0.0
                self.forward_time = 0.0
                self.backward_time = 0.0
                self.optimizer_time = 0.0
                self.metric_time = 0.0
        else:
            self.init = True
            self.epoch += 1
            if param.eval_metric is not None:
                name, value = param.eval_metric.get()
                s = "Epoch[%d] Batch [%d]\tSpeed: - samples/sec ETA: - d - h - m\tTrain-" % (param.epoch, 0)
                for n, v in zip(name, value):
                    s += "%s=%f,\t" % (n, v)
            else:
                s = "Epoch[%d] Batch [%d]\tSpeed: - samples/sec ETA: - d - h - m" % (param.epoch, 0)

            if param.rank is not None:
                s = 'Rank[%3d]' % param.rank + s

            logging.info(s)
            print(s)
            self.tic = time.time()
