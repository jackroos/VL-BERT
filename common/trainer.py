import os
import time
from collections import namedtuple
import torch

try:
    from apex import amp
    from apex.amp import _amp_state
except ImportError:
    pass
    #raise ImportError("Please install apex from https://www.github.com/nvidia/apex if you want to use fp16.")

# Parameter to pass to batch_end_callback
BatchEndParam = namedtuple('BatchEndParams',
                           ['epoch',
                            'nbatch',
                            'rank',
                            'add_step',
                            'data_in_time',
                            'data_transfer_time',
                            'forward_time',
                            'backward_time',
                            'optimizer_time',
                            'metric_time',
                            'eval_metric',
                            'locals'])


def _multiple_callbacks(callbacks, *args, **kwargs):
    """Sends args and kwargs to any configured callbacks.
    This handles the cases where the 'callbacks' variable
    is ``None``, a single function, or a list.
    """
    if isinstance(callbacks, list):
        for cb in callbacks:
            cb(*args, **kwargs)
        return
    if callbacks:
        callbacks(*args, **kwargs)


def to_cuda(batch):
    batch = list(batch)

    for i in range(len(batch)):
        if isinstance(batch[i], torch.Tensor):
            batch[i] = batch[i].cuda(non_blocking=True)
        elif isinstance(batch[i], list):
            for j, o in enumerate(batch[i]):
                if isinstance(batch[i], torch.Tensor):
                    batch[i][j] = o.cuda(non_blocking=True)

    return batch


def train(net,
          optimizer,
          lr_scheduler,
          train_loader,
          train_sampler,
          metrics,
          begin_epoch,
          end_epoch,
          logger,
          rank=None,
          batch_end_callbacks=None,
          epoch_end_callbacks=None,
          writer=None,
          validation_monitor=None,
          fp16=False,
          clip_grad_norm=-1,
          gradient_accumulate_steps=1):

    assert isinstance(gradient_accumulate_steps, int) and gradient_accumulate_steps >= 1

    for epoch in range(begin_epoch, end_epoch):
        print('PROGRESS: %.2f%%' % (100.0 * epoch / end_epoch))

        # set epoch as random seed of sampler while distributed training
        if train_sampler is not None and hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)

        # reset metrics
        metrics.reset()

        # set net to train mode
        net.train()

        # clear the paramter gradients
        # optimizer.zero_grad()

        # init end time
        end_time = time.time()

        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            name, value = validation_monitor.metrics.get()
            val = value[name.index(validation_monitor.host_metric_name)]
            lr_scheduler.step(val, epoch)

        # training
        for nbatch, batch in enumerate(train_loader):
            global_steps = len(train_loader) * epoch + nbatch
            os.environ['global_steps'] = str(global_steps)

            # record time
            data_in_time = time.time() - end_time

            # transfer data to GPU
            data_transfer_time = time.time()
            batch = to_cuda(batch)
            data_transfer_time = time.time() - data_transfer_time

            # forward
            forward_time = time.time()
            outputs, loss = net(*batch)
            loss = loss.mean()
            if gradient_accumulate_steps > 1:
                loss = loss / gradient_accumulate_steps
            forward_time = time.time() - forward_time

            # backward
            backward_time = time.time()
            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            backward_time = time.time() - backward_time

            optimizer_time = time.time()
            if (global_steps + 1) % gradient_accumulate_steps == 0:
                # step LR scheduler
                if lr_scheduler is not None and not isinstance(lr_scheduler,
                                                               torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step()

                # clip gradient
                if clip_grad_norm > 0:
                    if fp16:
                        total_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer),
                                                                    clip_grad_norm)
                    else:
                        total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(),
                                                                    clip_grad_norm)
                    if writer is not None:
                        writer.add_scalar(tag='grad-para/Total-Norm',
                                          scalar_value=float(total_norm),
                                          global_step=global_steps)

                optimizer.step()
                # clear the parameter gradients
                optimizer.zero_grad()
            optimizer_time = time.time() - optimizer_time

            # update metric
            metric_time = time.time()
            metrics.update(outputs)
            if writer is not None:
                with torch.no_grad():
                    for group_i, param_group in enumerate(optimizer.param_groups):
                        writer.add_scalar(tag='Initial-LR/Group_{}'.format(group_i),
                                          scalar_value=param_group['initial_lr'],
                                          global_step=global_steps)
                        writer.add_scalar(tag='LR/Group_{}'.format(group_i),
                                          scalar_value=param_group['lr'],
                                          global_step=global_steps)
                    writer.add_scalar(tag='Train-Loss',
                                      scalar_value=float(loss.item()),
                                      global_step=global_steps)
                    name, value = metrics.get()
                    for n, v in zip(name, value):
                        writer.add_scalar(tag='Train-' + n,
                                          scalar_value=v,
                                          global_step=global_steps)

            metric_time = time.time() - metric_time

            # execute batch_end_callbacks
            if batch_end_callbacks is not None:
                batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch, add_step=True, rank=rank,
                                                 data_in_time=data_in_time, data_transfer_time=data_transfer_time,
                                                 forward_time=forward_time, backward_time=backward_time,
                                                 optimizer_time=optimizer_time, metric_time=metric_time,
                                                 eval_metric=metrics, locals=locals())
                _multiple_callbacks(batch_end_callbacks, batch_end_params)

            # update end time
            end_time = time.time()

        # excute epoch_end_callbacks
        if validation_monitor is not None:
            validation_monitor(epoch, net, optimizer, writer)
        if epoch_end_callbacks is not None:
            _multiple_callbacks(epoch_end_callbacks, epoch, net, optimizer, writer, validation_monitor=validation_monitor)


