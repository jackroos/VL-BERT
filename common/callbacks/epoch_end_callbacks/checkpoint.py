import torch


class Checkpoint(object):
    def __init__(self, prefix, frequent):
        super(Checkpoint, self).__init__()
        self.prefix = prefix
        self.frequent = frequent

    def __call__(self, epoch_num, net, optimizer, writer, validation_monitor=None):
        if (epoch_num + 1) % self.frequent == 0:
            param_name = '{}-{:04d}.model'.format(self.prefix, epoch_num)
            checkpoint_dict = dict()
            checkpoint_dict['state_dict'] = net.state_dict()
            checkpoint_dict['optimizer'] = optimizer.state_dict()
            save_to_best = False
            if validation_monitor is not None:
                checkpoint_dict['validation_monitor'] = validation_monitor.state_dict()
                if validation_monitor.best_epoch == epoch_num:
                    save_to_best = True
            torch.save(checkpoint_dict, param_name)
            if save_to_best:
                best_param_name = '{}-best.model'.format(self.prefix)
                torch.save(checkpoint_dict, best_param_name)
                print('Save new best model to {}.'.format(best_param_name))
