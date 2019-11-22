import logging
import shutil


class ValidationMonitor(object):
    def __init__(self, val_func, val_loader, metrics, host_metric_name='Acc', label_index_in_batch=-1):
        super(ValidationMonitor, self).__init__()
        self.val_func = val_func
        self.val_loader = val_loader
        self.metrics = metrics
        self.host_metric_name = host_metric_name
        self.best_epoch = -1
        self.best_val = -1.0
        self.label_index_in_batch = label_index_in_batch

    def state_dict(self):
        return {'best_epoch': self.best_epoch,
                'best_val': self.best_val}

    def load_state_dict(self, state_dict):
        assert 'best_epoch' in state_dict, 'miss key \'best_epoch\''
        assert 'best_val' in state_dict, 'miss key \'best_val\''
        self.best_epoch = state_dict['best_epoch']
        self.best_val = state_dict['best_val']

    def __call__(self, epoch_num, net, optimizer, writer):
        self.val_func(net, self.val_loader, self.metrics, self.label_index_in_batch)

        name, value = self.metrics.get()
        s = "Epoch[%d] \tVal-" % (epoch_num)
        for n, v in zip(name, value):
            if n == self.host_metric_name and v > self.best_val:
                self.best_epoch = epoch_num
                self.best_val = v
                logging.info('New Best Val {}: {}, Epoch: {}'.format(self.host_metric_name, self.best_val, self.best_epoch))
                print('New Best Val {}: {}, Epoch: {}'.format(self.host_metric_name, self.best_val, self.best_epoch))
            s += "%s=%f,\t" % (n, v)
            if writer is not None:
                writer.add_scalar(tag='Val-' + n,
                                  scalar_value=v,
                                  global_step=epoch_num + 1)
        logging.info(s)
        print(s)

        logging.info('Best Val {}: {}, Epoch: {}'.format(self.host_metric_name, self.best_val, self.best_epoch))
        print('Best Val {}: {}, Epoch: {}'.format(self.host_metric_name, self.best_val, self.best_epoch))





