import torch
import torch.distributed as distributed


class EvalMetric(object):
    """Base class for all evaluation metrics.
    .. note::
        This is a base class that provides common metric interfaces.
        One should not use this class directly, but instead create new metric
        classes that extend it.
    Args
        name (str): Name of this metric instance for display.
    """

    def __init__(self, name, allreduce=False, num_replicas=1, **kwargs):
        self.name = str(name)
        self.allreduce=allreduce
        self.num_replicas = num_replicas
        self._kwargs = kwargs
        self.reset()

    def __str__(self):
        return "EvalMetric: {}".format(dict(self.get_name_value()))

    def update(self, outputs):
        """Updates the internal evaluation result.
        Args
            labels (list of `NDArray`): The labels of the data.
            preds (list of `NDArray`): Predicted values.
        """
        raise NotImplementedError()

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.num_inst = torch.tensor(0.)
        self.sum_metric = torch.tensor(0.)

    def get(self):
        """Returns the current evaluation result.
        Returns:
            names (list of str): Name of the metrics.
            values (list of float): Value of the evaluations.
        """
        if self.num_inst.item() == 0:
            return (self.name, float('nan'))
        else:
            if self.allreduce:
                num_inst = self.num_inst.clone().cuda()
                sum_metric = self.sum_metric.clone().cuda()
                distributed.all_reduce(num_inst, op=distributed.ReduceOp.SUM)
                distributed.all_reduce(sum_metric, op=distributed.ReduceOp.SUM)
                metric_tensor = (sum_metric / num_inst).detach().cpu()
            else:
                metric_tensor = (self.sum_metric / self.num_inst).detach().cpu()

            return (self.name, metric_tensor.item())

    def get_name_value(self):
        """Returns zipped name and value pairs.
        Returns
            A (list of tuples): (name, value) tuple list.
        """
        name, value = self.get()
        if not isinstance(name, list):
            name = [name]
        if not isinstance(value, list):
            value = [value]
        return list(zip(name, value))
