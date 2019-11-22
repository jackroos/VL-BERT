import numpy as np
from .eval_metric import EvalMetric
import torch

class CompositeEvalMetric(EvalMetric):
    """Manages multiple evaluation metrics.
    Args:
        metrics (list of EvalMetric): List of child metrics.
        name (str): Name of this metric instance for display.
    """

    def __init__(self, metrics=None, name='composite'):
        super(CompositeEvalMetric, self).__init__(name)
        if metrics is None:
            metrics = []
        self.metrics = metrics

    def add(self, metric):
        """Adds a child metric.
        Args:
            metric (EvalMetric): A metric instance.
        """
        self.metrics.append(metric)

    def get_metric(self, index):
        """Returns a child metric.
        Args:
            index (int): Index of child metric in the list of metrics.
        """
        try:
            return self.metrics[index]
        except IndexError:
            return ValueError("Metric index {} is out of range 0 and {}".format(
                index, len(self.metrics)))

    def update(self, outputs):
        """Updates the internal evaluation result.
        Args:
            labels (dict of `NDArray`): The labels of the data.
            preds (dict of `NDArray`): Predicted values.
        """
        for metric in self.metrics:
            metric.update(outputs)

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        try:
            for metric in self.metrics:
                metric.reset()
        except AttributeError:
            pass

    def get(self):
        """Returns the current evaluation result.
        Returns:
            names (list of str): Name of the metrics.
            values (list of float): Value of the evaluations.
        """
        names = []
        values = []
        for metric in self.metrics:
            name, value = metric.get()
            if isinstance(name, str):
                name = [name]
            if isinstance(value, (float, int, np.generic,torch.Tensor)):
                value = [value]
            names.extend(name)
            values.extend(value)
        return names, values
