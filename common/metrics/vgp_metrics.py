import torch
from .eval_metric import EvalMetric
import numpy as np


class LossLogger(EvalMetric):
    def __init__(self, output_name, display_name=None,
                 allreduce=False, num_replicas=1):
        self.output_name = output_name
        if display_name is None:
            display_name = output_name
        super(LossLogger, self).__init__(display_name, allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            if self.output_name in outputs:
                self.sum_metric += float(outputs[self.output_name].mean().item())
            self.num_inst += 1


class Accuracy(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(Accuracy, self).__init__('Acc', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            _filter = outputs['sentence_label'] != -1
            cls_logits = outputs['sentence_label_logits'][_filter]
            label = outputs['sentence_label'][_filter]
            if cls_logits.dim() == 2:
                cls_logits = cls_logits.view(-1)
                label = label.view(-1)
            self.sum_metric += float(((cls_logits > 0.5).float() == label.float()).sum().item())
            self.num_inst += cls_logits.shape[0]


class AlignmentAccuracy(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(AlignmentAccuracy, self).__init__('AlgnmtAcc', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            _filter = (outputs['sentence_label'] == 0)
            if _filter.sum() != 0:
                cls_logits = outputs['alignment_logits'][_filter]
                label = outputs['alignment_label'][_filter]
                if cls_logits.dim() == 2:
                    cls_logits = cls_logits.view(-1)
                    label = label.view(-1)
                self.sum_metric += float(((cls_logits > 0.5).float() == label.float()).sum().item())
                self.num_inst += cls_logits.shape[0]


def compute_metrics_sentence_level(metric, pred_probs, labels):
    if metric == "accuracy":
        pred_labels = np.zeros_like(labels)
        pred_labels[pred_probs >= 0.5] = 1
        result = (pred_labels == labels).mean()
    else:
        print("The metric {} has not been implemented".format(metric))
    return result
