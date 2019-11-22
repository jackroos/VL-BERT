import torch
from .eval_metric import EvalMetric


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


class RelationshipAccuracy(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(RelationshipAccuracy, self).__init__('RelAcc', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            logits = outputs['relationship_logits']
            label = outputs['relationship_label']
            self.sum_metric += float((logits.argmax(dim=1) == label).sum().item())
            self.num_inst += logits.shape[0]


class MLMAccuracy(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(MLMAccuracy, self).__init__('MLMAcc', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            logits = outputs['mlm_logits']
            label = outputs['mlm_label']
            keep = (label != -1)
            if keep.sum() > 0:
                self.sum_metric += float((logits[keep].argmax(dim=1) == label[keep]).sum().item())
                self.num_inst += keep.sum().item()


class MLMAccuracyWVC(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(MLMAccuracyWVC, self).__init__('MLMAccWVC', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            logits = outputs['mlm_logits_wvc']
            label = outputs['mlm_label_wvc']
            keep = (label != -1)
            if keep.sum() > 0:
                self.sum_metric += float((logits[keep].argmax(dim=1) == label[keep]).sum().item())
                self.num_inst += keep.sum().item()


class MLMAccuracyAUX(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(MLMAccuracyAUX, self).__init__('MLMAccAUX', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            logits = outputs['mlm_logits_aux']
            label = outputs['mlm_label_aux']
            keep = (label != -1)
            if keep.sum() > 0:
                self.sum_metric += float((logits[keep].argmax(dim=1) == label[keep]).sum().item())
                self.num_inst += keep.sum().item()


class MVRCAccuracy(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(MVRCAccuracy, self).__init__('MVRCAccuracy', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            logits = outputs['mvrc_logits']
            label = outputs['mvrc_label']
            keep = (label.sum(2) - 1.0).abs() < 0.1
            if keep.sum() > 0:
                self.sum_metric += float((logits[keep].argmax(dim=1) == label[keep].argmax(dim=1)).sum().item())
                self.num_inst += keep.sum().item()




