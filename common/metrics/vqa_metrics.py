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


class SoftAccuracy(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(SoftAccuracy, self).__init__('SoftAcc', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            cls_logits = outputs['label_logits']
            label = outputs['label']
            bs, num_classes = cls_logits.shape
            batch_inds = torch.arange(bs, device=cls_logits.device)
            self.sum_metric += float(label[batch_inds, cls_logits.argmax(1)].sum().item())
            self.num_inst += cls_logits.shape[0]




