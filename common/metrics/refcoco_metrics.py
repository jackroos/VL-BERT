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


class RefAccuracy(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(RefAccuracy, self).__init__('RefAcc', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            cls_logits = outputs['label_logits']
            label = outputs['label']
            bs, _ = cls_logits.shape
            batch_inds = torch.arange(bs, device=cls_logits.device)
            self.sum_metric += float((label[batch_inds, cls_logits.argmax(1)] > 0.5).sum().item())
            self.num_inst += cls_logits.shape[0]


class ClsAccuracy(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(ClsAccuracy, self).__init__('ClsAcc', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            cls_logits = outputs['label_logits']
            cls_pred = (cls_logits > 0).long()
            label = outputs['label'].long()
            keep = (label >= 0)
            self.sum_metric += float((cls_pred[keep] == label[keep]).sum().item())
            self.num_inst += keep.sum().item()


class ClsPosAccuracy(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(ClsPosAccuracy, self).__init__('ClsPosAcc', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            cls_logits = outputs['label_logits']
            cls_pred = (cls_logits > 0).long()
            label = outputs['label'].long()
            keep = (label == 1)
            self.sum_metric += float((cls_pred[keep] == label[keep]).sum().item())
            self.num_inst += keep.sum().item()


class ClsPosFraction(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(ClsPosFraction, self).__init__('ClsPosFrac', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            label = outputs['label'].long()
            num_pos = (label == 1).sum().item()
            num_valid = (label >= 0).sum().item()
            self.sum_metric += float(num_pos)
            self.num_inst += float(num_valid)






