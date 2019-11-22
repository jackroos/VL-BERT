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


class Accuracy(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(Accuracy, self).__init__('Acc', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            _filter = outputs['label'] != -1
            cls_logits = outputs['label_logits'][_filter]
            label = outputs['label'][_filter]
            if cls_logits.dim() == 1:
                cls_logits = cls_logits.view((-1, 4))
                label = label.view((-1, 4)).argmax(1)
            self.sum_metric += float((cls_logits.argmax(dim=1) == label).sum().item())
            self.num_inst += cls_logits.shape[0]


class AnsLoss(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(AnsLoss, self).__init__('AnsLoss', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            self.sum_metric += float(outputs['ans_loss'].mean().item())
            self.num_inst += 1


class CNNRegLoss(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(CNNRegLoss, self).__init__('CNNRegLoss', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            if 'cnn_regularization_loss' in outputs:
                self.sum_metric += float(outputs['cnn_regularization_loss'].mean().item())
            self.num_inst += 1


class PositiveFraction(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(PositiveFraction, self).__init__('PosFraction', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            self.sum_metric += float(outputs['positive_fraction'].mean().item())
            self.num_inst += 1


class JointAccuracy(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(JointAccuracy, self).__init__('JointAcc', allreduce, num_replicas)

    def update(self, outputs):
        a_cls_logits = outputs['answer_label_logits']
        a_label = outputs['answer_label']
        r_cls_logits = outputs['rationale_label_logits']
        r_label = outputs['rationale_label']
        self.sum_metric += float(((a_cls_logits.argmax(dim=1) == a_label)
                                  & (r_cls_logits.argmax(dim=1) == r_label)).sum().item())
        self.num_inst += a_cls_logits.shape[0]



